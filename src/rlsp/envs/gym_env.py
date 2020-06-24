# -*- coding: utf-8 -*-
"""
Gym envs representing the coordination-simulation from
REAL NFV https://github.com/RealVNF/coordination-simulation


For help on "Implementing New Environments" see:
https://github.com/openai/gym/blob/master/gym/core.py
https://github.com/rll/rllab/blob/master/docs/user/implement_env.rst

"""
import inspect
import logging
from typing import Tuple
import gym
from gym.utils import seeding
import numpy as np
from rlsp.envs.environment_limits import EnvironmentLimits
from rlsp.envs.simulator_wrapper import SimulatorWrapper
from spinterface import SimulatorInterface, SimulatorState
from coordsim.reader.reader import read_network, get_sfc, get_sf, network_diameter

logger = logging.getLogger(__name__)


class GymEnv(gym.Env):
    """
    Gym Environment class, which abstracts the coordination simulator.
    """
    current_simulator_state: SimulatorState
    simulator: SimulatorInterface = ...
    simulator_wrapper: SimulatorWrapper = ...

    metadata = {'render.modes': ['human']}

    def __init__(self, agent_config, simulator, network_file, service_file, seed=None, sim_seed=None):
        self.network_file = network_file
        self.agent_config = agent_config
        self.simulator = simulator
        self.sim_seed = sim_seed
        self.simulator_wrapper = None
        self.current_simulator_state = None

        self.last_succ_flow = 0
        self.last_drop_flow = 0
        self.last_gen_flow = 0
        self.run_count = 0

        self.np_random = np.random.RandomState()
        self.seed(seed)

        self.network, _ = read_network(self.network_file)
        self.network_diameter = network_diameter(self.network)
        self.sfc_list = get_sfc(service_file)
        self.sf_list = get_sf(service_file)
        self.env_limits = EnvironmentLimits(len(self.network.nodes), self.sfc_list)
        self.min_delay, self.max_delay = self.min_max_delay()

        self.reset()

        self.action_space = self.env_limits.action_space
        self.observation_space = self.env_limits.observation_space

        # order of permutation for shuffling state
        self.permutation = None

    def min_max_delay(self):
        """Return the min and max e2e-delay for the current network topology and SFC. Independent of capacities."""
        vnf_delays = sum([sf['processing_delay_mean'] for sf in self.sf_list.values()])
        # min delay = sum of VNF delays (corresponds to all VNFs at ingress)
        min_delay = vnf_delays
        # max delay = VNF delays + num_vnfs * network diameter (corresponds to max distance between all VNFs)
        max_delay = vnf_delays + len(self.sf_list) * self.network_diameter
        logger.info(f"min_delay: {min_delay}, max_delay: {max_delay}, diameter: {self.network_diameter}")
        return min_delay, max_delay

    def reset(self):
        """
        Resets the state of the envs, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space.
        (Initial reward is assumed to be 0.)

        """
        if self.sim_seed is None:
            simulator_seed = self.np_random.randint(0, np.iinfo(np.int32).max, dtype=np.int32)
        else:
            simulator_seed = self.sim_seed
        logger.debug(f"Simulator seed is {simulator_seed}")
        self.simulator_wrapper = SimulatorWrapper(self.simulator, self.env_limits)

        self.last_succ_flow = 0
        self.last_drop_flow = 0
        self.last_gen_flow = 0
        self.run_count = 0

        # to get initial state and instantiate
        vectorized_state, self.current_simulator_state = self.simulator_wrapper.init(simulator_seed)

        # permute state and save permutation for reversing action later
        if self.agent_config['shuffle_nodes']:
            vectorized_state, permutation = self.simulator_wrapper.permute_node_order(vectorized_state)
            self.permutation = permutation

        return vectorized_state

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[object, float, bool, dict]:

        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether episode has ended, in which case further step calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        done = False
        self.run_count += 1
        logger.debug(f"Action array (NN output + noise, normalized): {action}")

        # reverse action order using permutation from previous state shuffle
        if self.agent_config['shuffle_nodes']:
            action = self.simulator_wrapper.reverse_node_permutation(action, self.permutation)
            self.permutation = None

        # apply reversed action, calculate reward
        vectorized_state, self.current_simulator_state = self.simulator_wrapper.apply(action)
        reward = self.calculate_reward(self.current_simulator_state)

        # then shuffle new state again and save new permutation
        if self.agent_config['shuffle_nodes']:
            vectorized_state, permutation = self.simulator_wrapper.permute_node_order(vectorized_state)
            self.permutation = permutation
        if self.run_count == 200:
            done = True
            self.run_count = 0

        logger.debug(f"NN input (observation): {vectorized_state}")
        return vectorized_state, reward, done, {}

    def render(self, mode='cli'):
        """Renders the envs.
        Implementation required by Gym.
        """
        assert mode in ['human']

    def reward_func_repr(self):
        """returns a string describing the reward function"""
        return inspect.getsource(self.calculate_reward)

    def delay_reward(self, delay, t_min=-1, t_max=1):
        """
        Calculate and return the normalized delay reward in range of [t_min, t_max].
        :param delay: The avg. e2e delay of successful flows. The lower, the better.
        :param t_min: Lower limit of the target range, ie, normalized delay reward. Default: -1
        :param t_max: Upper limit of the target range, ie, normalized delay reward. Default: +1
        :return: Normalized delay reward in range [t_min, t_max]

        New: Simplified and more effective delay function: Only divide by 1x net. diameter. +1
        """
        # assert min_delay != max_delay

        # if no flows are successful, delay = 0, which may be better than min_delay and lead to positive reward
        # --> set delay to min_delay in this case
        if delay == 0:
            delay = self.min_delay

        # new, simplified delay function
        delay_reward = ((self.min_delay - delay) / self.network_diameter) + t_max
        # cap the delay to be within the defined bounds [t_min,t_max]
        delay_reward = max([delay_reward, t_min])
        delay_reward = min([delay_reward, t_max])
        return delay_reward

    def flow_reward(self, cur_succ_flow, cur_drop_flow, cur_gen_flow):
        """Calculate and return the normalized flow reward in the range [-1, 1]
        :param succ_flow: Number of successful flows.
        :param drop_flow: Number of dropped flows.
        :param gen_flow: Number of generated flows.
        """
        # calculate flow_reward related to successful and dropped flows
        succ_flow = cur_succ_flow - self.last_succ_flow
        drop_flow = cur_drop_flow - self.last_drop_flow
        gen_flow = cur_gen_flow - self.last_gen_flow

        self.last_succ_flow = cur_succ_flow
        self.last_drop_flow = cur_drop_flow
        self.last_gen_flow = cur_gen_flow

        # flow_reward = succ_flow - drop_flow
        # normalized flow reward: divide by sum of successful and dropped flows -> range [-1, 1]
        if succ_flow + drop_flow > 0:
            flow_reward = (succ_flow - drop_flow) / (succ_flow + drop_flow)
        else:
            flow_reward = 0

        # Debug output
        logger.debug(f"Flows since last run: {gen_flow} total, {succ_flow} successful, {drop_flow} dropped")

        return flow_reward

    def calculate_reward(self, simulator_state: SimulatorState) -> float:
        """ The reward function calculates the reward based on the current network/simulator state

        This is a key part of the RL algorithm.

        Parameters
        ----------
        simulator_state (SimulatorState)

        Returns
        -------
        reward (float)
        """
        cur_succ_flow = simulator_state.network_stats['successful_flows']
        cur_drop_flow = simulator_state.network_stats['dropped_flows']
        cur_gen_flow = simulator_state.network_stats['total_flows']

        # flow reward
        flow_reward = self.flow_reward(cur_succ_flow, cur_drop_flow, cur_gen_flow)

        # delay reward
        delay = simulator_state.network_stats['run_avg_end2end_delay']
        delay_reward = self.delay_reward(delay)

        flow_reward_weight = self.agent_config['flow_reward_weight']
        delay_reward_weight = self.agent_config['delay_reward_weight']
        reward = flow_reward_weight * flow_reward + delay_reward_weight * delay_reward

        # debug output
        logger.debug(f"Number of runs (applied actions): {self.run_count}")
        logger.debug(f"Avg e2e delay: {delay}")
        logger.debug(f"Rewards: flow: {flow_reward}, delay: {delay_reward}, weighted total: {reward}")

        return reward
