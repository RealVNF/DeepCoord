from keras.layers import Concatenate, Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LambdaCallback
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rlsp.envs.action_norm_processor import ActionScheduleProcessor
from rl.random import GaussianWhiteNoiseProcess
from rlsp.agents.rlsp_agent import RLSPAgent
from rlsp.utils.util_functions import create_simulator
import copy
import csv
import logging
import os

logger = logging.getLogger(__name__)

EPISODE_REWARDS = {}


class DDPG(RLSPAgent):
    """
    RLSP DDPG Agent
    This class creates a DDPG agent with params for RLSP
    """
    def __init__(self, agent_helper):
        self.agent_helper = agent_helper
        self.create()
        pass

    def create(self):
        """Create the agent"""
        assert len(self.agent_helper.env.action_space.shape) == 1
        nb_actions = int(self.agent_helper.env.action_space.shape[0])

        # set #nodes and #sfs based on env limits. used for splitting the output layer and action processor
        num_nodes = self.agent_helper.env.env_limits.MAX_NODE_COUNT
        num_sfcs = self.agent_helper.env.env_limits.MAX_SF_CHAIN_COUNT
        num_sfs = self.agent_helper.env.env_limits.MAX_SERVICE_FUNCTION_COUNT

        # create the actor NN
        observation_input = Input(shape=(1,) + self.agent_helper.env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        prev_layer = flattened_observation
        # create hidden layers according to config
        for num_hidden in self.agent_helper.config['actor_hidden_layer_nodes']:
            hidden_layer = Dense(num_hidden,
                                 activation=self.agent_helper.config['actor_hidden_layer_activation'])(prev_layer)
            prev_layer = hidden_layer
        # split output layer into separate parts for each node and SF and apply softmax individually
        out_parts = [Dense(num_nodes, activation='softmax')(prev_layer) for _ in range(num_nodes * num_sfs)]
        out = Concatenate()(out_parts)
        # normal output layer
        # out = Dense(nb_actions, activation='tanh')(prev_layer)
        actor = Model(inputs=observation_input, outputs=out)

        # create the critic NN
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + self.agent_helper.env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        prev_layer = Concatenate()([action_input, flattened_observation])
        # create hidden layers according to config
        for num_hidden in self.agent_helper.config['critic_hidden_layer_nodes']:
            hidden_layer = Dense(num_hidden,
                                 activation=self.agent_helper.config['critic_hidden_layer_activation'])(prev_layer)
            prev_layer = hidden_layer
        out_critic = Dense(1, activation='linear')(prev_layer)
        critic = Model(inputs=[action_input, observation_input], outputs=out_critic)

        # write NN summary to string
        actor_summary_lst = []
        actor.summary(print_fn=actor_summary_lst.append)
        actor_summary = "".join(actor_summary_lst)
        actor.summary(print_fn=logger.debug)

        # write NN summary to string
        critic_summary_lst = []
        critic.summary(print_fn=critic_summary_lst.append)
        critic_summary = "".join(critic_summary_lst)
        critic.summary(print_fn=logger.debug)

        # This following line is causing aliasing issues. Ex: 'nb_observation' is added to agent_config
        self.agent_helper.result.agent_config = copy.copy(self.agent_helper.config)  # Set agent params in result file
        self.agent_helper.result.agent_config['nb_observation'] = self.agent_helper.env.observation_space.shape[0]
        self.agent_helper.result.agent_config['nb_actions'] = nb_actions

        self.agent_helper.result.agent_config['actor'] = {}
        self.agent_helper.result.agent_config['actor']['summary'] = actor_summary

        self.agent_helper.result.agent_config['critic'] = {}
        self.agent_helper.result.agent_config['critic']['summary'] = critic_summary
        self.agent_helper.result.agent_config['metrics'] = ['mae']

        # creating the Agent
        processor = ActionScheduleProcessor(num_nodes=num_nodes, num_sfcs=num_sfcs, num_sfs=num_sfs)
        memory = SequentialMemory(limit=self.agent_helper.config['mem_limit'],
                                  window_length=self.agent_helper.config['mem_window_length'])
        random_process = GaussianWhiteNoiseProcess(sigma=self.agent_helper.config['rand_sigma'],
                                                   mu=self.agent_helper.config['rand_mu'], size=nb_actions)

        agent = DDPGAgent(nb_actions=nb_actions,
                          actor=actor,
                          critic=critic,
                          critic_action_input=action_input,
                          memory=memory,
                          nb_steps_warmup_critic=self.agent_helper.config['nb_steps_warmup_critic'],
                          nb_steps_warmup_actor=self.agent_helper.config['nb_steps_warmup_actor'],
                          random_process=random_process,
                          gamma=self.agent_helper.config['gamma'],
                          target_model_update=self.agent_helper.config['target_model_update'],
                          processor=processor,
                          batch_size=64)
        agent.compile(Adam(lr=self.agent_helper.config['learning_rate'],
                           decay=self.agent_helper.config['learning_rate_decay']), metrics=['mae'])
        self.agent = agent

    def fit(self, env, episodes, verbose, episode_steps, callbacks, log_interval, agent_id=-1):
        """Mask the agent fit function"""
        self.agent_helper.callbacks = self.create_callbacks(self.agent_helper.graph_path, self.agent_helper.config_dir)
        # create additional, custom callback to store agent's episode rewards
        EPISODE_REWARDS[agent_id] = []
        reward_dict_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: [
                EPISODE_REWARDS[agent_id].append(logs['episode_reward'])
            ]
        )
        self.agent_helper.callbacks.append(reward_dict_callback)
        steps = episodes * self.agent_helper.episode_steps
        self.agent.fit(env, steps, verbose=verbose, nb_max_episode_steps=episode_steps,
                       callbacks=self.agent_helper.callbacks, log_interval=log_interval, nb_max_start_steps=0)

    def test(self, env, episodes, verbose, episode_steps, callbacks):
        """Mask the agent fit function"""
        # Check to see if the test is called after training. Causes duplicate CSV headers
        # when agent is called only for testing.
        if self.agent_helper.train:
            # Create a fresh simulator with test argument
            self.agent_helper.env.simulator = create_simulator(self.agent_helper)
        self.agent_helper.callbacks = self.create_callbacks(self.agent_helper.graph_path, self.agent_helper.config_dir)
        self.agent.test(env, episodes, verbose=verbose,
                        nb_max_episode_steps=episode_steps,
                        callbacks=self.agent_helper.callbacks)

    def save_weights(self, file, overwrite=True):
        weights_file = f"{file}weights.h5f"
        dir_path = os.path.dirname(os.path.realpath(weights_file))
        os.makedirs(dir_path, exist_ok=True)

        # After training is done, we save the final weights in the result_base_path.
        logger.info("saving model and weights to %s", weights_file)
        self.agent.save_weights(weights_file, overwrite)

        with open(f"{file}model_critic.yaml", "w") as critic_yaml:
            critic_yaml.write(self.agent.critic.to_yaml())
        with open(f"{file}model_actor.yaml", "w") as actor_yaml:
            actor_yaml.write(self.agent.actor.to_yaml())

    def load_weights(self, weights_file):
        self.agent.load_weights(f"{weights_file}.h5f")

    def create_callbacks(self, graph_id, config_dir):
        # Now we create a tensorboard callback. This logs the episode rewards to the tensorboard.
        tensorboard_callback = TensorBoard(log_dir=graph_id, write_graph=True, write_images=True)

        # To log the observation vector we add an other callback, which simply outputs the vector.
        logger_callback = LambdaCallback()

        # write the reward to a csv.
        run_reward_csv_writer = csv.writer(open(f"{config_dir}run_reward.csv", 'a+', newline=''))
        episode_reward_csv_writer = csv.writer(open(f"{config_dir}episode_reward.csv", 'a+', newline=''))
        run_reward_csv_writer.writerow(['run', 'reward'])  # add a header
        episode_reward_csv_writer.writerow(['episode', 'reward'])  # add a header

        reward_csv_callback = LambdaCallback(
            on_batch_end=lambda step, logs: [
                run_reward_csv_writer.writerow([step, logs['reward']])
            ],
            on_epoch_end=lambda epoch, logs: [
                episode_reward_csv_writer.writerow([epoch, logs['episode_reward']])
            ]
        )

        return [tensorboard_callback, logger_callback, reward_csv_callback]
