from datetime import datetime
from logging import FileHandler, Formatter
import logging.config
import os
import os.path
from shutil import copyfile, copy
import click
import glob
import random
import yaml
from pathlib import Path
from rlsp.utils.constants import SUPPORTED_OBJECTIVES
from rlsp.utils.experiment_result import ExperimentResult, LiteralStr
from rlsp.utils.util_functions import create_simulator
from rlsp.agents.agent_helper import AgentHelper
from rlsp.agents.rlsp_ddpg import DDPG
import gym
import numpy as np
from keras import backend as K
import tensorflow as tf
import pandas as pd
from common.common_functionalities import create_input_file, num_ingress


ENV_NAME = 'rlsp-env-v1'
DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


logger = None


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('agent_config', type=click.Path(exists=True))
@click.argument('network', type=click.Path(exists=True))
@click.argument('service', type=click.Path(exists=True))
@click.argument('sim_config', type=click.Path(exists=True))
@click.argument('episodes', type=int)
@click.option('--seed', default=random.randint(1000, 9999),
              help="Specify the random seed for the environment and the learning agent.")
@click.option('-t', '--test', help="Name of the training run whose weights should be used for testing.")
@click.option('-w', '--weights', help="Continue training with the specified weights (similar to testing)")
@click.option('-a', '--append-test', is_flag=True, help="Append a test run of the previously trained agent.")
@click.option('-v', '--verbose', is_flag=True, help="Set console logger level to debug. (Default is INFO)")
@click.option('-b', '--best', is_flag=True, help="Test the best of the trained agents so far.")
@click.option('-ss', '--sim-seed', type=int, help="Set the simulator seed", default=None)
@click.option('-gs', '--gen-scenario', type=click.Path(exists=True),
              help="Diff. sim config file for additional scenario test", default=None)
def cli(agent_config, network, service, sim_config, episodes, seed, test, weights, append_test, verbose, best,
        sim_seed, gen_scenario):
    """rlsp cli for learning and testing"""
    global logger

    # Setup agent helper class
    agent_helper = setup(agent_config, network, service, sim_config, seed, episodes, weights, verbose, DATETIME, test,
                         append_test, best, sim_seed, gen_scenario)

    # Execute training or testing
    execute(agent_helper)
    # Save results
    wrap_up(agent_helper)


def setup(agent_config, network, service, sim_config, seed, episodes, weights,
          verbose, DATETIME, test, append_test, best, sim_seed, gen_scenario):
    """Overall setup for the rl variables"""
    if best:
        assert not (test or append_test or weights), "Cannot run 'best' with test, append_test, or weights"
        result_dir = f"results/{get_base_path(agent_config, network, service, sim_config)}"
        test = select_best_agent(result_dir)
    # Create the AgentHelper data class
    agent_helper = AgentHelper(agent_config, network, service, sim_config, seed, episodes, weights, verbose, DATETIME,
                               test, append_test, sim_seed, gen_scenario)

    # Setup the files and paths required for the agent
    setup_files(agent_helper, best)
    set_random_seed(seed, agent_helper)
    agent_helper.config = get_config(agent_helper.agent_config_path)
    agent_helper.episode_steps = agent_helper.config['episode_steps']
    agent_helper.result.episodes = agent_helper.episodes
    # Get number of ingress nodes in the network
    no_of_ingress = num_ingress(network)
    # Create an input file with num_of_ingress and Algorithm used as attributes in the results directory
    create_input_file(agent_helper.config_dir, no_of_ingress, agent_helper.config.get('agent_type'))
    return agent_helper


def select_best_agent(result_dir, num_agents=None):
    """Return best agent out of last num_agents trained in the given result_dir. If num_agents=None, consider all."""
    agent_dirs = os.listdir(result_dir)
    if num_agents is not None and num_agents > 0:
        agent_dirs = agent_dirs[-num_agents:]

    # compare avg episode testing reward
    best_reward = None
    best_agent_dir = None
    for ag_dir in agent_dirs:
        # get first subdir of agent dir = first test dir
        sub_dirs = next(os.walk(f"{result_dir}/{ag_dir}"))[1]
        if len(sub_dirs) == 0 or ag_dir == 'best':
            print(f"Skipping {ag_dir}, which doesn't have any test results or is the 'best' directory.")
            continue
        test_dir = sub_dirs[0]
        # compare avg testing rewards and choose best
        ep_rewards = pd.read_csv(f"{result_dir}/{ag_dir}/{test_dir}/episode_reward.csv")
        avg_reward = ep_rewards['reward'].mean()
        if best_reward is None or avg_reward > best_reward:
            best_reward = avg_reward
            best_agent_dir = ag_dir

    print(f"Best agent found: {best_agent_dir} with avg reward of {best_reward}")
    return best_agent_dir


def execute(agent_helper):
    """Execution function for testing or training"""
    agent_helper.env = create_environment(agent_helper)
    agent_helper.agent = create_agent(agent_helper)

    if agent_helper.train:
        if agent_helper.weights:
            load_weights(agent_helper.agent, f"{agent_helper.result_base_path}/{agent_helper.weights}/weights")
        train_agent(agent_helper)
        if agent_helper.test:
            # if test after training (append_test) test for 1 episodes
            agent_helper.episodes = 1
            agent_helper.result = ExperimentResult(agent_helper.experiment_id)
            agent_helper.result.episodes = agent_helper.episodes
            agent_helper.test_mode = True
            setup_files(agent_helper)
    if agent_helper.test:
        test_agent(agent_helper)
        if agent_helper.gen_scenario:
            agent_helper.gen_scenario_test = True
            setup_files(agent_helper)
            logger.info("Testing with a different sim config file")
            test_agent(agent_helper)


def train_agent(agent_helper):
    """Calling the agent's train function"""
    agent_helper.result.runtime_start()
    training(agent_helper.agent, agent_helper.env, agent_helper.callbacks, agent_helper.episodes, agent_helper.result)
    agent_helper.result.runtime_stop()
    agent_helper.agent.save_weights(agent_helper.config_dir, overwrite=True)


def test_agent(agent_helper):
    """Calling the agent's test function"""
    logger.info("Switching to testing mode")
    load_weights(agent_helper.agent, agent_helper.weights_path)
    agent_helper.result.runtime_start()
    testing(agent_helper.agent, agent_helper.env, agent_helper.callbacks, agent_helper.episode_steps,
            agent_helper.episodes, agent_helper.result)
    agent_helper.result.runtime_stop()


def wrap_up(agent_helper):
    """Wrap up execution and write results"""
    logger.info(f"writing result to file: {agent_helper.result_file}")
    agent_helper.result.write_to_file(agent_helper.result_file)
    logger.info(f"See {agent_helper.logfile} for {'full (DEBUG)' if agent_helper.verbose else 'INFO'} log output.")


def get_base_path(agent_config_path, network_path, service_path, sim_config_path):
    """Return base path based on specified input paths."""
    agent_config_stem = os.path.splitext(os.path.basename(agent_config_path))[0]
    network_stem = os.path.splitext(os.path.basename(network_path))[0]
    service_stem = os.path.splitext(os.path.basename(service_path))[0]
    config_stem = os.path.splitext(os.path.basename(sim_config_path))[0]
    return f"{agent_config_stem}/{network_stem}/{service_stem}/{config_stem}"


def setup_files(agent_helper, best=False):
    """Setup result files and paths"""
    if agent_helper.gen_scenario_test:
        agent_helper.gen_scenario_result_base_path = agent_helper.result_base_path
        agent_helper.sim_config_path = click.format_filename(agent_helper.gen_scenario)
    else:
        agent_helper.sim_config_path = click.format_filename(agent_helper.sim_config_path)

    agent_helper.agent_config_path = click.format_filename(agent_helper.agent_config_path)
    agent_helper.network_path = click.format_filename(agent_helper.network_path)
    agent_helper.service_path = click.format_filename(agent_helper.service_path)

    # set result and graph base path based on network, service, config name
    base_path = get_base_path(agent_helper.agent_config_path, agent_helper.network_path,
                              agent_helper.service_path, agent_helper.sim_config_path)
    agent_helper.result_base_path = f"./results/{base_path}"
    agent_helper.graph_base_path = f"./graph/{base_path}"

    # Set config and log file paths
    agent_helper.config_dir = f"{agent_helper.result_base_path}/{agent_helper.experiment_id}/"
    agent_helper.logfile = f"{agent_helper.config_dir}training.log"
    agent_helper.graph_path = f"{agent_helper.graph_base_path}/{agent_helper.experiment_id}"
    if agent_helper.test and not agent_helper.append_test:
        # Set paths used for test
        if best:
            agent_helper.config_dir = f"{agent_helper.result_base_path}/best/{agent_helper.test}" \
                                      f"/test-{DATETIME}_seed{agent_helper.seed}/"
            agent_helper.graph_path = f"{agent_helper.graph_base_path}/{agent_helper.test}" \
                                      f"/test-{DATETIME}_seed{agent_helper.seed}_best/"
        else:
            agent_helper.config_dir = f"{agent_helper.result_base_path}/{agent_helper.test}" \
                                      f"/test-{DATETIME}_seed{agent_helper.seed}/"
            agent_helper.graph_path = f"{agent_helper.graph_base_path}/{agent_helper.test}" \
                                      f"/test-{DATETIME}_seed{agent_helper.seed}/"
        agent_helper.logfile = f"{agent_helper.config_dir}test.log"
        agent_helper.weights_path = f"{agent_helper.result_base_path}/{agent_helper.test}/weights"
    if agent_helper.append_test:
        # reset append test flag so that next time setup_files is called result files are set properly for tests
        agent_helper.append_test = False
    agent_helper.result.log_file = agent_helper.logfile
    agent_helper.result_file = f"{agent_helper.config_dir}result.yaml"
    # FIXME: Logging setup has to be done here for now. Move to a proper location
    global logger
    logger = setup_logging(agent_helper.verbose, agent_helper.logfile)

    # Copy files to result dir
    agent_helper.agent_config_path, agent_helper.network_path, agent_helper.service_path, \
        agent_helper.sim_config_path = copy_input_files(
            agent_helper.config_dir,
            agent_helper.agent_config_path,
            agent_helper.network_path,
            agent_helper.service_path,
            agent_helper.sim_config_path)

    if agent_helper.gen_scenario_test:
        weights = f"{agent_helper.gen_scenario_result_base_path}/{agent_helper.test}/weights*"
        for file in glob.glob(r'{}'.format(weights)):
            copy(file, f"{agent_helper.result_base_path}/{agent_helper.test}/")


def set_random_seed(seed, agent_helper):
    """Set random seed for all involved libraries (random, numpy, tensorflow)"""
    logger.info(f"Using Seed: {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.compat.v1.set_random_seed(seed)
    agent_helper.sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    K.set_session(agent_helper.sess)
    random.seed(seed)
    np.random.seed(seed)


def get_config(config_file):
    """Parse agent config params in specified yaml file and return as Python dict"""
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # if missing in the config, use following defaults
        config.setdefault('shuffle_nodes', False)
        config.setdefault('observation_space', ['ingress_traffic'])
        # safety checks
        assert 'objective' in config and config['objective'] in SUPPORTED_OBJECTIVES, \
            f"Objective {config['objective']} not recognized. Must be one of {SUPPORTED_OBJECTIVES}, " \
            f"recommended default: 'prio-flow'."
        if config['objective'] == 'prio-flow':
            assert 'target_success' in config and \
                   (config['target_success'] == 'auto' or 0 <= config['target_success'] <= 1)
        if config['objective'] in {'soft-deadline', 'soft-deadline-exp'}:
            assert 'soft_deadline' in config
            if config['objective'] == 'soft-deadline-exp':
                assert 'dropoff' in config and config['dropoff'] > 0, "Use 'soft-deadline' objective for 0 dropoff."
        if config['objective'] == 'weighted':
            for weight in ['flow_weight', 'delay_weight', 'node_weight', 'instance_weight']:
                if weight not in config:
                    logger.warning(f"Using weighted objective, but {weight} not configured. Defaulting to {weight}=0.")
        config.setdefault('target_success', None)
        config.setdefault('soft_deadline', None)
        config.setdefault('dropoff', None)
        for weight in ['flow_weight', 'delay_weight', 'node_weight', 'instance_weight']:
            config.setdefault(weight, 0)
    return config


def copy_input_files(target_dir, agent_config_path, network_path, service_path, sim_config_path):
    """Create the results directory and copy input files"""
    new_agent_config_path = target_dir + os.path.basename(agent_config_path)
    new_network_path = target_dir + os.path.basename(network_path)
    new_service_path = target_dir + os.path.basename(service_path)
    new_sim_config_path = target_dir + os.path.basename(sim_config_path)

    os.makedirs(target_dir, exist_ok=True)
    copyfile(agent_config_path, new_agent_config_path)
    copyfile(network_path, new_network_path)
    copyfile(service_path, new_service_path)
    copyfile(sim_config_path, new_sim_config_path)

    return new_agent_config_path, new_network_path, new_service_path, new_sim_config_path


def setup_logging(verbose, logfile):
    # main.py --> agents --> rlsp --> src --> project root
    project_root = Path(os.path.abspath(__file__)).parent.parent.parent.parent.absolute()
    logging_config_path = os.path.join(project_root, "logging.conf")
    logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
    logger = logging.getLogger()

    # disable tensorflow warnings
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    # we assume here, that there is at least one handler set in basic config and it is the console logger
    if verbose:
        logger.handlers[0].setLevel(logging.DEBUG)

    if logfile:
        dir_path = os.path.dirname(os.path.realpath(logfile))
        os.makedirs(dir_path, exist_ok=True)
        formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler = FileHandler(logfile, mode='a', encoding=None, delay=False)
        file_handler.setFormatter(formatter)
        if verbose:
            file_handler.level = logging.DEBUG
        else:
            file_handler.level = logging.INFO
        logger.addHandler(file_handler)

    return logger


def load_weights(agent, file):
    # Here we load the trained model from EXPERIMENTS_BASE_PATH
    logger.info(f"load weights:{file}")
    agent.load_weights(file)


def create_environment(agent_helper):
    # not sure why, but simulator has to be loaded here (not at top) for proper logging

    agent_helper.result.env_config['seed'] = agent_helper.seed
    agent_helper.result.env_config['sim-seed'] = agent_helper.sim_seed
    agent_helper.result.env_config['network_file'] = agent_helper.network_path
    agent_helper.result.env_config['service_file'] = agent_helper.service_path
    agent_helper.result.env_config['sim_config_file'] = agent_helper.sim_config_path
    agent_helper.result.env_config['simulator_cls'] = "siminterface.Simulator"

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME,
                   agent_config=agent_helper.config,
                   simulator=create_simulator(agent_helper),
                   network_file=agent_helper.network_path,
                   service_file=agent_helper.service_path,
                   seed=agent_helper.seed,
                   sim_seed=agent_helper.sim_seed)

    agent_helper.result.env_config['reward_fnc'] = LiteralStr(env.reward_func_repr())
    return env


def create_agent(agent_helper):
    """ Create the RL Agent"""
    agent_type = agent_helper.config.get('agent_type')
    agent = DDPG(agent_helper)
    return agent


def testing(agent, env, callbacks, episode_steps, episodes, result):
    result.agent_config['episode_steps'] = episode_steps
    # run single episode with specified number of steps
    agent.test(env, episodes=episodes, verbose=1, episode_steps=episode_steps, callbacks=callbacks)
    logger.info("FINISHED TEST")


def training(agent, env, callbacks, episodes, result):
    episode_steps = agent.agent_helper.episode_steps
    result.agent_config['episodes'] = episodes
    result.agent_config['episode_steps'] = episode_steps
    agent.fit(env, episodes=episodes, verbose=1, episode_steps=episode_steps, callbacks=callbacks,
              log_interval=episodes * episode_steps)
    logger.info("FINISHED TRAINING")


if __name__ == '__main__':
    agent_config = 'res/config/agent/ddpg/agent_obs1_weighted-f0d0n1_64a_64c_099gam_00001tau_001alp_0001dec.yaml'
    network = 'res/networks/abilene/abilene-in4-rand-cap0-2.graphml'
    service = 'res/service_functions/abc.yaml'
    sim_config = 'res/config/simulator/rand-mmp-arrival12-8_det-size001_dur100.yaml'
    # sim_config = 'res/config/simulator/det-mmp-arrival7-3_det-size0_dur100_no_traffic_prediction.yaml'

    # training for 1 episode
    # cli([agent_config, network, service, sim_config, '1', '-v'])

    # testing for 4 episode
    # cli([agent_config, network, service, sim_config, '1', '-t', '2021-01-07_13-00-43_seed1234'])

    # training & testing for 1 episodes
    cli([agent_config, network, service, sim_config, '70', '--append-test'])

    # training & testing for 4 episodes, with fixed simulator seed.
    # cli([agent_config, network, service, sim_config, '4', '--append-test', '-ss', '5555'])

    # continue training for 5 episodes
    # cli([agent_config, network, service, sim_config, '20', '--append-test', '--weights',
    #      '2020-10-13_07-47-53_seed9764'])

    # test select_best
    # cli([agent_config, network, service, sim_config, '1', '--best', '--sim-seed', '1234'])

    # Generalization: Test on two scenarios with the same trained weights
    # cli([agent_config, network, service, sim_config, '1', '--append-test', '-ss', '5555', '-gs', gen_sim_config])
