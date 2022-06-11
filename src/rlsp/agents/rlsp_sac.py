
from rlsp.agents.rlsp_agent import RLSPAgent
from rlsp.utils.util_functions import create_simulator
from stable_baselines.sac import SAC
from stable_baselines.sac.policies import MlpPolicy
import numpy as np
from gym.spaces import Box
import csv
import os
import logging

logger = logging.getLogger(__name__)
EPISODE_REWARDS = {}


class SAC_Agent(RLSPAgent):
    """SAC Agent for RLSP"""


    def __init__(self, agent_helper, logger):
        self.agent_helper = agent_helper

        self.callbacks_prepared = False
        shape_flattened = (np.prod(self.agent_helper.env.env_limits.scheduling_shape),)
        # self.env = create_environment(agent_helper)

        self.agent_helper.env.action_space = Box(low=-1, high=1, shape=shape_flattened)

        self.agent = SAC(MlpPolicy, agent_helper.env,
                         gamma=self.agent_helper.config['gamma'],
                         learning_rate=self.agent_helper.config['learning_rate'],
                         buffer_size=self.agent_helper.config['buffer_size'],
                         learning_starts=self.agent_helper.config['learning_starts'],
                         train_freq=self.agent_helper.config['train_freq'],
                         batch_size=self.agent_helper.config['batch_size'],
                         tau=self.agent_helper.config['tau'],
                         ent_coef=self.agent_helper.config['ent_coef'],
                         target_update_interval=self.agent_helper.config['target_update_interval'],
                         gradient_steps=self.agent_helper.config['gradient_steps'],
                         target_entropy=self.agent_helper.config['target_entropy'],
                         random_exploration=self.agent_helper.config['random_exploration'],
                         policy_kwargs={'layers': self.agent_helper.config['hidden_layers']},
                         tensorboard_log='./')

    def fit(self, env, episodes, verbose, episode_steps, callbacks, log_interval):
        """Mask the agent fit function"""
        steps = episodes * self.agent_helper.episode_steps
        self.agent.env = env    ## added command
        self.agent.learn(steps, callback=self._callbacks, tb_log_name=self.agent_helper.graph_path)
        self.close_callbacks()

    def test(self, env, episodes, verbose, episode_steps, callbacks):
        """Mask the agent fit function"""
        # Check to see if the test is called after training. Causes duplicate CSV headers
        # when agent is called only for testing.
        if self.agent_helper.train:
            # Create a fresh simulator with test argument
            self.agent_helper.env.simulator = create_simulator(self.agent_helper)

        self.callbacks_prepared = False
        obs = self.agent_helper.env.reset()
        locals_ = {}
        locals_['episode_rewards'] = [0]
        for i in range(episodes * self.agent_helper.episode_steps):
            action, _states = self.agent.predict(obs)
            obs, reward, done, info = self.agent_helper.env.step(action)
            # Rough implementation of callbacks
            locals_['step'] = i
            locals_['reward'] = reward
            locals_['episode_rewards'][-1] += reward
            self._callbacks(locals_, {})  # Call callbacks before adding a new episode reward
            if done:
                logger.info(f"Finished testing step {i + 1}. Episode reward = {locals_['episode_rewards'][-1]}")
                locals_['episode_rewards'].append(reward)

        self.close_callbacks()

    ## save and load weights as .zip files
    def save_weights(self, weights_file, overwrite=True):
        logger.info("saving model and weights to %s", weights_file)
        dir_path = os.path.dirname(os.path.realpath(weights_file))
        os.makedirs(dir_path, exist_ok=True)
        self.agent.save(f'{weights_file}weights')

    def load_weights(self, weights_file):
        self.agent = SAC.load(weights_file)

    ## save and load weights as .h5f files as DDPG
    # def save_weights(self, weights_file, overwrite=True):
    #     weights_files = f"{weights_file}weights.h5f"
    #     dir_path = os.path.dirname(os.path.realpath(weights_files))
    #     os.makedirs(dir_path, exist_ok=True)
    #
    #     # After training is done, we save the final weights in the result_base_path.
    #     logger.info("saving model and weights to %s", weights_file)
    #     self.agent.save(weights_files, overwrite)
    #
    #     # with open(f"{file}model_critic.yaml", "w") as critic_yaml:
    #     #     critic_yaml.write(self.agent.critic.to_yaml())
    #     # with open(f"{file}model_actor.yaml", "w") as actor_yaml:
    #     #     actor_yaml.write(self.agent.actor.to_yaml())
    #
    # def load_weights(self, weights_files):
    #     # self.agent.load_weights(f"{weights_files}.h5f")
    #     # self.agent = SAC.load(weights_files)
    #     self.agent = SAC.load(f"{weights_files}.h5f")

    def _callbacks(self, locals_, globals_):
        # write the reward to a csv. will also be called in training and testing  process
        if not self.callbacks_prepared:
            self.prepare_callbacks()

        self.run_reward_csv_writer.writerow([locals_['step'], locals_.get('reward')])
        if (locals_['step'] != 1) and ((locals_['step'] + 1) % self.agent_helper.episode_steps == 0):
            self.episode_reward_csv_writer.writerow([len(locals_['episode_rewards']), locals_['episode_rewards'][-1]])

            #print episode_rewards during training
            print(len(locals_['episode_rewards']),",", locals_['episode_rewards'][-1])



    def prepare_callbacks(self):
        self.run_reward_file = open(f"{self.agent_helper.config_dir}run_reward.csv", 'a+', newline='')
        self.run_reward_csv_writer = csv.writer(self.run_reward_file)
        self.episode_rewards_file = open(f"{self.agent_helper.config_dir}episode_reward.csv", 'a+', newline='')
        self.episode_reward_csv_writer = csv.writer(self.episode_rewards_file)
        self.run_reward_csv_writer.writerow(['run', 'reward'])  # add a header
        self.episode_reward_csv_writer.writerow(['episode', 'reward'])  # add a header
        self.callbacks_prepared = True

    def close_callbacks(self):
        self.episode_rewards_file.close()
        self.run_reward_file.close()
