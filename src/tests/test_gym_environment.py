# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
import gym
# noinspection PyUnresolvedReferences
import rlsp
from rlsp.agents.main import get_config
from siminterface.simulator import Simulator

AGENT_CONFIG = 'res/config/agent/sample_agent.yaml'
NETWORK = 'res/networks/sample_network.graphml'
SERVICE = 'res/service_functions/abc.yaml'
SIM_CONFIG = 'res/config/simulator/sample_config.yaml'


class TestGymEnvironment(TestCase):
    def test_gym_registration(self):
        config = get_config(AGENT_CONFIG)
        simulator = Simulator(NETWORK, SERVICE, SIM_CONFIG)
        env = gym.make('rlsp-env-v1', agent_config=config, simulator=simulator, network_file=NETWORK,
                       service_file=SERVICE)
        np.random.seed(123)
        env.seed(123)

    def test_step(self):
        config = get_config(AGENT_CONFIG)
        simulator = Simulator(NETWORK, SERVICE, SIM_CONFIG)
        env = gym.make('rlsp-env-v1', agent_config=config, simulator=simulator, network_file=NETWORK,
                       service_file=SERVICE)
        np.random.seed(123)
        env.seed(123)
        env.reset()
        action = env.action_space.sample()
        env.step(action)

    def test_min_max_delay(self):
        config = get_config(AGENT_CONFIG)
        simulator = Simulator(NETWORK, SERVICE, SIM_CONFIG)
        env = gym.make('rlsp-env-v1', agent_config=config, simulator=simulator, network_file=NETWORK,
                       service_file=SERVICE)
        self.assertEqual(env.min_delay, 15)
        self.assertEqual(env.max_delay, 45)
