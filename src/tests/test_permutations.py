"""Test permutation and de-permutation of node order in states and actions"""

import random
import numpy as np
from unittest import TestCase
from rlsp.envs.simulator_wrapper import SimulatorWrapper
from rlsp.envs.environment_limits import EnvironmentLimits
from dummy_env import DummySimulator
from coordsim.reader.reader import read_network, get_sfc


class TestPermutations(TestCase):
    def setUp(self) -> None:
        network_file = 'res/networks/sample_network.graphml'
        service_file = 'res/service_functions/abc.yaml'
        config_file = 'res/config/simulator/sample_config.yaml'

        self.num_nodes = 2
        self.num_sfcs = 1
        self.num_sfs = 2
        simulator = DummySimulator(network_file, service_file, config_file)
        network, ing_nodes, _ = read_network(network_file)
        sfc_list = get_sfc(service_file)
        self.env_limits = EnvironmentLimits(len(network.nodes), sfc_list)
        self.wrapper = SimulatorWrapper(simulator, self.env_limits)
        # self.wrapper.init(seed=1234)

    def test_permutation_inverse(self):
        original = [random.randint(0, 10) for _ in range(10)]
        perm = np.random.permutation(10)
        rev = self.wrapper.reverse_permutation(perm)

        original_perm = [original[i] for i in perm]
        original_rev = [original_perm[i] for i in rev]
        self.assertEqual(original, original_rev)

    # def test_state_permutation(self):
    #     state = [random.random() for _ in range(self.num_nodes)]
    #     perm_state, perm = self.wrapper.permute_node_order(state)
    #     rev = self.wrapper.reverse_permutation(perm)
    #     rev_state = self.wrapper.permute_node_order(perm_state, perm=rev)
    #     self.assertEqual(state, rev_state)
