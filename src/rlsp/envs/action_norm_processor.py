from rl.processors import Processor
from common.common_functionalities import normalize_scheduling_probabilities
import numpy as np


class ActionScheduleProcessor(Processor):
    """
    Passed to the agent during training for processing action probabilities after applying noise.
    Does two things:
    1) Rounds probabilities below threshold to 0.
    2) Normalizes probabilities to sum up to 1 for each node, SFC, SF

    This was previously done in the simulator_wrapper's apply to convert the action into a schedule the simulator
    understands.
    Problem: The processed actions were not returned and put into memory for training.
    This is solved with the action processor.
    """

    def __init__(self, num_nodes, num_sfcs, num_sfs, schedule_threshold=0.1):
        self.num_nodes = num_nodes
        self.num_sfcs = num_sfcs
        self.num_sfs = num_sfs
        self.schedule_threshold = schedule_threshold

    def process_action(self, action):
        assert len(action) == self.num_nodes * self.num_sfcs * self.num_sfs * self.num_nodes

        # iterate through action array, select slice with probabilities belonging to one SF
        # processes probabilities (round low probs to 0, normalize), append and move on
        processed_action = []
        start_idx = 0
        for _ in range(self.num_nodes * self.num_sfcs * self.num_sfs):
            end_idx = start_idx + self.num_nodes
            probs = action[start_idx:end_idx]
            rounded_probs = [p if p >= self.schedule_threshold else 0 for p in probs]
            normalized_probs = normalize_scheduling_probabilities(rounded_probs)
            # check that normalized probabilities sum up to 1 (accurate to specified float accuracy)
            assert (1 - sum(normalized_probs)) < np.sqrt(np.finfo(np.float64).eps)
            processed_action.extend(normalized_probs)
            start_idx += self.num_nodes

        assert len(processed_action) == len(action)
        return np.array(processed_action)
