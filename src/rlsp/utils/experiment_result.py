# -*- coding: utf-8 -*-
"""
Helps to gather all related information from an agents experiment run.
"""
import time

import yaml
from yaml.representer import SafeRepresenter


class ExperimentResult:
    """
    A data class to gather experiment related information.
    """
    # id is a timestamp in iso format
    id = ...  # type: str
    env_config = {}
    agent_config = {}
    episodes = ...  # type: int
    runtime_process_start = ...  # type: float
    runtime_walltime_start = ...  # type: float
    runtime_process = ...  # type: float
    runtime_walltime = ...  # type: float
    log_file = ...  # type: str

    def __init__(self, _id):
        self.id = _id

    def write_to_file(self, filename):
        """ Writes the gathered information to a yaml file.
        """
        with open(filename, 'w') as file:
            data = {
                'id': self.id,
                'env_config': self.env_config,
                'agent_config': self.agent_config,
                'episodes': self.episodes,
                'runtime_process': self.runtime_process,
                'runtime_walltime': self.runtime_walltime,
                'log_file': self.log_file
            }
            represent_literal_str = ExperimentResult.change_style('|', SafeRepresenter.represent_str)
            yaml.add_representer(LiteralStr, represent_literal_str)
            yaml.dump(data=data, stream=file, default_flow_style=False)

    def runtime_start(self):
        """start measuring the runtime"""
        self.runtime_process_start = time.process_time()
        self.runtime_walltime_start = time.perf_counter()

    def runtime_stop(self):
        """stop measuring the runtime"""
        self.runtime_process = time.process_time() - self.runtime_process_start
        self.runtime_walltime = time.perf_counter() - self.runtime_walltime_start

    @staticmethod
    def change_style(style, representer):
        def new_representer(dumper, data):
            scalar = representer(dumper, data)
            scalar.style = style
            return scalar

        return new_representer


class LiteralStr(str):
    pass
