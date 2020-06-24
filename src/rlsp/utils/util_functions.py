"""
RLSP utility functions module
"""


def create_simulator(agent_helper):
    """Create a simulator object"""
    from siminterface.simulator import Simulator

    return Simulator(agent_helper.network_path, agent_helper.service_path, agent_helper.sim_config_path,
                     test_mode=agent_helper.test_mode, test_dir=agent_helper.config_dir)
