class RLSPAgent:
    """RLSP Agents superclass"""
    def __init__(self, agent_helper):
        """
        Initialize the RLSP Agent
        Parameters
        ----------
        agent_helper: AgentHelper: Data class containing required runtime information
        logger: Logger
        """
        pass

    def create(self):
        """
        Create the agent
        """
        pass

    def fit(self, env):
        """
        Train the agent
        Params can vary based on agent
        """
        pass

    def test(self, env):
        """
        Test the trained agent
        Params can vary based on agent
        """
        pass

    def load_weights(self, weights_file):
        """
        Load agent NN weight
        """

    def create_callbacks(self, graph_id, config_dir):
        """
        Create callbacks to store Tensorboard graphs and result files
        """
        pass
