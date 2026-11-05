class BaseAgent:
    """
    Every custom algorithm you write should inherit from this class.
    This guarantees your agent will plug seamlessly into the training loop.
    """
    def __init__(self, config, action_space, obs_space):
        self.config = config
        self.action_space = action_space
        self.obs_space = obs_space

    def act(self, obs, training=True):
        """
        Takes an observation (numpy array) and returns an integer action.
        If training=True, algorithms like DQN should use epsilon-greedy exploration.
        """
        raise NotImplementedError

    def learn(self, obs, action, reward, next_obs, done):
        """
        Receives the transition from the environment.
        For DQN: Store in Replay Buffer and sample a batch to compute loss.
        """
        raise NotImplementedError