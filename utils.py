import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate numpy arrays to prevent memory fragmentation
        # Storing images as uint8 saves 4x RAM compared to float32
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # Randomly sample indices
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs]
        )
        
        
