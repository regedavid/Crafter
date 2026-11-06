from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from agent import BaseAgent
from utils import ReplayBuffer


class RBDQNAgent(BaseAgent):
    def __init__(self, config, model_class, action_space, channels, obs_space=(64, 64, 3)):
        super().__init__(config, action_space, obs_space)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_net = model_class(self.action_space.n, in_channels=channels).to(self.device)
        self.target_net = model_class(self.action_space.n, in_channels=channels).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.lr)
        
        # Initialize Replay Buffer (Stores raw uint8 to save RAM)
        self.buffer = ReplayBuffer(capacity=self.config.buffer_capacity, obs_space=obs_space)
        
        # N-Step Sliding Window Setup
        self.n_step = getattr(self.config, 'n_step', 1)
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.gamma_n = self.config.gamma ** self.n_step
        
        self.epsilon = self.config.epsilon_start
        self.step_count = 0
        self.env_step_count = 0
        
        # How many steps to take randomly before starting backpropagation
        self.learning_starts = self.config.learning_starts

    def act(self, obs, training=True):
        if training and random.random() < self.epsilon:
            return self.action_space.sample()

        with torch.no_grad():
            # Convert single HWC image to CHW for PyTorch
            obs_transposed = np.transpose(obs, (2, 0, 1))
            
            # Add batch dimension and scale
            obs_tensor = torch.tensor(obs_transposed, dtype=torch.float32, device=self.device)
            obs_tensor = obs_tensor.unsqueeze(0) / 255.0
            
            q_values = self.q_net(obs_tensor)
            return q_values.argmax(dim=1).item()

    def _get_n_step_info(self):
        """Computes the compressed N-step transition from the current deque."""
        obs_0, act_0 = self.n_step_buffer[0][:2]
        
        # Final state and done flag come from the latest transition in the window
        next_obs_n, done_n = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        
        # Calculate discounted sum of rewards: r_0 + gamma*r_1 + gamma^2*r_2 ...
        n_step_reward = 0.0
        for idx, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            n_step_reward += (self.config.gamma ** idx) * r
            
        return obs_0, act_0, n_step_reward, next_obs_n, done_n

    def _add_to_buffer(self, obs, action, reward, next_obs, done):
        """Manages the N-step sliding window and flushes on episode termination."""
        if self.n_step == 1:
            self.buffer.add(obs, action, reward, next_obs, done)
            return

        self.n_step_buffer.append((obs, action, reward, next_obs, done))

        # Case 1: Episode ended! Flush all remaining transitions in the window
        if done:
            while len(self.n_step_buffer) > 0:
                obs_0, act_0, n_rew, next_obs_n, done_n = self._get_n_step_info()
                self.buffer.add(obs_0, act_0, n_rew, next_obs_n, done_n)
                self.n_step_buffer.popleft()
            return

        # Case 2: Window is not yet full, wait for more steps
        if len(self.n_step_buffer) < self.n_step:
            return

        # Case 3: Window is full (len == n_step) and episode is still ongoing
        obs_0, act_0, n_rew, next_obs_n, done_n = self._get_n_step_info()
        self.buffer.add(obs_0, act_0, n_rew, next_obs_n, done_n)

    def _compute_next_q(self, t_next_obs):
        """Standard DQN: target_net both selects and evaluates."""
        return self.target_net(t_next_obs).max(1, keepdim=True)[0]

    def learn(self, obs, action, reward, next_obs, done):
        # 1. Store the transition through the N-step sliding window
        self._add_to_buffer(obs, action, reward, next_obs, done)

        # 2. Do not learn until the buffer has enough diverse data
        if self.buffer.size < self.learning_starts:
            return
        
        self.env_step_count += 1
        if self.env_step_count % self.config.train_freq != 0:
            return
        
        # 3. Decay epsilon only when learning has actually started
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

        # 4. Sample a random batch of 32 transitions
        b_obs, b_act, b_rew, b_next_obs, b_done = self.buffer.sample(self.config.batch_size)

        # 5. Convert Batch to Tensors
        t_obs = torch.tensor(b_obs, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0
        t_next_obs = torch.tensor(b_next_obs, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0
        
        t_act = torch.tensor(b_act, dtype=torch.int64, device=self.device).unsqueeze(1)
        t_rew = torch.tensor(b_rew, dtype=torch.float32, device=self.device).unsqueeze(1)
        t_done = torch.tensor(b_done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 6. Calculate Current Q
        current_q = self.q_net(t_obs).gather(1, t_act)

        # 7. Calculate Target Q (Using self.gamma_n = gamma ** n_step)
        with torch.no_grad():
            next_q = self._compute_next_q(t_next_obs)
            target_q = t_rew + self.gamma_n * next_q * (1.0 - t_done)

        # 8. Optimize
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 9. Sync Target Network
        self.step_count += 1
        if self.step_count % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


class RBDDQNAgent(RBDQNAgent):
    """Replay Buffer Double DQN Agent."""
    def _compute_next_q(self, t_next_obs):
        """Double DQN: q_net selects action, target_net evaluates it."""
        best_next_actions = self.q_net(t_next_obs).argmax(dim=1, keepdim=True)
        return self.target_net(t_next_obs).gather(1, best_next_actions)