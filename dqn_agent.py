from agent import BaseAgent
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import torch.optim as optim
import random

from models import ClassicCNN

class DQNAgent(BaseAgent):
    """
    A simple DQN agent.
    """
    def __init__(self, config, model:ClassicCNN, action_space, channels, obs_space=(64, 64, 3)):
        super().__init__(config, action_space, obs_space)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Online CNN Agent initialized. Using device: {self.device}")
        
        self.q_net = model(
            num_actions=action_space.n, 
            in_channels=channels
            ).to(self.device)
        self.target_net = model(
            num_actions=self.action_space.n, 
            in_channels=channels
            ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=self.config.lr)
        
        self.epsilon = self.config.epsilon_start
        self.step_count = 0

    def act(self, obs, training=True):
        if training and random.random() < self.epsilon:
            return self.action_space.sample()

        with torch.no_grad():
            # transpose, convert to tensor, add batch dimension (Batch=1), and scale 0-1
            obs_tensor = self._format(obs, dtype=torch.float32)
            
            q_values = self.q_net(obs_tensor)
            return q_values.argmax(dim=1).item()

    def learn(self, obs, action, reward, next_obs, done):
        self.epsilon = max(
            self.config.epsilon_min, 
            self.config.epsilon_start * self.config.epsilon_decay
            )
        
        t_obs = self._format(obs, dtype=torch.float32)
        t_next_obs = self._format(next_obs, dtype=torch.float32)
        
        t_action = self._tensor([[action]], dtype=torch.int64)
        t_reward = self._tensor([[reward]], dtype=torch.float32)
        t_done = self._tensor([[float(done)]], dtype=torch.float32)
        
        current_q = self.q_net(t_obs).gather(1, t_action)
        with torch.no_grad():
            max_next_q = self.target_net(t_next_obs).max(1, keepdim=True)[0]
            td_target = t_reward + self.config.gamma * max_next_q * (1.0 - t_done)
        
        loss = self._loss_function(current_q, td_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
    def _loss_function(self, x, y):
        return nn.SmoothL1Loss()(x, y)
    
    def _tensor(self, x, dtype):
        return torch.tensor(x, dtype=dtype, device=self.device)
        
    def _normalize(self, x):
        return (x / 255)
        
    def _unsqueeze(self, x):
        return x.unsqueeze(0)
    
    def _transpose(self, x):
        return np.transpose(x, (2, 0, 1))
    
    def _format(self, x, dtype):
        return self._normalize(self._unsqueeze(self._tensor(self._transpose(x), dtype)))
    
        