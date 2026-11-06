import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from agent import BaseAgent


class A2CAgent(BaseAgent):
    def __init__(self, config, model_class, action_space, channels, obs_space=(64, 64, 3)):
        super().__init__(config, action_space, obs_space)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Single online network with shared CNN backbone + ActorCriticHead
        self.net = model_class(
            self.action_space.n, 
            in_channels=channels
        ).to(self.device)
        
        # Alias self.q_net to self.net so existing checkpointing code in train.py works seamlessly
        self.q_net = self.net
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr)
        
        # A2C Hyperparameters (with safe fallbacks if not in config)
        self.rollout_steps = getattr(self.config, 'rollout_steps', 32)
        self.vf_coef = getattr(self.config, 'vf_coef', 0.5)
        self.ent_coef = getattr(self.config, 'ent_coef', 0.01)
        
        # Temporary on-policy rollout storage
        self.reset_rollout()
        
        # Dummy epsilon=0.0 so tqdm progress bar logging in train.py doesn't break
        self.epsilon = 0.0

    def reset_rollout(self):
        self.obs_list = []
        self.act_list = []
        self.rew_list = []
        self.done_list = []

    def act(self, obs, training=True):
        with torch.no_grad():
            obs_transposed = np.transpose(obs, (2, 0, 1))
            obs_tensor = torch.tensor(obs_transposed, dtype=torch.float32, device=self.device)
            obs_tensor = obs_tensor.unsqueeze(0) / 255.0
            
            logits, _ = self.net(obs_tensor)
            
            if training:
                # Sample stochastically according to action probabilities
                dist = Categorical(logits=logits)
                return dist.sample().item()
            else:
                # During evaluation, pick the highest-probability action
                return logits.argmax(dim=1).item()

    def learn(self, obs, action, reward, next_obs, done):
        # 1. Append transition to current on-policy rollout
        self.obs_list.append(obs)
        self.act_list.append(action)
        self.rew_list.append(reward)
        self.done_list.append(done)

        # 2. Wait until we have collected a full rollout of N steps
        if len(self.obs_list) < self.rollout_steps:
            return

        # 3. Bootstrap the value of the final next_obs if the episode didn't end
        with torch.no_grad():
            next_obs_transposed = np.transpose(next_obs, (2, 0, 1))
            t_next_obs = torch.tensor(next_obs_transposed, dtype=torch.float32, device=self.device)
            t_next_obs = t_next_obs.unsqueeze(0) / 255.0
            
            _, next_value = self.net(t_next_obs)
            # If the last step terminated the episode, future return is 0.0
            R = next_value.item() * (1.0 - float(done))

        # 4. Compute discounted n-step returns backward through the rollout
        returns = [0.0] * self.rollout_steps
        for i in reversed(range(self.rollout_steps)):
            # (1.0 - done) resets R to 0 whenever crossing an episode boundary
            R = self.rew_list[i] + self.config.gamma * R * (1.0 - float(self.done_list[i]))
            returns[i] = R

        # 5. Convert rollout lists to PyTorch tensors
        b_obs = np.array(self.obs_list, dtype=np.uint8)
        t_obs = torch.tensor(b_obs, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0
        t_act = torch.tensor(self.act_list, dtype=torch.int64, device=self.device)
        t_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # 6. Forward pass through both heads
        logits, values = self.net(t_obs)
        values = values.squeeze(-1)  # Shape: (rollout_steps,)

        # 7. Calculate Advantage: A(s, a) = R - V(s)
        advantages = t_returns - values

        # 8. Compute Actor (Policy), Critic (Value), and Entropy losses
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(t_act)
        entropy = dist.entropy().mean()

        # Detach advantages so policy loss doesn't backpropagate into the Critic head
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.smooth_l1_loss(values, t_returns)

        loss = policy_loss + (self.vf_coef * value_loss) - (self.ent_coef * entropy)

        # 9. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.optimizer.step()

        # 10. Discard the used on-policy rollout
        self.reset_rollout()