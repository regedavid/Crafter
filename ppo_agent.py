import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from agent import BaseAgent


class PPOAgent(BaseAgent):
    def __init__(self, config, model_class, action_space, channels, obs_space=(64, 64, 3)):
        super().__init__(config, action_space, obs_space)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Shared CNN backbone + ActorCriticHead
        self.net = model_class(
            self.action_space.n, 
            in_channels=channels
        ).to(self.device)
        
        # Alias self.q_net so train.py checkpoint saving works without changes
        self.q_net = self.net
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr, eps=1e-5)
        
        # PPO Hyperparameters (with defaults tuned for Crafter)
        self.rollout_steps = getattr(self.config, 'ppo_rollout_steps', 1024)
        self.ppo_epochs = getattr(self.config, 'ppo_epochs', 4)
        self.minibatch_size = getattr(self.config, 'ppo_batch_size', 64)
        self.clip_eps = getattr(self.config, 'clip_eps', 0.2)
        self.gae_lambda = getattr(self.config, 'gae_lambda', 0.95)
        self.vf_coef = getattr(self.config, 'vf_coef', 0.5)
        self.ent_coef = getattr(self.config, 'ent_coef', 0.01)
        
        self.reset_rollout()
        
        # Display entropy coefficient in tqdm's Eps slot
        self.epsilon = self.ent_coef

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
                dist = Categorical(logits=logits)
                return dist.sample().item()
            else:
                return logits.argmax(dim=1).item()

    def learn(self, obs, action, reward, next_obs, done):
        # 1. Collect transition into on-policy rollout buffer
        self.obs_list.append(obs)
        self.act_list.append(action)
        self.rew_list.append(reward)
        self.done_list.append(done)

        # 2. Wait until we have collected a full horizon (e.g., 1024 steps)
        if len(self.obs_list) < self.rollout_steps:
            return

        # 3. Convert rollout observations and actions to tensors
        b_obs = np.array(self.obs_list, dtype=np.uint8)
        t_obs = torch.tensor(b_obs, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0
        t_act = torch.tensor(self.act_list, dtype=torch.int64, device=self.device)

        # 4. Compute old_log_probs, old_values, and bootstrap value under pi_theta_old
        with torch.no_grad():
            old_logits, old_values = self.net(t_obs)
            old_values = old_values.squeeze(-1)  # Shape: (rollout_steps,)
            old_dist = Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(t_act)

            # Bootstrap value of the final next_obs
            next_obs_transposed = np.transpose(next_obs, (2, 0, 1))
            t_next_obs = torch.tensor(next_obs_transposed, dtype=torch.float32, device=self.device)
            t_next_obs = t_next_obs.unsqueeze(0) / 255.0
            _, next_value = self.net(t_next_obs)
            next_value = next_value.item()

        # 5. Compute Generalized Advantage Estimation (GAE) and Returns
        values_np = old_values.cpu().numpy()
        advantages = np.zeros(self.rollout_steps, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - float(done)
                next_val = next_value
            else:
                next_non_terminal = 1.0 - float(self.done_list[t])
                next_val = values_np[t + 1]

            delta = self.rew_list[t] + self.config.gamma * next_val * next_non_terminal - values_np[t]
            last_gae = delta + self.config.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values_np

        t_advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        t_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # 6. PPO Multi-Epoch Mini-Batch Update Loop
        indices = np.arange(self.rollout_steps)
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, self.rollout_steps, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                mb_obs = t_obs[mb_idx]
                mb_act = t_act[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_adv = t_advantages[mb_idx]
                mb_ret = t_returns[mb_idx]

                # Normalize advantages per mini-batch for gradient stability
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Forward pass with current updated network parameters
                logits, values = self.net(mb_obs)
                values = values.squeeze(-1)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                # Probability ratio: r_t(theta) = pi_theta(a|s) / pi_theta_old(a|s)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Clipped Surrogate Policy Loss
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic Value Loss
                value_loss = F.smooth_l1_loss(values, mb_ret)

                # Total PPO Loss
                loss = policy_loss + (self.vf_coef * value_loss) - (self.ent_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()

        # 7. Clear the on-policy rollout buffer for the next horizon
        self.reset_rollout()