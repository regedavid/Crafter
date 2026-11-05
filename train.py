import argparse

import crafter
import tqdm
import gym
import torch
import numpy as np
import warnings
import psutil
import os


from dqn_agent import DQNAgent
from models import ClassicCNN
from config import DQNConfig
warnings.filterwarnings("ignore")

def run_training(env, agent, total_steps, outdir):
    obs = env.reset()
    done = False
    episode_reward = 0
    episodes_completed = 0

    # Using tqdm for a nice progress bar
    progress_bar = tqdm.tqdm(total=total_steps, desc="Training")

    for step in range(int(total_steps)):
        if step % 1000 == 0:
            process = psutil.Process(os.getpid())
            ram_mb = process.memory_info().rss / (1024 * 1024)
            print(f"Step {step} | Python RAM Usage: {ram_mb:.2f} MB")
        # 1. Agent picks an action
        action = agent.act(obs, training=True)

        # 2. Environment steps forward (Gym 0.25.2 API)
        next_obs, reward, done, info = env.step(action)

        # 3. Agent learns from the transition
        agent.learn(obs, action, reward, next_obs, done)

        episode_reward += reward
        obs = next_obs

        # 4. Handle episode end
        if done:
            episodes_completed += 1
            # You can log metrics to TensorBoard or Weights & Biases here
            progress_bar.set_postfix({'Ep': episodes_completed, 'Reward': f"{episode_reward:.1f}"})
            
            obs = env.reset()
            episode_reward = 0
            done = False
            
            if episodes_completed % 50 == 0:
                torch.save(agent.q_net.state_dict(), f"{outdir}/dqn_crafter_latest.pth")
        
        progress_bar.update(1)

    progress_bar.close()
    print("Training completed!")

# ==========================================
# 4. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='logdir/custom_run')
    parser.add_argument('--steps', type=float, default=1e6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize Environment
    env = gym.make('CrafterReward-v1')
    
    # Crafter Recorder for saving stats and videos
    env = crafter.Recorder(
        env, 
        args.outdir,
        save_stats=True,
        save_episode=False,
        save_video=False
    )

    # Initialize your chosen agent here
    # agent = YourCustomDQN(env.action_space)
    config = DQNConfig()
    model = ClassicCNN
    agent = DQNAgent(config, model, env.action_space, channels=3)

    print(f"Starting training for {args.steps} steps...")
    run_training(env, agent, args.steps, args.outdir)
    
    env.close()