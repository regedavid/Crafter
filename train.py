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
from buffered_dqn_agent import RBDQNAgent, RBDDQNAgent
from models import ClassicCNN, DuelingImpalaCNN, DuelingNatureCNN, NatureCNN, ImpalaCNN, DuelingClassicCNN
from config import DQNConfig
from logger import RunLogger

warnings.filterwarnings("ignore")

def run_training(env, agent, total_steps, outdir, logger, seed):
    obs = env.reset()
    done = False
    episode_reward = 0
    episodes_completed = 0

    # Using tqdm for a nice progress bar
    progress_bar = tqdm.tqdm(
        total=total_steps, 
        desc=f"Seed {seed}"
    )
    process = psutil.Process(os.getpid())
    for step in range(int(total_steps)):

        action = agent.act(obs, training=True)

        next_obs, reward, done, info = env.step(action)

        agent.learn(obs, action, reward, next_obs, done)

        episode_reward += reward
        obs = next_obs

        if done:
            episodes_completed += 1
            logger.log_step(
                step=step, 
                episode=episodes_completed, 
                reward=episode_reward, 
                epsilon=getattr(agent, "epsilon", 1.0) 
            )
            ram_mb = process.memory_info().rss / (1024 * 1024)
            progress_bar.set_postfix({
                'Ep': episodes_completed, 
                'Rew': f"{episode_reward:.1f}", 
                'Eps': f"{getattr(agent, 'epsilon', 1.0):.3f}",
                'RAM': f"{ram_mb:.0f}MB"
            })
            
            obs = env.reset()
            episode_reward = 0
            done = False
            
            if episodes_completed % 50 == 0:
                torch.save(agent.q_net.state_dict(), f"{outdir}/dqn_crafter_latest.pth")
        
        progress_bar.update(1)

    progress_bar.close()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='default_run', help='Name of the experiment')
    parser.add_argument('--model', 
                        type=str, 
                        default='nature', 
                        choices=['classic', 'nature', 'impala', 'dueling_classic', 'dueling_nature', 'dueling_impala'])
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'rbdqn', 'ddqn'])
    parser.add_argument('--steps', type=float, default=1e5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    MODEL_REGISTRY = {
            'classic': ClassicCNN,
            'nature': NatureCNN,
            'impala': ImpalaCNN,
            'dueling_classic': DuelingClassicCNN,
            'dueling_nature': DuelingNatureCNN,
            'dueling_impala': DuelingImpalaCNN
            
        }
    AGENT_REGISTRY = {
        'dqn': DQNAgent,
        'ddqn': RBDDQNAgent,
        'rbdqn': RBDQNAgent
    }
        
    ModelClass = MODEL_REGISTRY[args.model]
    AgentClass = AGENT_REGISTRY[args.agent]
    
    outdir = f"logdir/{args.exp_name}/seed_{args.seed}"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make('CrafterReward-v1')
    
    env = crafter.Recorder(
        env, 
        outdir,
        save_stats=True,
        save_episode=False,
        save_video=False
    )

    config = DQNConfig(total_steps=args.steps)

    print(f"--- Config Initialized ---")
    print(f"Total Steps:     {config.total_steps}")
    print(f"Buffer Capacity: {config.buffer_capacity}")
    print(f"Learning Starts: {config.learning_starts}")
    print(f"Epsilon Decay:   {config.epsilon_decay:.6f}")
    print(f"--------------------------")
    
    logger = RunLogger(outdir=outdir, config=config)
    
    agent = AgentClass(config, ModelClass, env.action_space, channels=3)
    print(f"Using torch {agent.device}")

    print(f"Starting training for {args.steps} steps...")
    run_training(env, agent, args.steps, outdir, logger, args.seed)
    
    env.close()