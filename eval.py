import os
import argparse
import torch
import crafter
import numpy as np
import json

# Import configurations and registries from your main script
from config import DQNConfig
from dqn_agent import DQNAgent
from buffered_dqn_agent import RBDQNAgent, RBDDQNAgent
from a2c_agent import A2CAgent
from ppo_agent import PPOAgent
from models import ActorCriticImpalaCNN, ActorCriticNatureCNN, ActorCriticClassicCNN, ClassicCNN, DuelingImpalaCNN, DuelingNatureCNN, NatureCNN, ImpalaCNN, DuelingClassicCNN


MODEL_REGISTRY = {
                'classic': ClassicCNN,
                'nature': NatureCNN,
                'impala': ImpalaCNN,
                'dueling_classic': DuelingClassicCNN,
                'dueling_nature': DuelingNatureCNN,
                'dueling_impala': DuelingImpalaCNN,
                'ac_classic': ActorCriticClassicCNN,
                'ac_nature': ActorCriticNatureCNN,
                'ac_impala': ActorCriticImpalaCNN
                
            }
AGENT_REGISTRY = {
    'dqn': DQNAgent,
    'ddqn': RBDDQNAgent,
    'rbdqn': RBDQNAgent,
    'a2c': A2CAgent,
    'ppo': PPOAgent
}
    
def parse_experiment_name(exp_name):
    """
    Dynamically extracts agent and model from the directory name.
    Sorts by length descending so 'ac_classic' matches before 'classic'.
    """
    agents = sorted(list(AGENT_REGISTRY.keys()), key=len, reverse=True)
    models = sorted(list(MODEL_REGISTRY.keys()), key=len, reverse=True)

    matched_agent = None
    for a in agents:
        if exp_name.startswith(a + "_"):
            matched_agent = a
            break
            
    if not matched_agent:
        return None, None

    remainder = exp_name[len(matched_agent) + 1:] 
    
    matched_model = None
    for m in models:
        if remainder.startswith(m + "_"):
            matched_model = m
            break
            
    return matched_agent, matched_model

def evaluate_seed(agent_name, model_name, seed, weights_path, config_path, eval_logdir, episodes):
    print(f"  -> Loading {agent_name} with {model_name}...")
    
    env = crafter.Env(seed=seed + 100)
    env = crafter.Recorder(env, eval_logdir)

    # 1. Load the exact configuration used during training
    config = DQNConfig()
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_params = json.load(f)
        
        # Safely overwrite default attributes with the saved JSON values
        for key, value in saved_params.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        print(f"     [Warning] {config_path} not found. Using default DQNConfig.")

    # 2. Initialize Agent with the loaded config
    model_class = MODEL_REGISTRY[model_name]
    agent = AGENT_REGISTRY[agent_name](config, model_class, env.action_space, channels=3)

    # 3. Load Weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.q_net.load_state_dict(torch.load(weights_path, map_location=device))
    agent.q_net.eval()

    # 4. Evaluation Loop
    completed_episodes = 0
    episode_reward = 0.0
    obs = env.reset()

    while completed_episodes < episodes:
        action = agent.act(obs, training=False)
        print(f"Action taken: {action}")
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        if done:
            completed_episodes += 1
            if completed_episodes % 10 == 0 or completed_episodes == episodes:
                print(f"     Episode {completed_episodes:02d}/{episodes} | Return: {episode_reward:5.1f}")
            episode_reward = 0.0
            
            if completed_episodes < episodes:
                obs = env.reset()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='logdir', help="Directory containing training experiments")
    parser.add_argument('--eval_dir', type=str, default='eval', help="Top-level directory to save evaluation results")
    parser.add_argument('--episodes', type=int, default=50, help="Evaluation episodes per seed")
    parser.add_argument('--weights_file', type=str, default="dqn_crafter_latest.pth", help="Name of saved weights")
    parser.add_argument('--config_file', type=str, default="config.json", help="Name of saved config file")
    parser.add_argument('--force', action='store_true', help="Overwrite existing evaluations")
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        raise FileNotFoundError(f"Training directory '{args.logdir}' not found.")

    for exp_name in sorted(os.listdir(args.logdir)):
        exp_path = os.path.join(args.logdir, exp_name)
        if not os.path.isdir(exp_path): 
            continue

        agent_name, model_name = parse_experiment_name(exp_name)
        if not agent_name or not model_name:
            print(f"[Skipping] {exp_name}: Could not match a known agent and model.")
            continue

        print(f"\n========================================")
        print(f"Evaluating Experiment: {exp_name}")
        print(f"========================================")

        for seed_folder in sorted(os.listdir(exp_path)):
            if not seed_folder.startswith("seed_"): 
                continue
                
            seed = int(seed_folder.split("_")[1])
            seed_path = os.path.join(exp_path, seed_folder)
            
            weights_path = os.path.join(seed_path, args.weights_file)
            config_path = os.path.join(seed_path, args.config_file)
            eval_logdir = os.path.join(args.eval_dir, exp_name, seed_folder)
            
            if not os.path.exists(weights_path):
                print(f"  [Skipping] Seed {seed}: No {args.weights_file} found.")
                continue

            stats_file = os.path.join(eval_logdir, "stats.jsonl")
            if os.path.exists(stats_file) and not args.force:
                print(f"  [Skipping] Seed {seed}: Already evaluated. (Use --force to redo)")
                continue
            
            print(f"Starting Seed {seed}...")
            evaluate_seed(agent_name, model_name, seed, weights_path, config_path, eval_logdir, args.episodes)

    print("\nAll evaluations complete!")
    
    budget = args.episodes * 500

if __name__ == "__main__":
    main()