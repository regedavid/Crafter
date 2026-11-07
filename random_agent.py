import os
import argparse
import crafter
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200_000, help="Total environment steps")
    parser.add_argument('--exp_name', type=str, default='random_baseline', help="Experiment name")
    args = parser.parse_args()

    print(f"========================================")
    print(f"Generating Random Baseline: {args.steps} steps")
    print(f"========================================")

    # Run across the standard 3 seeds to match your other experiments
    for seed in [1, 2, 3]:
        print(f"Running Seed {seed}...")
        
        # Save to your main logdir so plot.py picks it up automatically
        logdir = os.path.join("logdir", args.exp_name, f"seed_{seed}")
        os.makedirs(logdir, exist_ok=True)
        
        env = crafter.Env(seed=seed)
        env = crafter.Recorder(env, logdir)
        
        obs = env.reset()
        episode_reward = 0.0
        completed_episodes = 0
        for step in tqdm(range(args.steps)):
            # Purely random action sampled directly from the environment
            action = env.action_space.sample()
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                completed_episodes += 1
                if completed_episodes % 50 == 0:
                    print(f"  Step {step}/{args.steps} | Episode {completed_episodes} | Return: {episode_reward:5.1f}")
                
                obs = env.reset()
                episode_reward = 0.0

    print(f"\nRandom baseline complete! Data saved to logdir/{args.exp_name}/")

if __name__ == "__main__":
    main()