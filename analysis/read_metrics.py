import collections
import json
import pathlib
import numpy as np
import common

def read_stats(indir, outdir, task, method, budget=int(1e6), verbose=True, allow_incomplete=True):
    indir = pathlib.Path(indir)
    outdir = pathlib.Path(outdir)
    runs = []
    print(f'Loading {indir.name}...')
    filenames = sorted(list(indir.glob('**/stats.jsonl')))
    
    for index, filename in enumerate(filenames):
        if not filename.is_file():
            continue
        rewards, lengths, achievements = load_stats(filename, budget)
        
        # Only skip incomplete runs if allow_incomplete is False
        if not allow_incomplete and sum(lengths) < budget - 1e4:
            message = f'Skipping incomplete run ({sum(lengths)} < {budget} steps): '
            message += f'{filename.relative_to(indir.parent)}'
            print(f'==> {message}')
            continue
            
        if len(lengths) == 0:
            continue
            
        runs.append(dict(
            task=task,
            method=method,
            seed=str(index),
            xs=np.cumsum(lengths).tolist(),
            reward=rewards,
            length=lengths,
            **achievements,
        ))
        
    if not runs:
        print('No runs found.\n')
        return
        
    # Dynamically adjust budget to the shortest seed if checking mid-training
    actual_budget = min(budget, min(run['xs'][-1] for run in runs))
    print(f'Evaluating across {len(runs)} seed(s) up to {actual_budget} steps:')
    
    print_summary(runs, actual_budget, verbose)
    outdir.mkdir(exist_ok=True, parents=True)
    filename = (outdir / f'{task}-{method}.json')
    filename.write_text(json.dumps(runs))
    print('Wrote', filename)
    print('')

def load_stats(filename, budget):
    steps = 0
    rewards = []
    lengths = []
    achievements = collections.defaultdict(list)
    for line in filename.read_text().split('\n'):
        if not line.strip():
            continue
        episode = json.loads(line)
        steps += episode['length']
        if steps > budget:
            break
        lengths.append(episode['length'])
        for key, value in episode.items():
            if key.startswith('achievement_'):
                achievements[key].append(value)
        unlocks = int(np.sum([(v[-1] >= 1) for v in achievements.values()]))
        health = -0.9
        rewards.append(unlocks + health)
    return rewards, lengths, achievements

def print_summary(runs, budget, verbose):
    episodes = np.array([len(x['length']) for x in runs])
    rewards = np.array([np.mean(x['reward']) for x in runs])
    lengths = np.array([np.mean(x['length']) for x in runs])
    percents, methods, seeds, tasks = common.compute_success_rates(
        runs, budget, sortby=0)
    scores = np.squeeze(common.compute_scores(percents))
    
    print(f'Score:        {np.mean(scores):10.2f} ± {np.std(scores):.2f}')
    print(f'Reward:       {np.mean(rewards):10.2f} ± {np.std(rewards):.2f}')
    print(f'Length:       {np.mean(lengths):10.2f} ± {np.std(lengths):.2f}')
    print(f'Episodes:     {np.mean(episodes):10.2f} ± {np.std(episodes):.2f}')
    
    if verbose:
        print('\n--- Achievement Unlock Rates ---')
        # Fixed bug: avg across seeds (axis 1) for method 0, then zip with tasks
        mean_percents = np.nanmean(percents[0], axis=0)
        for task, percent in zip(tasks, mean_percents):
            name = task[len('achievement_'):].replace('_', ' ').title()
            print(f'{name:<20}  {percent:6.2f}%')

if __name__ == '__main__':
    # Point this to your experiment folder in logdir/
    read_stats(
        indir='logdir/nature_cnn_baseline',
        outdir='scores',
        task='crafter_reward',
        method='nature_cnn',
        budget=int(1e6),
        verbose=True,
        allow_incomplete=True
    )