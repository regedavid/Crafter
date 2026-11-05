import os
import json
from dataclasses import asdict, is_dataclass

class RunLogger:
    def __init__(self, outdir: str, config):
        """
        Manages saving static hyperparameters and tracking dynamic training variables.
        """
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        
        self.config_path = os.path.join(self.outdir, "config.json")
        self.history_path = os.path.join(self.outdir, "hyperparams_history.jsonl")
        
        # Save the static hyperparameters once at startup
        self._save_config(config)

    def _save_config(self, config):
        """Converts any dataclass or namespace configuration into a readable JSON."""
        if is_dataclass(config):
            config_dict = asdict(config)
        else:
            # Fallback for argparse Namespaces or standard dicts
            config_dict = config if isinstance(config, dict) else vars(config)
            
        with open(self.config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        print(f"[Logger] Saved hyperparameters to: {self.config_path}")

    def log_step(self, step: int, episode: int, reward: float, epsilon: float):
        """
        Appends tracking data (including your changing epsilon) to a jsonl file 
        every time an episode completes.
        """
        log_data = {
            "step": step,
            "episode": episode,
            "episode_reward": float(reward),
            "epsilon": float(epsilon)
        }
        
        # Appending ensures the data doesn't get wiped if your run gets interrupted
        with open(self.history_path, "a") as f:
            f.write(json.dumps(log_data) + "\n")