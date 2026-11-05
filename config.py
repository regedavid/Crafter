import math
from dataclasses import dataclass, field

@dataclass
class DQNConfig:
    total_steps: int = 1_000_000 
    
    lr: float = 1e-4
    gamma: float = 0.99
    
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    
    epsilon_decay: float = field(init=False) 
    buffer_capacity: int = field(init=False)
    learning_starts: int = field(init=False)
    
    target_update_freq: int = 1000
    batch_size: int = 32
    train_freq: int = 4
    
    n_step: int = 3

    def __post_init__(self):
        # 2% of total steps (Min 1k, Max 10k)
        self.learning_starts = max(1000, min(10_000, int(self.total_steps * 0.02)))
        
        # 20% of total steps (max 100k)
        self.buffer_capacity = min(100_000, int(self.total_steps * 0.20))
        
        # 80% of full run
        exploration_end_step = int(self.total_steps * 0.8)
        updates_during_exploration = (exploration_end_step - self.learning_starts) / self.train_freq
        
        if updates_during_exploration > 0:
            # decay = (target_min / start_val) ^ (1 / number_of_updates)
            self.epsilon_decay = math.pow(
                self.epsilon_min / self.epsilon_start, 
                1.0 / updates_during_exploration
            )
        else:
            self.epsilon_decay = 0.99995