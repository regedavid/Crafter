from dataclasses import dataclass

@dataclass
class DQNConfig:
    # Learning parameters
    lr: float = 1e-4
    gamma: float = 0.99
    
    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.99995
    
    target_update_freq: int = 1000
    
    #replay buffer
    batch_size: int = 32
    buffer_capacity: int = 100_000
    
    learning_starts: int = 10_000
    train_freq: int = 4  # <-- Add this