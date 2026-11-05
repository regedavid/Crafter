from torch import nn
from torch.nn import functional as F
import torch
import numpy as np


class DuelingHead(nn.Module):
    """Splits flattened CNN features into Value V(s) and Advantage A(s,a) streams."""
    def __init__(self, in_features, hidden_dim, num_actions):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Stream 1: Estimates State Value V(s) -> shape (Batch, 1)
        self.value_stream = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Stream 2: Estimates Action Advantage A(s, a) -> shape (Batch, num_actions)
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        x = self.flatten(x)
        value = self.value_stream(x)               # (Batch, 1)
        advantage = self.advantage_stream(x)       # (Batch, num_actions)
        
        # Combine using the mean-subtraction trick
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class ClassicCNN(nn.Module):
    def __init__(self, num_actions, in_channels=3, dueling=False):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        if dueling:
            self.fc = DuelingHead(64 * 8 * 8, 256, num_actions)
        else:
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions)
            )

    def forward(self, x):
        return self.fc(self.features(x))


class NatureCNN(nn.Module):
    def __init__(self, num_actions, in_channels=3, dueling=False):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        if dueling:
            self.fc = DuelingHead(64 * 4 * 4, 512, num_actions)
        else:
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )

    def forward(self, x):
        return self.fc(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.pool(self.conv(x))
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    def __init__(self, num_actions, in_channels=3, dueling=False):
        super().__init__()
        
        self.features = nn.Sequential(
            ImpalaBlock(in_channels, 16),
            ImpalaBlock(16, 32),
            ImpalaBlock(32, 32),
            nn.ReLU()
        )
        
        if dueling:
            self.fc = DuelingHead(32 * 8 * 8, 256, num_actions)
        else:
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions)
            )

    def forward(self, x):
        return self.fc(self.features(x))


# Convenience subclasses so you can register them directly in MODEL_REGISTRY
class DuelingClassicCNN(ClassicCNN):
    def __init__(self, num_actions, in_channels=3):
        super().__init__(num_actions, in_channels=in_channels, dueling=True)

class DuelingNatureCNN(NatureCNN):
    def __init__(self, num_actions, in_channels=3):
        super().__init__(num_actions, in_channels=in_channels, dueling=True)

class DuelingImpalaCNN(ImpalaCNN):
    def __init__(self, num_actions, in_channels=3):
        super().__init__(num_actions, in_channels=in_channels, dueling=True)