from torch import nn
from torch.nn import functional as F
import torch
import numpy as np

class ClassicCNN(nn.Module):
    def __init__(self, num_actions, in_channels=3):
        super().__init__()
        
        # A textbook CNN: 3x3 kernels, padding to preserve edges, and Max Pooling
        self.features = nn.Sequential(
            # Input: 64x64x3
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces to 32x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces to 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # Reduces to 8x8
        )
        
        # Flattened size: 64 channels * 8 width * 8 height = 4096
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.fc(self.features(x))
    
    
class NatureCNN(nn.Module):
    def __init__(self, num_actions, in_channels=3):
        super().__init__()
        
        # Image size progression: 64x64 -> 15x15 -> 6x6 -> 4x4
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Flattened dimension: 64 channels * 4 width * 4 height = 1024
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
        # Pre-activation Residual Block
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
    def __init__(self, num_actions, in_channels=3):
        super().__init__()
        
        # Standard IMPALA channel progression: [16, 32, 32]
        # Image size progression (due to pooling): 64x64 -> 32x32 -> 16x16 -> 8x8
        self.features = nn.Sequential(
            ImpalaBlock(in_channels, 16),
            ImpalaBlock(16, 32),
            ImpalaBlock(32, 32),
            nn.ReLU()
        )
        
        # Flattened dimension: 32 channels * 8 width * 8 height = 2048
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.fc(self.features(x))