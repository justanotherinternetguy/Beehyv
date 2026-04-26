"""Weak MNIST fully connected starting baseline for research-swarm tests."""

from __future__ import annotations

import torch
from torch import nn


MODEL_CONFIG = {
    "hidden_sizes": [12],
    "dropout": 0.2,
    "activation": "sigmoid",
}

TRAINING_CONFIG = {
    "epochs": 1,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "optimizer": "sgd",
    "weight_decay": 0.0,
    "seed": 7,
    "limit_train": 6000,
    "limit_test": 1000,
}


class Absolute(nn.Module):
    """Custom activation module for Absolute value."""
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class CNNMNIST(nn.Module):
    """LeNet-5 inspired CNN architecture for MNIST."""

    def __init__(self) -> None:
        super().__init__()
        # Input: (N, 1, 28, 28)

        # Layer 1: Conv -> Abs -> Pool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.abs1 = Absolute()
        self.pool = nn.MaxPool2d(2, 2) # Output: (N, 32, 14, 14)

        # Layer 2: Conv -> Abs -> Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.abs2 = Absolute()
        self.pool2 = nn.MaxPool2d(2, 2) # Output: (N, 64, 7, 7)

        # Fully Connected Layers
        # Input size: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(images)
        x = self.abs1(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.abs2(x)
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.abs1(x) # Apply Absolute activation after linear layer
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def build_model() -> nn.Module:
    return CNNMNIST()
