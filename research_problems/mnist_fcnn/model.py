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
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "seed": 7,
    "limit_train": 6000,
    "limit_test": 1000,
    "scheduler": "ReduceLROnPlateau",
    "scheduler_patience": 3,
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

        # Layer 1: Conv -> ReLU -> Pool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) # Output: (N, 32, 14, 14)

        # Layer 2: Conv -> ReLU -> Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # Output: (N, 64, 7, 7)

        # Layer 3: Stacked Conv -> ReLU -> Pool
        self.conv3a = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # Output: (N, 64, 3, 3)

        # Fully Connected Layers
        # Input size: 64 * 3 * 3 = 576
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(images)
        x = self.relu1(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3 (Stacked Convolutions)
        x = self.conv3a(x)
        x = nn.ReLU()(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.fc1(x)
        x = nn.ReLU()(x) # Use ReLU instead of Absolute
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def build_model() -> nn.Module:
    return CNNMNIST()
