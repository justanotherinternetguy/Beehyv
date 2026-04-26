"""Deliberately weak MNIST fully connected baseline for research-swarm tests."""

from __future__ import annotations

import torch
from torch import nn


MODEL_CONFIG = {
    "hidden_sizes": [12],
    "dropout": 0.90,
    "activation": "sigmoid",
}

TRAINING_CONFIG = {
    "epochs": 20,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "seed": 7,
    "limit_train": 6000,
    "limit_test": 1000,
    "label_smoothing": 0.1,
}


class BadFullyConnectedMNIST(nn.Module):
    """A CNN architecture replacing the MLP baseline to improve MNIST performance."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        # CNN structure for MNIST (28x28 input)
        
        # Block 1: Conv -> ReLU -> Pool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # Block 2: Conv -> ReLU -> Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Calculate size after pooling: 28 -> 14 -> 7
        # 64 channels * 7 * 7 spatial dimensions
        self.fc_input_size = 64 * 7 * 7
        
        # Final classification layer
        self.fc = nn.Linear(self.fc_input_size, 10)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Input: (B, 1, 28, 28)
        x = self.conv1(images)
        x = self.relu(x)
        x = self.pool(x) # (B, 32, 14, 14)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x) # (B, 64, 7, 7)
        
        # Flatten
        x = x.view(x.size(0), -1) # (B, 64*7*7)
        
        # Linear classification
        x = self.fc(x)
        return x


def build_model() -> nn.Module:
    return BadFullyConnectedMNIST()
