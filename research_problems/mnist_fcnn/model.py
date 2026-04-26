"""Improved MNIST classifier using a small CNN."""

from __future__ import annotations

import torch
from torch import nn


MODEL_CONFIG = {
    "conv_channels": 32,
}

TRAINING_CONFIG = {
    "epochs": 5,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "seed": 7,
    "limit_train": 6000,
    "limit_test": 1000,
    "dropout": 0.25,
}


class SimpleCNN(nn.Module):
    """A small CNN for MNIST with Conv2d→ReLU→MaxPool2d feature extractor and classifier head."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        config = dict(MODEL_CONFIG if config is None else config)
        conv_channels = int(config.get("conv_channels", 32))

        self.features = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(p=TRAINING_CONFIG["dropout"]),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_model() -> nn.Module:
    return SimpleCNN()
