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
    "epochs": 1,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "optimizer": "sgd",
    "weight_decay": 0.0,
    "seed": 7,
    "limit_train": 6000,
    "limit_test": 1000,
}


class BadFullyConnectedMNIST(nn.Module):
    """A tiny, high-dropout MLP expected to perform poorly on MNIST."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        config = dict(MODEL_CONFIG if config is None else config)
        hidden_sizes = list(config.get("hidden_sizes", [12]))
        dropout = float(config.get("dropout", 0.90))
        activation_name = str(config.get("activation", "sigmoid")).lower()

        activation: nn.Module
        if activation_name == "relu":
            activation = nn.ReLU()
        elif activation_name == "gelu":
            activation = nn.GELU()
        else:
            activation = nn.Sigmoid()

        layers: list[nn.Module] = [nn.Flatten()]
        in_features = 28 * 28
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, int(hidden_size)))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            in_features = int(hidden_size)
        layers.append(nn.Linear(in_features, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


def build_model() -> nn.Module:
    return BadFullyConnectedMNIST()
