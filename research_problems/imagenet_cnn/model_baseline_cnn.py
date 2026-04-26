from __future__ import annotations

import torch
from torch import nn


MODEL_CONFIG = {
    "channels": [8, 16],
    "dropout": 0.85,
    "activation": "sigmoid",
    "use_batchnorm": False,
    "num_classes": 200,
}

TRAINING_CONFIG = {
    "epochs": 1,
    "batch_size": 32,
    "learning_rate": 1e-5,
    "optimizer": "sgd",
    "weight_decay": 0.0,
    "momentum": 0.0,
    "seed": 42,
    "limit_train": 5000,
    "limit_test": 1000,
    "image_size": 64,
}


class BaselineCNNImageNet(nn.Module):

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        config = dict(MODEL_CONFIG if config is None else config)
        channels = list(config.get("channels", [8, 16]))
        dropout = float(config.get("dropout", 0.85))
        activation_name = str(config.get("activation", "sigmoid")).lower()
        use_batchnorm = bool(config.get("use_batchnorm", False))
        num_classes = int(config.get("num_classes", 1000))

        activation: nn.Module
        if activation_name == "relu":
            activation = nn.ReLU()
        elif activation_name == "gelu":
            activation = nn.GELU()
        else:
            activation = nn.Sigmoid()

        features: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            features.append(nn.Conv2d(in_channels, int(out_channels), kernel_size=3, stride=2, padding=1))
            if use_batchnorm:
                features.append(nn.BatchNorm2d(int(out_channels)))
            features.append(activation)
            features.append(nn.Dropout2d(dropout))
            features.append(nn.MaxPool2d(2, 2))
            in_channels = int(out_channels)

        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model() -> nn.Module:
    return BaselineCNNImageNet()
