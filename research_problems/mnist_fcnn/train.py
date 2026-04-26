"""Train and evaluate the MNIST research-swarm baseline."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import TRAINING_CONFIG, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MNIST research problem model.")
    parser.add_argument("--data-dir", default="data", help="Where MNIST is stored.")
    parser.add_argument("--dataset", choices=["mnist", "fake"], default="mnist",
                        help="Use real MNIST by default; fake is for offline smoke tests.")
    parser.add_argument("--download", action="store_true", help="Download MNIST if missing.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or other torch device.")
    parser.add_argument("--metrics-out", default="logs/latest_metrics.json")
    parser.add_argument("--log-file", default="logs/train_events.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = dict(TRAINING_CONFIG)
    seed = int(args.seed if args.seed is not None else config.get("seed", 7))
    _set_seed(seed)

    device = _resolve_device(args.device)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log:
        started = time.monotonic()
        _log(log, "run_start", {"args": vars(args), "training_config": config, "device": str(device)})

        train_loader, test_loader = _load_data(args, config, seed)
        model = build_model().to(device)
        optimizer = _make_optimizer(model, args, config)
        loss_fn = nn.CrossEntropyLoss()

        history = []
        epochs = int(args.epochs if args.epochs is not None else config.get("epochs", 1))
        for epoch in range(1, epochs + 1):
            train_metrics = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            test_metrics = _evaluate(model, test_loader, loss_fn, device)
            row = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
            history.append(row)
            _log(log, "epoch", row)
            print(
                f"epoch={epoch} train_accuracy={row['train_accuracy']:.4f} "
                f"test_accuracy={row['test_accuracy']:.4f}"
            )

        final = history[-1] if history else {}
        metrics = {
            "dataset": args.dataset,
            "seed": seed,
            "epochs": epochs,
            "train_size": len(train_loader.dataset),
            "test_size": len(test_loader.dataset),
            "model_parameters": sum(param.numel() for param in model.parameters()),
            "elapsed_s": round(time.monotonic() - started, 3),
            "train_accuracy": float(final.get("train_accuracy", 0.0)),
            "test_accuracy": float(final.get("test_accuracy", 0.0)),
            "test_loss": float(final.get("test_loss", 0.0)),
        }
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _log(log, "run_done", metrics)
        print("METRICS_JSON:" + json.dumps(metrics, sort_keys=True))

    return 0


def _load_data(args: argparse.Namespace, config: dict[str, Any], seed: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    data_dir = Path(args.data_dir)
    if args.dataset == "mnist":
        train_data = datasets.MNIST(data_dir, train=True, download=args.download, transform=transform)
        test_data = datasets.MNIST(data_dir, train=False, download=args.download, transform=transform)
    else:
        train_size = int(args.limit_train if args.limit_train is not None else config.get("limit_train", 6000))
        test_size = int(args.limit_test if args.limit_test is not None else config.get("limit_test", 1000))
        train_data = datasets.FakeData(size=train_size, image_size=(1, 28, 28), num_classes=10, transform=transform)
        test_data = datasets.FakeData(size=test_size, image_size=(1, 28, 28), num_classes=10, transform=transform)

    train_limit = int(args.limit_train if args.limit_train is not None else config.get("limit_train", 6000))
    test_limit = int(args.limit_test if args.limit_test is not None else config.get("limit_test", 1000))
    train_data = _limited_subset(train_data, train_limit, seed)
    test_data = _limited_subset(test_data, test_limit, seed + 1)

    batch_size = int(args.batch_size if args.batch_size is not None else config.get("batch_size", 128))
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(test_data, batch_size=batch_size, shuffle=False),
    )


def _limited_subset(dataset, limit: int, seed: int):
    if limit <= 0 or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def _make_optimizer(model: nn.Module, args: argparse.Namespace, config: dict[str, Any]):
    lr = float(args.learning_rate if args.learning_rate is not None else config.get("learning_rate", 1e-4))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else config.get("weight_decay", 0.0))
    optimizer_name = str(args.optimizer if args.optimizer is not None else config.get("optimizer", "sgd")).lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += float(loss.item()) * labels.numel()
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _log(log, event: str, payload: dict[str, Any]) -> None:
    log.write(json.dumps({"time": time.time(), "event": event, "payload": payload}, sort_keys=True) + "\n")
    log.flush()


if __name__ == "__main__":
    raise SystemExit(main())
