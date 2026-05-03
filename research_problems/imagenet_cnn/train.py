"""Train and evaluate the Tiny-ImageNet research-swarm baseline."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

from model import TRAINING_CONFIG, build_model

# Tiny-ImageNet statistics (computed from the training set)
_TINY_MEAN = [0.4802, 0.4481, 0.3975]
_TINY_STD  = [0.2770, 0.2691, 0.2821]

TINY_IMAGENET_DIR = "tiny-imagenet-200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Tiny-ImageNet research problem model.")
    parser.add_argument("--data-dir", default="data",
                        help="Root directory; Tiny-ImageNet lives at <data-dir>/tiny-imagenet-200/.")
    parser.add_argument(
        "--dataset",
        choices=["tinyimagenet", "fake"],
        default="tinyimagenet",
        help="tinyimagenet (default) or fake for offline smoke tests.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metrics-out", default="logs/latest_metrics.json")
    parser.add_argument("--log-file", default="logs/train_events.jsonl")
    parser.add_argument(
        "--test-split", action="store_true",
        help=(
            "Deterministic 80/10/10 train/dev/test split (seed=0) of all "
            "available data. When set, emits both dev_accuracy and "
            "test_accuracy; the latter is computed only on the held-out 10% "
            "the swarm has not trained on. Augmentation is disabled to keep "
            "the partition deterministic across splits (audit §7 row 3)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = dict(TRAINING_CONFIG)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    _set_seed(seed)

    device = _resolve_device(args.device)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log:
        started = time.monotonic()
        _log(log, "run_start", {"args": vars(args), "training_config": config, "device": str(device)})

        train_loader, dev_loader, test_loader = _load_data(args, config, seed)
        model = build_model().to(device)
        optimizer = _make_optimizer(model, args, config)
        loss_fn = nn.CrossEntropyLoss()

        history = []
        epochs = int(args.epochs if args.epochs is not None else config.get("epochs", 1))
        for epoch in range(1, epochs + 1):
            train_metrics = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            test_metrics = _score_model(model, test_loader, loss_fn, device)
            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            if dev_loader is not None:
                dev_metrics = _score_model(model, dev_loader, loss_fn, device)
                row.update({f"dev_{k}": v for k, v in dev_metrics.items()})
            history.append(row)
            _log(log, "epoch", row)
            dev_str = (
                f" dev_top1={row['dev_top1_accuracy']:.4f}" if dev_loader is not None else ""
            )
            print(
                f"epoch={epoch} train_top1={row['train_top1_accuracy']:.4f} "
                f"test_top1={row['test_top1_accuracy']:.4f} "
                f"test_top5={row['test_top5_accuracy']:.4f}{dev_str}"
            )

        final = history[-1] if history else {}
        metrics = {
            "dataset": args.dataset,
            "seed": seed,
            "epochs": epochs,
            "train_size": len(train_loader.dataset),
            "test_size": len(test_loader.dataset),
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "elapsed_s": round(time.monotonic() - started, 3),
            "train_accuracy": float(final.get("train_top1_accuracy", 0.0)),
            "test_accuracy": float(final.get("test_top1_accuracy", 0.0)),
            "test_top1_accuracy": float(final.get("test_top1_accuracy", 0.0)),
            "test_top5_accuracy": float(final.get("test_top5_accuracy", 0.0)),
            "test_loss": float(final.get("test_loss", 0.0)),
            "test_split_mode": bool(args.test_split),
        }
        if dev_loader is not None:
            metrics["dev_size"] = len(dev_loader.dataset)
            metrics["dev_accuracy"] = float(final.get("dev_top1_accuracy", 0.0))
            metrics["dev_top1_accuracy"] = float(final.get("dev_top1_accuracy", 0.0))
            metrics["dev_top5_accuracy"] = float(final.get("dev_top5_accuracy", 0.0))
            metrics["dev_loss"] = float(final.get("dev_loss", 0.0))
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _log(log, "run_done", metrics)
        print("METRICS_JSON:" + json.dumps(metrics, sort_keys=True))

    return 0


def _three_way_split(
    dataset, seed: int = 0, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> tuple[Subset, Subset, Subset]:
    """Deterministic 80/10/10 (or other) train/dev/test split of a dataset."""
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(n * fractions[0])
    n_dev = int(n * fractions[1])
    return (
        Subset(dataset, perm[:n_train]),
        Subset(dataset, perm[n_train:n_train + n_dev]),
        Subset(dataset, perm[n_train + n_dev:]),
    )


def _load_data(
    args: argparse.Namespace, config: dict[str, Any], seed: int
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    image_size = int(args.image_size if args.image_size is not None else config.get("image_size", 64))
    normalize = transforms.Normalize(mean=_TINY_MEAN, std=_TINY_STD)

    train_transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    # In --test-split mode we re-shuffle train+val together, so per-index
    # transforms must be identical across the partition. Use val_transform
    # for the whole pool; loses augmentation but preserves reproducibility.
    pool_transform = val_transform if args.test_split else train_transform

    train_limit = int(args.limit_train if args.limit_train is not None else config.get("limit_train", 5000))
    test_limit = int(args.limit_test if args.limit_test is not None else config.get("limit_test", 1000))

    if args.dataset == "tinyimagenet":
        root = Path(args.data_dir) / TINY_IMAGENET_DIR
        _ensure_val_reorganized(root)
        train_data = datasets.ImageFolder(str(root / "train"), transform=pool_transform)
        test_data = datasets.ImageFolder(str(root / "val"), transform=val_transform)
    else:
        train_data = datasets.FakeData(
            size=train_limit, image_size=(3, image_size, image_size),
            num_classes=200, transform=pool_transform,
        )
        test_data = datasets.FakeData(
            size=test_limit, image_size=(3, image_size, image_size),
            num_classes=200, transform=val_transform,
        )

    batch_size = int(args.batch_size if args.batch_size is not None else config.get("batch_size", 32))
    generator = torch.Generator().manual_seed(seed)

    if args.test_split:
        full = ConcatDataset([train_data, test_data])
        train_subset, dev_subset, test_subset = _three_way_split(full, seed=0)
        train_subset = _limited_subset(train_subset, train_limit, seed)
        dev_subset = _limited_subset(dev_subset, max(test_limit // 2 if test_limit > 0 else 0, 1), seed + 2)
        test_subset = _limited_subset(test_subset, test_limit, seed + 1)
        return (
            DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=generator,
                       num_workers=4, pin_memory=True),
            DataLoader(dev_subset, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True),
            DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True),
        )

    train_data = _limited_subset(train_data, train_limit, seed)
    test_data = _limited_subset(test_data, test_limit, seed + 1)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator,
                   num_workers=4, pin_memory=True),
        None,
        DataLoader(test_data, batch_size=batch_size, shuffle=False,
                   num_workers=4, pin_memory=True),
    )


def _ensure_val_reorganized(root: Path) -> None:
    """
    Tiny-ImageNet ships val images flat in val/images/ with a val_annotations.txt.
    Reorganize into val/{class_id}/ so ImageFolder can load it. No-op if already done.
    """
    val_dir = root / "val"
    images_dir = val_dir / "images"
    annotations = val_dir / "val_annotations.txt"

    if not images_dir.exists():
        return  # already reorganized or missing

    print("Reorganizing Tiny-ImageNet val set into ImageFolder layout (one-time)...")
    mapping: dict[str, str] = {}
    for line in annotations.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1]

    for fname, class_id in mapping.items():
        src = images_dir / fname
        if not src.exists():
            continue
        dst_dir = val_dir / class_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst_dir / fname))

    shutil.rmtree(str(images_dir))
    print("Val reorganization complete.")


def _limited_subset(dataset, limit: int, seed: int):
    if limit <= 0 or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def _make_optimizer(model: nn.Module, args: argparse.Namespace, config: dict[str, Any]):
    lr = float(args.learning_rate if args.learning_rate is not None else config.get("learning_rate", 1e-5))
    wd = float(args.weight_decay if args.weight_decay is not None else config.get("weight_decay", 0.0))
    mom = float(args.momentum if args.momentum is not None else config.get("momentum", 0.0))
    name = str(args.optimizer if args.optimizer is not None else config.get("optimizer", "sgd")).lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=mom)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        t1, t5 = _topk_correct(logits, labels, topk=(1, 5))
        top1_correct += t1
        top5_correct += t5
        total += int(labels.numel())
    n = max(total, 1)
    return {"loss": total_loss / n, "top1_accuracy": top1_correct / n, "top5_accuracy": top5_correct / n}


@torch.no_grad()
def _score_model(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> dict[str, float]:
    model.train(mode=False)
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += float(loss.item()) * labels.numel()
        t1, t5 = _topk_correct(logits, labels, topk=(1, 5))
        top1_correct += t1
        top5_correct += t5
        total += int(labels.numel())
    model.train()
    n = max(total, 1)
    return {"loss": total_loss / n, "top1_accuracy": top1_correct / n, "top5_accuracy": top5_correct / n}


def _topk_correct(logits: torch.Tensor, labels: torch.Tensor, topk: tuple[int, ...]) -> tuple[int, ...]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.unsqueeze(0).expand_as(pred))
    results = []
    for k in topk:
        results.append(int(correct[:k].reshape(-1).sum(0).item()))
    return tuple(results)


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
