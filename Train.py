"""Train and evaluate CelebA facial-attribute recognition models."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataloader import CelebADataset, DefaultTransform, TrainTransform
from Model import NUM_ATTRIBUTES, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int, int, int, int]:
    """Return counts needed for label accuracy and micro F1."""
    probs = torch.sigmoid(logits)
    preds = probs >= 0.5
    targets = labels >= 0.5

    correct = int((preds == targets).sum().item())
    total = int(targets.numel())
    tp = int((preds & targets).sum().item())
    fp = int((preds & ~targets).sum().item())
    fn = int((~preds & targets).sum().item())
    return correct, total, tp, fp, fn


def summarize_metrics(correct: int, total: int, tp: int, fp: int, fn: int) -> Dict[str, float]:
    eps = 1e-12
    accuracy = correct / max(total, 1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "label_accuracy": float(accuracy),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    sample_count = 0
    correct = total = tp = fp = fn = 0

    for inputs, labels in tqdm(dataloader, leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = inputs.size(0)
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size

        c, n, batch_tp, batch_fp, batch_fn = batch_metrics(logits.detach(), labels)
        correct += c
        total += n
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn

    metrics = summarize_metrics(correct, total, tp, fp, fn)
    metrics["loss"] = loss_sum / max(sample_count, 1)
    return metrics


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
):
    """Train with validation-based checkpoint selection."""
    start = time.time()
    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    history = {"train": [], "val": []}

    for epoch in range(num_epochs):
        train_metrics = run_epoch(
            model, dataloaders["train"], criterion, device, optimizer=optimizer
        )
        val_metrics = run_epoch(
            model, dataloaders["val"], criterion, device, optimizer=None
        )
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs:02d} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"f1={train_metrics['micro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"f1={val_metrics['micro_f1']:.4f} "
            f"acc={val_metrics['label_accuracy']:.4f}"
        )

        if val_metrics["micro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["micro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    elapsed = time.time() - start
    print(f"Training complete in {elapsed / 60:.1f} min; best val micro-F1={best_val_f1:.4f}")
    return model, history


def make_dataloaders(args) -> Dict[str, DataLoader]:
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }

    return {
        "train": DataLoader(
            CelebADataset(
                data_root=args.data_root,
                transform=TrainTransform,
                train_mode="train",
            ),
            shuffle=True,
            **loader_kwargs,
        ),
        "val": DataLoader(
            CelebADataset(
                data_root=args.data_root,
                transform=DefaultTransform,
                train_mode="val",
            ),
            shuffle=False,
            **loader_kwargs,
        ),
        "test": DataLoader(
            CelebADataset(
                data_root=args.data_root,
                transform=DefaultTransform,
                train_mode="test",
            ),
            shuffle=False,
            **loader_kwargs,
        ),
    }


def save_learning_curve(history: Dict[str, list], output_path: Path) -> None:
    train_f1 = [row["micro_f1"] for row in history["train"]]
    val_f1 = [row["micro_f1"] for row in history["val"]]

    plt.figure(figsize=(7, 4))
    plt.plot(train_f1, label="train micro-F1")
    plt.plot(val_f1, label="val micro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Micro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="CelebA facial-attribute recognition")
    parser.add_argument("--data-root", type=str, default=None, help="CelebA root; falls back to CELEBA_ROOT")
    parser.add_argument("--model", choices=["SimpleNN", "SimpleCNN", "ResNet50"], default="ResNet50")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet weights for ResNet50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    dataloaders = make_dataloaders(args)
    model = build_model(
        args.model,
        num_classes=NUM_ATTRIBUTES,
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    model, history = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        device,
        num_epochs=args.epochs,
    )

    test_metrics = run_epoch(model, dataloaders["test"], criterion, device)
    print("test:", json.dumps(test_metrics, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": args.model,
        "num_attributes": NUM_ATTRIBUTES,
        "state_dict": model.state_dict(),
        "test_metrics": test_metrics,
        "config": vars(args),
    }
    torch.save(checkpoint, output_dir / f"{args.model}_best.pt")

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    save_learning_curve(history, output_dir / "learning_curve.png")


if __name__ == "__main__":
    main()
