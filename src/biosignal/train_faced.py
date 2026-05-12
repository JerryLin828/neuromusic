"""
Train a small FACED 8-class EEG emotion baseline.

This script is intentionally separate from the DREAMER production classifier.
It uses the local braindecode/FACED Zarr mirror, extracts compact log-bandpower
features, and trains a lightweight MLP with subject-disjoint splits.

Usage:
    python -m src.biosignal.train_faced --rebuild-cache
    python -m src.biosignal.train_faced --max-epochs 50
    python -m src.biosignal.train_faced --smoke-test --rebuild-cache
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.biosignal.faced import (
    FACED_LABELS,
    build_feature_cache,
    load_feature_cache,
)
from src.utils.io import add_config_arg, load_config

logger = logging.getLogger(__name__)

DEFAULT_FACED_DIR = "data/raw/faced"
DEFAULT_CACHE_PATH = "data/processed/faced/log_bandpower_features.npz"
DEFAULT_OUTPUT_DIR = "checkpoints/faced/log_bandpower_mlp"


class FacedMLP(nn.Module):
    """Small MLP for cached FACED log-bandpower features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def split_subjects(
    subject_ids: np.ndarray,
    val_subject_fraction: float = 0.1,
    test_subject_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create subject-disjoint train/val/test masks."""
    subjects = np.unique(subject_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    n_test = max(1, int(round(len(subjects) * test_subject_fraction)))
    n_val = max(1, int(round(len(subjects) * val_subject_fraction)))
    test_subjects = set(subjects[:n_test])
    val_subjects = set(subjects[n_test: n_test + n_val])
    train_subjects = set(subjects[n_test + n_val:])

    train_mask = np.asarray([s in train_subjects for s in subject_ids])
    val_mask = np.asarray([s in val_subjects for s in subject_ids])
    test_mask = np.asarray([s in test_subjects for s in subject_ids])
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(
            "Subject split produced an empty train/val/test partition. "
            "Use more subjects or smaller validation/test fractions."
        )
    return train_mask, val_mask, test_mask


def standardize_features(
    X: np.ndarray,
    train_mask: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Z-score features using train-set statistics only."""
    mean = X[train_mask].mean(axis=0, keepdims=True)
    std = X[train_mask].std(axis=0, keepdims=True)
    return ((X - mean) / (std + eps)).astype(np.float32)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    X_t = torch.from_numpy(X[mask]).float()
    y_t = torch.from_numpy(y[mask]).long()
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def class_weights(y: np.ndarray, train_mask: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights normalized around 1.0."""
    counts = np.bincount(y[train_mask], minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one train or eval epoch."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * len(y)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += len(y)

    return total_loss / max(total, 1), correct / max(total, 1)


def predict(model: nn.Module, X: np.ndarray, mask: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict labels for a masked split."""
    model.eval()
    x = torch.from_numpy(X[mask]).float().to(device)
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            logits = model(x[start: start + 4096])
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict:
    """Compute multi-class metrics for FACED emotion classification."""
    labels = list(range(len(label_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_faced(
    data_dir: Path,
    cache_path: Path,
    output_dir: Path,
    rebuild_cache: bool = False,
    max_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    val_subject_fraction: float = 0.1,
    test_subject_fraction: float = 0.2,
    early_stopping_patience: int = 10,
    seed: int = 42,
    num_workers: int = 0,
    limit_recordings: int | None = None,
) -> dict:
    """Train and evaluate the FACED log-bandpower MLP baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if rebuild_cache or not cache_path.exists():
        logger.info("Building FACED feature cache at %s", cache_path)
        build_feature_cache(
            data_dir=data_dir,
            cache_path=cache_path,
            limit_recordings=limit_recordings,
        )

    data = load_feature_cache(cache_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    subject_ids = data["subject_id"]
    label_names = [str(x) for x in data.get("label_names", np.asarray(FACED_LABELS))]

    train_mask, val_mask, test_mask = split_subjects(
        subject_ids,
        val_subject_fraction=val_subject_fraction,
        test_subject_fraction=test_subject_fraction,
        seed=seed,
    )
    X = standardize_features(X, train_mask)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FacedMLP(
        input_dim=X.shape[1],
        num_classes=len(label_names),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(y, train_mask, len(label_names)).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = make_loader(X, y, train_mask, batch_size, True, num_workers)
    val_loader = make_loader(X, y, val_mask, batch_size, False, num_workers)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_val_acc = -1.0
    best_epoch = -1
    stale_epochs = 0
    history = []

    logger.info("FACED cache: X=%s, classes=%s", X.shape, label_names)
    logger.info(
        "Subject-disjoint split: train=%d, val=%d, test=%d windows",
        int(train_mask.sum()),
        int(val_mask.sum()),
        int(test_mask.sum()),
    )
    logger.info("Training on %s", device)

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        logger.info(
            "epoch=%03d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": X.shape[1],
                    "num_classes": len(label_names),
                    "label_names": label_names,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "feature_cache": str(cache_path),
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                logger.info("Early stopping after %d stale epochs", stale_epochs)
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_pred = predict(model, X, val_mask, device)
    test_pred = predict(model, X, test_mask, device)
    val_metrics = evaluate_predictions(y[val_mask], val_pred, label_names)
    test_metrics = evaluate_predictions(y[test_mask], test_pred, label_names)

    results = {
        "dataset": "braindecode/faced",
        "task": "8-class emotion classification",
        "model": "log_bandpower_mlp",
        "feature_cache": str(cache_path),
        "label_names": label_names,
        "split": "subject-disjoint random split",
        "seed": seed,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "split_sizes": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "history": history,
        "validation": val_metrics,
        "test": test_metrics,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Best epoch: %s", best_epoch)
    logger.info("Validation accuracy: %.4f", val_metrics["accuracy"])
    logger.info("Validation macro-F1: %.4f", val_metrics["macro_f1"])
    logger.info("Test accuracy: %.4f", test_metrics["accuracy"])
    logger.info("Test macro-F1: %.4f", test_metrics["macro_f1"])
    logger.info("Saved results to %s", results_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FACED emotion baseline")
    add_config_arg(parser)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--cache-path", type=str, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-subject-fraction", type=float, default=0.1)
    parser.add_argument("--test-subject-fraction", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use two recordings and one epoch for a fast loader/training smoke test.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    training = cfg.get("training", {})
    faced_cfg = cfg.get("faced", {})

    data_dir = Path(
        args.data_dir
        or faced_cfg.get("data_dir")
        or paths.get("faced_data_dir")
        or DEFAULT_FACED_DIR
    )
    max_epochs = args.max_epochs or training.get("max_epochs", 50)
    batch_size = args.batch_size or training.get("batch_size", 256)
    patience = args.early_stopping_patience or training.get("early_stopping_patience", 10)
    seed = args.seed if args.seed is not None else training.get("seed", 42)
    num_workers = args.num_workers if args.num_workers is not None else training.get("num_workers", 0)
    limit_recordings = None

    cache_path = Path(args.cache_path)
    output_dir = Path(args.output_dir)
    if args.smoke_test:
        max_epochs = 1
        limit_recordings = 5
        args.rebuild_cache = True
        if args.cache_path == DEFAULT_CACHE_PATH:
            cache_path = Path("data/processed/faced/smoke_log_bandpower_features.npz")
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            output_dir = Path("checkpoints/faced/smoke_log_bandpower_mlp")

    train_faced(
        data_dir=data_dir,
        cache_path=cache_path,
        output_dir=output_dir,
        rebuild_cache=args.rebuild_cache,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        val_subject_fraction=args.val_subject_fraction,
        test_subject_fraction=args.test_subject_fraction,
        early_stopping_patience=patience,
        seed=seed,
        num_workers=num_workers,
        limit_recordings=limit_recordings,
    )


if __name__ == "__main__":
    main()
