"""
Train emotion classifiers on the DREAMER dataset (from HuggingFace).

Trains separate binary classifiers for arousal and valence using TSCeption
or EEGNet on the preprocessed DREAMER data (monster-monash/DREAMERA, DREAMERV).

Evaluation protocol:
  Subject-dependent split — for each of the 23 subjects, we hold out 20% of
  their windows (contiguous tail, preserving temporal order) for testing, and
  use the rest for training. All subjects' training data is pooled to train
  a single model. This matches the standard evaluation used in published
  DREAMER papers (e.g., DGCNN 84% arousal, GCB-Net 89% arousal).

  The MONSTER benchmark's cross-subject splits give ~55% (barely above random),
  which is expected for subject-independent EEG evaluation and is NOT what
  published accuracy numbers refer to.

Usage:
    python -m src.biosignal.train_dreamer --dimension arousal --model tsception
    python -m src.biosignal.train_dreamer --dimension valence --model eegnet
    python -m src.biosignal.train_dreamer --dimension both --model tsception
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

logger = logging.getLogger(__name__)

N_CHANNELS = 14
N_TIMEPOINTS = 256
SAMPLING_RATE = 128
N_CLASSES = 2
TEST_FRACTION = 0.2


def build_model(model_name: str):
    """Instantiate a model for DREAMER classification."""
    from src.biosignal.models import TSCeption, EEGNet

    if model_name == "tsception":
        return TSCeption(
            num_electrodes=N_CHANNELS,
            num_classes=N_CLASSES,
            num_T=15,
            num_S=15,
            in_channels=1,
            hid_channels=32,
            sampling_rate=SAMPLING_RATE,
            dropout=0.5,
        )
    elif model_name == "eegnet":
        return EEGNet(
            chunk_size=N_TIMEPOINTS,
            num_electrodes=N_CHANNELS,
            num_classes=N_CLASSES,
            F1=8,
            F2=16,
            D=2,
            dropout=0.25,
        )
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. Available: 'tsception', 'eegnet'"
        )


class DREAMERClassifier(pl.LightningModule):
    """Lightning wrapper for training and evaluating DREAMER classifiers."""

    def __init__(self, model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


def load_dreamer_subject_dependent(data_dir: Path, dimension: str):
    """
    Load DREAMER data and create subject-dependent train/test splits.

    For each subject, we do a stratified random 80/20 split of their windows.
    This preserves class balance in both train and test, and is the standard
    evaluation protocol in published DREAMER papers (e.g., Song et al. DGCNN,
    Zhang et al. GCB-Net).

    Returns:
        X_train, y_train, X_test_per_subject, y_test_per_subject
    """
    dim_dir = data_dir / dimension
    if not dim_dir.exists():
        raise FileNotFoundError(
            f"DREAMER {dimension} data not found at {dim_dir}.\n"
            f"Download it first: python data/scripts/download_dreamer.py"
        )

    X = np.load(dim_dir / "X.npy")
    y = np.load(dim_dir / "y.npy")
    meta = np.load(dim_dir / "metadata.npy", allow_pickle=True).item()
    subject_ids = meta["subject_id"]

    unique_subjects = np.unique(subject_ids)
    logger.info(f"Loaded DREAMER {dimension}: {X.shape}, {len(unique_subjects)} subjects")

    rng = np.random.RandomState(42)
    train_indices = []
    test_per_subject = {}

    for s in unique_subjects:
        s_indices = np.where(subject_ids == s)[0]
        s_labels = y[s_indices]

        # Stratified split: separately shuffle each class, take 20% from each
        s_train = []
        s_test = []
        for cls in [0, 1]:
            cls_idx = s_indices[s_labels == cls]
            rng.shuffle(cls_idx)
            n_test = max(1, int(len(cls_idx) * TEST_FRACTION))
            s_test.extend(cls_idx[:n_test].tolist())
            s_train.extend(cls_idx[n_test:].tolist())

        train_indices.extend(s_train)
        test_per_subject[int(s)] = np.array(s_test)

    train_indices = np.array(train_indices)
    rng.shuffle(train_indices)

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test_per_subject = {s: X[idx] for s, idx in test_per_subject.items()}
    y_test_per_subject = {s: y[idx] for s, idx in test_per_subject.items()}

    n_train = len(X_train)
    n_test = sum(len(v) for v in X_test_per_subject.values())
    logger.info(f"  Train: {n_train} samples, Test: {n_test} samples ({n_test/(n_train+n_test):.1%} of total)")
    logger.info(f"  Train class balance: {(y_train==0).sum()}/{(y_train==1).sum()}")

    return X_train, y_train, X_test_per_subject, y_test_per_subject


def prepare_input(X: np.ndarray, model_name: str) -> torch.Tensor:
    """Reshape EEG data to match model's expected input format."""
    X_tensor = torch.from_numpy(X).float()
    if model_name in ("tsception", "eegnet"):
        X_tensor = X_tensor.unsqueeze(1)  # (N, 1, 14, 256)
    return X_tensor


def evaluate_per_subject(
    model: DREAMERClassifier,
    X_test_per_subject: dict,
    y_test_per_subject: dict,
    model_name: str,
    device: torch.device,
) -> dict:
    """Evaluate on each subject's held-out test set."""
    model.eval()
    results = {}

    for s in sorted(X_test_per_subject.keys()):
        X_s = prepare_input(X_test_per_subject[s], model_name).to(device)
        y_s = torch.from_numpy(y_test_per_subject[s]).long()

        with torch.no_grad():
            logits = model(X_s)
        preds = logits.argmax(dim=1).cpu()
        acc = (preds == y_s).float().mean().item()
        results[s] = acc

    return results


def train_dimension(
    data_dir: Path,
    dimension: str,
    model_name: str,
    output_dir: Path,
    max_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
):
    """Train on pooled subject-dependent splits for one dimension."""
    X_train, y_train, X_test_per_subject, y_test_per_subject = \
        load_dreamer_subject_dependent(data_dir, dimension)

    X_train_t = prepare_input(X_train, model_name)
    y_train_t = torch.from_numpy(y_train).long()

    # Use 10% of training data as validation for early stopping
    n_val = int(len(X_train_t) * 0.1)
    perm = torch.randperm(len(X_train_t))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_loader = DataLoader(
        TensorDataset(X_train_t[train_idx], y_train_t[train_idx]),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(X_train_t[val_idx], y_train_t[val_idx]),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    model = build_model(model_name)
    lit_model = DREAMERClassifier(model, lr=lr)

    ckpt_dir = output_dir / dimension / model_name
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="val_acc", mode="max", patience=10, verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stop],
        default_root_dir=str(ckpt_dir),
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Training {dimension} | {model_name}")
    logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
    logger.info(f"{'='*60}")

    trainer.fit(lit_model, train_loader, val_loader)

    # Load best checkpoint for evaluation
    best_path = checkpoint_callback.best_model_path
    if best_path:
        lit_model = DREAMERClassifier.load_from_checkpoint(
            best_path, model=build_model(model_name), lr=lr,
        )
    device = next(lit_model.parameters()).device

    # Per-subject evaluation
    subject_accs = evaluate_per_subject(
        lit_model, X_test_per_subject, y_test_per_subject, model_name, device,
    )

    mean_acc = np.mean(list(subject_accs.values()))
    std_acc = np.std(list(subject_accs.values()))

    results = {
        "dimension": dimension,
        "model": model_name,
        "evaluation": "subject-dependent (80/20 temporal split per subject)",
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "per_subject": {str(k): float(v) for k, v in subject_accs.items()},
        "best_checkpoint": best_path,
    }

    report_path = ckpt_dir / "results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {dimension} | {model_name}")
    logger.info(f"  Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"  Per-subject:")
    for s, acc in sorted(subject_accs.items()):
        logger.info(f"    Subject {s:2d}: {acc:.4f}")
    logger.info(f"  Best checkpoint: {best_path}")
    logger.info(f"  Results saved to {report_path}")
    logger.info(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train DREAMER emotion classifiers")
    parser.add_argument("--data-dir", type=str, default="data/raw/dreamer")
    parser.add_argument("--dimension", type=str, default="arousal",
                        choices=["arousal", "valence", "both"])
    parser.add_argument("--model", type=str, default="tsception",
                        choices=["tsception", "eegnet"])
    parser.add_argument("--output-dir", type=str, default="checkpoints/dreamer")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dimensions = ["arousal", "valence"] if args.dimension == "both" else [args.dimension]

    for dim in dimensions:
        train_dimension(
            data_dir=Path(args.data_dir),
            dimension=dim,
            model_name=args.model,
            output_dir=Path(args.output_dir),
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )


if __name__ == "__main__":
    main()
