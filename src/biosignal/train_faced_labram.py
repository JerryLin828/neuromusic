"""
Fine-tune LaBraM on the local FACED 8-class emotion dataset.

The local braindecode/FACED mirror stores raw EEG windows in Zarr chunks. This
script converts those windows into a float32 memmap cache (with optional
paper-style filtering), then fine-tunes a pretrained LaBraM classifier using
subject-disjoint splits and paper-faithful optimization defaults.

Usage:
    python -m src.biosignal.train_faced_labram --build-cache
    python -m src.biosignal.train_faced_labram --max-epochs 10
    python -m src.biosignal.train_faced_labram --smoke-test --build-cache
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from src.biosignal.faced import (
    FACED_LABELS,
    find_faced_zarr_root,
    iter_recording_chunks,
    list_recordings,
    read_metadata,
)
from src.biosignal.train_faced import (
    class_weights,
    evaluate_predictions,
    split_subjects,
)
from src.utils.io import add_config_arg, load_config

logger = logging.getLogger(__name__)

DEFAULT_FACED_DIR = "data/raw/faced"
DEFAULT_CACHE_DIR = "data/processed/faced/labram_raw_float32"
DEFAULT_OUTPUT_DIR = "checkpoints/faced/labram"
DEFAULT_HF_CACHE_DIR = "data/processed/hf_cache"
DEFAULT_PRETRAINED_ID = "braindecode/labram-pretrained"


@dataclass(frozen=True)
class RawFacedCache:
    """Memory-mapped FACED windows plus aligned labels and metadata."""

    windows: np.memmap
    labels: np.ndarray
    subjects: np.ndarray
    videos: np.ndarray
    label_names: list[str]
    channel_names: list[str]
    sampling_rate: float
    scale: float


class FacedRawDataset(Dataset):
    """Dataset backed by a FACED raw-window memmap."""

    def __init__(
        self,
        windows: np.memmap,
        labels: np.ndarray,
        indices: np.ndarray,
        input_divisor: float = 100.0,
    ):
        self.windows = windows
        self.labels = labels
        self.indices = indices.astype(np.int64, copy=False)
        self.input_divisor = float(input_divisor)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        x_np = np.array(self.windows[idx], dtype=np.float32, copy=True)
        if self.input_divisor != 0:
            x_np = x_np / self.input_divisor
        x = torch.from_numpy(x_np)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def read_channel_names(data_dir: Path, subject: str | None = None) -> list[str]:
    """Read channel names from a BIDS channels.tsv file."""
    subject_glob = f"sub-{subject}" if subject is not None else "sub-*"
    matches = sorted(
        (data_dir / "sourcedata" / "braindecode").glob(
            f"{subject_glob}/eeg/*_channels.tsv"
        )
    )
    if not matches:
        raise FileNotFoundError(f"No FACED channels.tsv found under {data_dir}")

    with open(matches[0], newline="") as f:
        return [row["name"] for row in csv.DictReader(f, delimiter="\t")]


def build_raw_cache(
    data_dir: Path,
    cache_dir: Path,
    limit_recordings: int | None = None,
    scale: float = 1e6,
    apply_preprocessing: bool = True,
) -> RawFacedCache:
    """Build a raw memmap cache of FACED windows."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    recordings = list_recordings(data_dir)
    if limit_recordings is not None:
        recordings = recordings[:limit_recordings]
    if not recordings:
        raise ValueError("No FACED recordings found")

    n_windows = sum(recording.shape[0] for recording in recordings)
    n_channels = recordings[0].shape[1]
    n_times = recordings[0].shape[2]
    sampling_rate = recordings[0].sampling_rate
    for recording in recordings:
        if recording.shape[1:] != (n_channels, n_times):
            raise ValueError(f"Inconsistent FACED shape for {recording.path}")
        if recording.sampling_rate != sampling_rate:
            raise ValueError(f"Inconsistent sampling rate for {recording.path}")

    windows_path = cache_dir / "windows_float32.dat"
    metadata_path = cache_dir / "metadata.npz"
    label_to_id = {label: i for i, label in enumerate(FACED_LABELS)}
    channel_names = read_channel_names(data_dir, recordings[0].subject)

    windows = np.memmap(
        windows_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_windows, n_channels, n_times),
    )
    labels: list[int] = []
    subjects: list[str] = []
    videos: list[int] = []
    offset = 0

    logger.info(
        "Building LaBraM raw cache: %d windows, %d channels, %d samples",
        n_windows,
        n_channels,
        n_times,
    )
    for recording in recordings:
        rows = read_metadata(recording)
        row_offset = 0
        for chunk in iter_recording_chunks(recording):
            if apply_preprocessing:
                chunk = preprocess_chunk(
                    chunk,
                    sampling_rate=sampling_rate,
                )
            chunk_rows = rows[row_offset: row_offset + len(chunk)]
            row_offset += len(chunk)
            stop = offset + len(chunk)
            windows[offset:stop] = np.nan_to_num(
                chunk * scale,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
            labels.extend(label_to_id[row["emotion_label"]] for row in chunk_rows)
            subjects.extend([recording.subject] * len(chunk_rows))
            videos.extend(int(row["video_index"]) for row in chunk_rows)
            offset = stop

        if row_offset != len(rows):
            raise ValueError(
                f"Metadata/data mismatch for {recording.path.name}: "
                f"used {row_offset} rows but found {len(rows)}"
            )

    windows.flush()
    labels_array = np.asarray(labels, dtype=np.int64)
    subjects_array = np.asarray(subjects)
    videos_array = np.asarray(videos, dtype=np.int64)
    np.savez_compressed(
        metadata_path,
        shape=np.asarray(windows.shape, dtype=np.int64),
        labels=labels_array,
        subjects=subjects_array,
        videos=videos_array,
        label_names=np.asarray(FACED_LABELS),
        channel_names=np.asarray(channel_names),
        sampling_rate=np.asarray(sampling_rate, dtype=np.float32),
        scale=np.asarray(scale, dtype=np.float32),
        apply_preprocessing=np.asarray(int(apply_preprocessing), dtype=np.int64),
        dtype=np.asarray("float32"),
        zarr_root=str(find_faced_zarr_root(data_dir)),
    )
    logger.info("Saved LaBraM raw cache to %s", cache_dir)
    return load_raw_cache(cache_dir)


def load_raw_cache(cache_dir: Path) -> RawFacedCache:
    """Load a previously built raw FACED memmap cache."""
    metadata_path = cache_dir / "metadata.npz"
    windows_path_f32 = cache_dir / "windows_float32.dat"
    windows_path_f16 = cache_dir / "windows_float16.dat"
    if not metadata_path.exists() or (not windows_path_f32.exists() and not windows_path_f16.exists()):
        raise FileNotFoundError(
            f"Missing LaBraM FACED cache under {cache_dir}. "
            "Run with --build-cache first."
        )

    with np.load(metadata_path, allow_pickle=False) as data:
        shape = tuple(int(v) for v in data["shape"])
        labels = data["labels"].astype(np.int64)
        subjects = data["subjects"]
        videos = data["videos"].astype(np.int64)
        label_names = [str(x) for x in data["label_names"]]
        channel_names = [str(x) for x in data["channel_names"]]
        sampling_rate = float(data["sampling_rate"])
        scale = float(data["scale"])
        dtype = str(data["dtype"]) if "dtype" in data.files else "float16"

    if windows_path_f32.exists():
        windows_path = windows_path_f32
    elif windows_path_f16.exists():
        windows_path = windows_path_f16
    else:
        raise FileNotFoundError(
            f"Missing window data file under {cache_dir} "
            f"(expected {windows_path_f32.name} or {windows_path_f16.name})."
        )
    np_dtype = np.float32 if dtype == "float32" or windows_path == windows_path_f32 else np.float16
    windows = np.memmap(windows_path, dtype=np_dtype, mode="r", shape=shape)
    return RawFacedCache(
        windows=windows,
        labels=labels,
        subjects=subjects,
        videos=videos,
        label_names=label_names,
        channel_names=channel_names,
        sampling_rate=sampling_rate,
        scale=scale,
    )


def preprocess_chunk(
    chunk: np.ndarray,
    sampling_rate: float,
    l_freq: float = 0.1,
    h_freq: float = 75.0,
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
) -> np.ndarray:
    """Apply paper-faithful filtering to a FACED chunk."""
    data = np.asarray(chunk, dtype=np.float32)
    nyquist = sampling_rate / 2.0
    high = min(h_freq, nyquist - 1e-3)
    low = max(l_freq, 1e-4)
    if not (0 < low < high < nyquist):
        raise ValueError(
            f"Invalid bandpass for fs={sampling_rate}: low={low}, high={high}, nyquist={nyquist}"
        )
    sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    data = sosfiltfilt(sos, data, axis=-1)

    if 0 < notch_freq < nyquist:
        b_notch, a_notch = iirnotch(notch_freq / nyquist, notch_q)
        data = filtfilt(b_notch, a_notch, data, axis=-1)
    return data.astype(np.float32, copy=False)


def set_runtime_cache_dirs(hf_cache_dir: Path, output_dir: Path) -> None:
    """Keep third-party caches out of unwritable home directories."""
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache_dir = output_dir / ".matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))


def make_loader(
    cache: RawFacedCache,
    mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    input_divisor: float,
) -> DataLoader:
    indices = np.flatnonzero(mask)
    dataset = FacedRawDataset(
        cache.windows,
        cache.labels,
        indices,
        input_divisor=input_divisor,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_chs_info(channel_names: list[str], sampling_rate: float) -> list[dict] | None:
    """Create MNE channel info for LaBraM channel embeddings if MNE is present."""
    try:
        import mne
    except ImportError:
        logger.warning("mne is not installed; loading LaBraM without chs_info")
        return None

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sampling_rate,
        ch_types=["eeg"] * len(channel_names),
    )
    return info["chs"]


def load_labram(
    cache: RawFacedCache,
    pretrained_id: str,
    num_classes: int,
    freeze_encoder: bool,
) -> nn.Module:
    """Load pretrained LaBraM and adapt the classifier head."""
    try:
        from braindecode.models import Labram
    except ImportError as exc:
        raise ImportError(
            "Install Braindecode with Hugging Face support first: "
            "pip install 'braindecode[hug]'"
        ) from exc

    n_chans = cache.windows.shape[1]
    n_times = cache.windows.shape[2]
    chs_info = make_chs_info(cache.channel_names, cache.sampling_rate)
    if hasattr(Labram, "from_pretrained"):
        model = Labram.from_pretrained(
            pretrained_id,
            n_outputs=num_classes,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=cache.sampling_rate,
            input_window_seconds=n_times / cache.sampling_rate,
            chs_info=chs_info,
        )
    else:
        model = Labram(
            n_outputs=num_classes,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=cache.sampling_rate,
            input_window_seconds=n_times / cache.sampling_rate,
            chs_info=chs_info,
            qk_norm=nn.LayerNorm,
            init_values=0.1,
        )
        load_compatible_hf_state(model, pretrained_id)

    if freeze_encoder:
        for param in model.parameters():
            param.requires_grad = False
        trainable = 0
        trainable_names = []
        for name, param in model.named_parameters():
            if (
                name.startswith("head.")
                or name.startswith("final_layer.")
                or name.startswith("position_embedding")
                or name.startswith("temporal_embedding")
                or name.startswith("norm.")
            ):
                param.requires_grad = True
                trainable += param.numel()
                trainable_names.append(name)
        if trainable == 0:
            raise RuntimeError("Could not find LaBraM classification head to fine-tune")
        logger.info(
            "Frozen LaBraM encoder; trainable adaptation parameters=%d (%s)",
            trainable,
            trainable_names,
        )
    else:
        logger.info("Fine-tuning all LaBraM parameters")

    return model


def load_compatible_hf_state(model: nn.Module, repo_id: str) -> None:
    """Load Hugging Face weights into older Braindecode models when possible."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub to download LaBraM pretrained weights."
        ) from exc

    if Path(repo_id).exists():
        weights_path = Path(repo_id)
    else:
        weights_path = Path(hf_hub_download(repo_id, filename="pytorch_model.bin"))

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = adapt_pretrained_state_for_target(model, state)
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    skipped = sorted(set(state) - set(compatible))
    model.load_state_dict(compatible, strict=False)
    logger.info(
        "Loaded %d compatible LaBraM tensors from %s; skipped %d mismatched tensors",
        len(compatible),
        weights_path,
        len(skipped),
    )
    if skipped:
        logger.info("First skipped LaBraM tensors: %s", skipped[:8])


def interpolate_token_tensor(value: torch.Tensor, target_len: int) -> torch.Tensor:
    """Interpolate token sequence tensor (1, L, D) to target length."""
    if value.shape[1] == target_len:
        return value
    source = value.transpose(1, 2)  # (1, D, L)
    resized = torch.nn.functional.interpolate(
        source,
        size=target_len,
        mode="linear",
        align_corners=False,
    )
    return resized.transpose(1, 2).contiguous()


def adapt_pretrained_state_for_target(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Adapt pretrained tensors to target LaBraM shape conventions."""
    model_state = model.state_dict()
    adapted = dict(state)

    # Map final normalization if naming differs between implementations.
    if "norm.weight" in adapted and "fc_norm.weight" in model_state and "fc_norm.weight" not in adapted:
        adapted["fc_norm.weight"] = adapted["norm.weight"]
    if "norm.bias" in adapted and "fc_norm.bias" in model_state and "fc_norm.bias" not in adapted:
        adapted["fc_norm.bias"] = adapted["norm.bias"]

    if "position_embedding" in adapted and "position_embedding" in model_state:
        source = adapted["position_embedding"]
        target = model_state["position_embedding"]
        if source.ndim == 3 and target.ndim == 3 and source.shape[-1] == target.shape[-1]:
            cls_src = source[:, :1, :]
            patch_src = source[:, 1:, :]
            patch_tgt_len = target.shape[1] - 1
            patch_interp = interpolate_token_tensor(patch_src, patch_tgt_len)
            adapted["position_embedding"] = torch.cat([cls_src, patch_interp], dim=1)

    if "temporal_embedding" in adapted and "temporal_embedding" in model_state:
        source = adapted["temporal_embedding"]
        target = model_state["temporal_embedding"]
        if source.ndim == 3 and target.ndim == 3 and source.shape[-1] == target.shape[-1]:
            adapted["temporal_embedding"] = interpolate_token_tensor(source, target.shape[1])

    return adapted


def get_layer_id_for_vit(name: str, max_layer_id: int) -> int:
    """Map parameter names to layer IDs for layer-wise LR decay."""
    if name in {"cls_token", "position_embedding", "temporal_embedding"}:
        return 0
    if name.startswith("patch_embed"):
        return 0
    if name.startswith("blocks."):
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1]) + 1
    return max_layer_id


def layer_decay_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
) -> list[dict]:
    """Build AdamW parameter groups using paper-style layer-wise LR decay."""
    num_layers = (len(model.blocks) + 2) if hasattr(model, "blocks") else 14
    max_layer_id = num_layers - 1
    layer_scales = [layer_decay ** (max_layer_id - i) for i in range(num_layers)]
    groups: dict[tuple[int, bool], dict] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = get_layer_id_for_vit(name, max_layer_id=max_layer_id)
        no_decay = param.ndim == 1 or name.endswith(".bias")
        key = (layer_id, no_decay)
        if key not in groups:
            lr_scale = layer_scales[layer_id]
            groups[key] = {
                "params": [],
                "lr": base_lr * lr_scale,
                "lr_scale": lr_scale,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        groups[key]["params"].append(param)
    return list(groups.values())


def cosine_warmup_lambda(
    epoch: int,
    max_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


class ModelEma:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        import copy

        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for key, ema_val in ema_state.items():
            if key not in model_state:
                continue
            model_val = model_state[key].detach()
            if ema_val.dtype.is_floating_point:
                ema_val.mul_(self.decay).add_(model_val, alpha=1.0 - self.decay)
            else:
                ema_val.copy_(model_val)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    use_amp: bool = False,
    ema: ModelEma | None = None,
    max_grad_norm: float = 0.8,
) -> tuple[float, float]:
    """Run one train/eval epoch and return loss plus accuracy."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0
    scaler = torch.cuda.amp.GradScaler(enabled=is_train and use_amp and device.type == "cuda")
    grad_context = nullcontext() if is_train else torch.no_grad()

    with grad_context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    "LaBraM produced non-finite logits. This usually indicates "
                    "mixed-precision instability; rerun without --amp."
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "LaBraM loss became non-finite. This usually indicates "
                    "mixed-precision instability or invalid cached inputs."
                )

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)

            total_loss += float(loss.item()) * len(y)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += len(y)

    return total_loss / max(total, 1), correct / max(total, 1)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> np.ndarray:
    """Predict labels from a dataloader."""
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    "LaBraM produced non-finite logits during prediction."
                )
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_faced_labram(
    data_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    build_cache: bool = False,
    max_epochs: int = 10,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 0.05,
    min_lr: float = 1e-6,
    warmup_epochs: int = 5,
    layer_decay: float = 0.65,
    use_model_ema: bool = True,
    model_ema_decay: float = 0.9999,
    max_grad_norm: float = 0.8,
    val_subject_fraction: float = 0.1,
    test_subject_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    seed: int = 42,
    num_workers: int = 0,
    limit_recordings: int | None = None,
    pretrained_id: str = DEFAULT_PRETRAINED_ID,
    freeze_encoder: bool = False,
    use_amp: bool = False,
    scale: float = 1e6,
    hf_cache_dir: Path = Path(DEFAULT_HF_CACHE_DIR),
    input_divisor: float = 100.0,
    apply_preprocessing: bool = True,
    allow_cpu: bool = False,
    label_smoothing: float = 0.0,
) -> dict:
    """Fine-tune LaBraM and evaluate FACED emotion classification."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    set_runtime_cache_dirs(hf_cache_dir, output_dir)

    if build_cache or not (cache_dir / "metadata.npz").exists():
        cache = build_raw_cache(
            data_dir=data_dir,
            cache_dir=cache_dir,
            limit_recordings=limit_recordings,
            scale=scale,
            apply_preprocessing=apply_preprocessing,
        )
    else:
        cache = load_raw_cache(cache_dir)

    train_mask, val_mask, test_mask = split_subjects(
        cache.subjects,
        val_subject_fraction=val_subject_fraction,
        test_subject_fraction=test_subject_fraction,
        seed=seed,
    )
    train_loader = make_loader(
        cache, train_mask, batch_size, True, num_workers, input_divisor
    )
    val_loader = make_loader(
        cache, val_mask, batch_size, False, num_workers, input_divisor
    )
    test_loader = make_loader(
        cache, test_mask, batch_size, False, num_workers, input_divisor
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available for LaBraM training. "
            "Refusing to run on CPU. "
            "If you really want CPU, pass --allow-cpu."
        )
    model = load_labram(
        cache=cache,
        pretrained_id=pretrained_id,
        num_classes=len(cache.label_names),
        freeze_encoder=freeze_encoder,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(cache.labels, train_mask, len(cache.label_names)).to(device),
        label_smoothing=label_smoothing,
    )
    param_groups = layer_decay_param_groups(
        model=model,
        base_lr=lr,
        weight_decay=weight_decay,
        layer_decay=layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
    min_lr_ratio = min_lr / max(lr, 1e-12)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_warmup_lambda(
            epoch=epoch,
            max_epochs=max_epochs,
            warmup_epochs=warmup_epochs,
            min_lr_ratio=min_lr_ratio,
        ),
    )
    ema = ModelEma(model, decay=model_ema_decay) if use_model_ema else None

    best_path = output_dir / "best.pt"
    best_val_acc = -1.0
    best_epoch = -1
    stale_epochs = 0
    history = []

    logger.info(
        "FACED LaBraM cache: X=%s labels=%s scale=%s",
        cache.windows.shape,
        cache.label_names,
        cache.scale,
    )
    logger.info(
        "Subject-disjoint split: train=%d, val=%d, test=%d windows",
        int(train_mask.sum()),
        int(val_mask.sum()),
        int(test_mask.sum()),
    )
    logger.info("Training on %s", device)
    logger.info("AMP mixed precision: %s", "enabled" if use_amp else "disabled")
    logger.info(
        "Paper-faithful settings: wd=%.4f min_lr=%.2e warmup=%d layer_decay=%.2f ema=%s clip=%.2f",
        weight_decay,
        min_lr,
        warmup_epochs,
        layer_decay,
        use_model_ema,
        max_grad_norm,
    )

    run_config = {
        "data_dir": str(data_dir),
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "build_cache": build_cache,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "min_lr": min_lr,
        "warmup_epochs": warmup_epochs,
        "layer_decay": layer_decay,
        "use_model_ema": use_model_ema,
        "model_ema_decay": model_ema_decay,
        "max_grad_norm": max_grad_norm,
        "val_subject_fraction": val_subject_fraction,
        "test_subject_fraction": test_subject_fraction,
        "early_stopping_patience": early_stopping_patience,
        "seed": seed,
        "num_workers": num_workers,
        "limit_recordings": limit_recordings,
        "pretrained_id": pretrained_id,
        "freeze_encoder": freeze_encoder,
        "use_amp": use_amp,
        "scale": scale,
        "hf_cache_dir": str(hf_cache_dir),
        "input_divisor": input_divisor,
        "apply_preprocessing": apply_preprocessing,
        "allow_cpu": allow_cpu,
        "label_smoothing": label_smoothing,
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    initial_train_loss, initial_train_acc = run_epoch(
        model, train_loader, criterion, device, use_amp=use_amp
    )
    initial_val_loss, initial_val_acc = run_epoch(
        model, val_loader, criterion, device, use_amp=use_amp
    )
    initial_test_loss, initial_test_acc = run_epoch(
        model, test_loader, criterion, device, use_amp=use_amp
    )
    initial_metrics = {
        "train_loss": initial_train_loss,
        "train_accuracy": initial_train_acc,
        "val_loss": initial_val_loss,
        "val_accuracy": initial_val_acc,
        "test_loss": initial_test_loss,
        "test_accuracy": initial_test_acc,
    }
    logger.info(
        "initial train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f "
        "test_loss=%.4f test_acc=%.4f",
        initial_train_loss,
        initial_train_acc,
        initial_val_loss,
        initial_val_acc,
        initial_test_loss,
        initial_test_acc,
    )

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            use_amp=use_amp,
            ema=ema,
            max_grad_norm=max_grad_norm,
        )
        scheduler.step()
        eval_model = ema.module if ema is not None else model
        val_loss, val_acc = run_epoch(
            eval_model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": float(optimizer.param_groups[0]["lr"]),
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
                    "model_ema_state_dict": ema.module.state_dict() if ema is not None else None,
                    "pretrained_id": pretrained_id,
                    "num_classes": len(cache.label_names),
                    "label_names": cache.label_names,
                    "channel_names": cache.channel_names,
                    "sampling_rate": cache.sampling_rate,
                    "n_times": cache.windows.shape[2],
                    "freeze_encoder": freeze_encoder,
                    "cache_dir": str(cache_dir),
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
                logger.info("Early stopping after %d stale epochs", stale_epochs)
                break

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        if use_model_ema and checkpoint.get("model_ema_state_dict") is not None:
            model.load_state_dict(checkpoint["model_ema_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

    val_pred = predict(model, val_loader, device, use_amp=use_amp)
    test_pred = predict(model, test_loader, device, use_amp=use_amp)
    val_metrics = evaluate_predictions(cache.labels[val_mask], val_pred, cache.label_names)
    test_metrics = evaluate_predictions(cache.labels[test_mask], test_pred, cache.label_names)

    results = {
        "dataset": "braindecode/faced",
        "task": "8-class emotion classification",
        "model": "labram",
        "pretrained_id": pretrained_id,
        "freeze_encoder": freeze_encoder,
        "split": "subject-disjoint random split",
        "seed": seed,
        "n_samples": int(len(cache.labels)),
        "input_shape": [int(v) for v in cache.windows.shape[1:]],
        "sampling_rate": cache.sampling_rate,
        "scale": cache.scale,
        "input_divisor": input_divisor,
        "apply_preprocessing": apply_preprocessing,
        "label_names": cache.label_names,
        "channel_names": cache.channel_names,
        "split_sizes": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "initial": initial_metrics,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path) if best_path.exists() else None,
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
    parser = argparse.ArgumentParser(description="Fine-tune LaBraM on FACED")
    add_config_arg(parser)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--layer-decay", type=float, default=0.65)
    parser.add_argument("--model-ema-decay", type=float, default=0.9999)
    parser.add_argument("--max-grad-norm", type=float, default=0.8)
    parser.add_argument("--val-subject-fraction", type=float, default=0.1)
    parser.add_argument("--test-subject-fraction", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pretrained-id", type=str, default=DEFAULT_PRETRAINED_ID)
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument("--scale", type=float, default=1e6)
    parser.add_argument(
        "--input-divisor",
        type=float,
        default=100.0,
        help="Final scaling divisor before LaBraM forward pass (paper uses 100).",
    )
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Disable 0.1-75 Hz bandpass + 50 Hz notch cache preprocessing.",
    )
    parser.add_argument(
        "--disable-model-ema",
        action="store_true",
        help="Disable model EMA during fine-tuning.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU training when CUDA is unavailable (not recommended).",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Only fine-tune classifier/adaptation parameters.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA mixed precision. Disabled by default because LaBraM can produce NaNs here.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Deprecated compatibility flag; AMP is disabled unless --amp is passed.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for CrossEntropyLoss (e.g. 0.1).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use five recordings and one epoch for a fast loader/training check.",
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
    max_epochs = args.max_epochs or training.get("max_epochs", 10)
    seed = args.seed if args.seed is not None else training.get("seed", 42)
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else training.get("num_workers", 0)
    )
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    limit_recordings = None

    if args.smoke_test:
        max_epochs = 1
        limit_recordings = 5
        args.build_cache = True
        if args.cache_dir == DEFAULT_CACHE_DIR:
            cache_dir = Path("data/processed/faced/smoke_labram_raw_float32")
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            output_dir = Path("checkpoints/faced/smoke_labram")

    train_faced_labram(
        data_dir=data_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        build_cache=args.build_cache,
        max_epochs=max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        layer_decay=args.layer_decay,
        use_model_ema=not args.disable_model_ema,
        model_ema_decay=args.model_ema_decay,
        max_grad_norm=args.max_grad_norm,
        val_subject_fraction=args.val_subject_fraction,
        test_subject_fraction=args.test_subject_fraction,
        early_stopping_patience=args.early_stopping_patience,
        seed=seed,
        num_workers=num_workers,
        limit_recordings=limit_recordings,
        pretrained_id=args.pretrained_id,
        freeze_encoder=args.freeze_encoder,
        use_amp=args.amp and not args.disable_amp,
        scale=args.scale,
        hf_cache_dir=Path(
            args.hf_cache_dir
            or paths.get("hf_cache_dir")
            or DEFAULT_HF_CACHE_DIR
        ),
        input_divisor=args.input_divisor,
        apply_preprocessing=not args.no_preprocessing,
        allow_cpu=args.allow_cpu,
        label_smoothing=args.label_smoothing,
    )


if __name__ == "__main__":
    main()
