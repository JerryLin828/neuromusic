"""Utilities for the local braindecode/FACED Zarr mirror.

The downloaded mirror stores one Zarr v3 group per subject. We keep this
reader small and dependency-light so baseline experiments can run before
bringing in Braindecode or TorchEEG.
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.signal import welch


FACED_LABELS = [
    "Anger",
    "Disgust",
    "Fear",
    "Sadness",
    "Amusement",
    "Inspiration",
    "Joy",
    "Tenderness",
]

FACED_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 30.0),
    "gamma": (30.0, 47.0),
}


@dataclass(frozen=True)
class FacedRecording:
    """Metadata for one subject recording in the local FACED Zarr mirror."""

    path: Path
    subject: str
    shape: tuple[int, int, int]
    chunk_shape: tuple[int, int, int]
    sampling_rate: float


def find_faced_zarr_root(data_dir: str | Path) -> Path:
    """Return the non-cache braindecode/FACED dataset.zarr path."""
    data_dir = Path(data_dir)
    preferred = data_dir / "sourcedata" / "braindecode" / "dataset.zarr"
    if preferred.exists():
        return preferred

    matches = [
        p for p in data_dir.rglob("dataset.zarr")
        if p.is_dir() and ".cache" not in p.parts
    ]
    if not matches:
        raise FileNotFoundError(
            f"No FACED dataset.zarr found under {data_dir}. "
            "Run: python data/scripts/download_faced.py"
        )
    return matches[0]


def list_recordings(data_dir: str | Path) -> list[FacedRecording]:
    """List all subject recordings in the local FACED Zarr mirror."""
    zarr_root = find_faced_zarr_root(data_dir)
    recordings = sorted(
        [
            p for p in zarr_root.iterdir()
            if p.is_dir() and p.name.startswith("recording_")
        ],
        key=lambda p: int(p.name.split("_", 1)[1]),
    )

    parsed = []
    for recording in recordings:
        with open(recording / "zarr.json") as f:
            group_meta = json.load(f)
        with open(recording / "data" / "zarr.json") as f:
            data_meta = json.load(f)

        info = group_meta["attributes"]["info"]
        description = group_meta["attributes"]["description"]
        parsed.append(
            FacedRecording(
                path=recording,
                subject=str(description["subject"]),
                shape=tuple(int(v) for v in data_meta["shape"]),
                chunk_shape=tuple(
                    int(v)
                    for v in data_meta["chunk_grid"]["configuration"]["chunk_shape"]
                ),
                sampling_rate=float(info["sfreq"]),
            )
        )
    return parsed


def read_metadata(recording: FacedRecording) -> list[dict[str, str]]:
    """Read per-window metadata rows for a recording."""
    metadata_path = recording.path / "metadata.tsv"
    with open(metadata_path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def iter_recording_chunks(recording: FacedRecording) -> Iterator[np.ndarray]:
    """Yield data chunks shaped ``(windows, channels, timepoints)``."""
    n_windows, n_channels, n_times = recording.shape
    chunk_windows = recording.chunk_shape[0]
    n_chunks = int(np.ceil(n_windows / chunk_windows))

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_windows
        stop = min(start + chunk_windows, n_windows)
        expected_shape = (stop - start, n_channels, n_times)
        chunk_path = recording.path / "data" / "c" / str(chunk_idx) / "0" / "0"
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing FACED Zarr chunk: {chunk_path}")

        raw = gzip.decompress(chunk_path.read_bytes())
        values = np.frombuffer(raw, dtype="<f4")
        expected_size = int(np.prod(expected_shape))
        full_size = int(np.prod(recording.chunk_shape))

        if values.size == expected_size:
            chunk = values.reshape(expected_shape)
        elif values.size == full_size:
            chunk = values.reshape(recording.chunk_shape)[: expected_shape[0]]
        else:
            raise ValueError(
                f"Unexpected chunk size in {chunk_path}: {values.size} values, "
                f"expected {expected_size} or {full_size}"
            )

        yield chunk.astype(np.float32, copy=False)


def log_bandpower_features(
    windows: np.ndarray,
    sampling_rate: float,
    nperseg_seconds: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute compact log bandpower features for FACED windows.

    Args:
        windows: EEG data shaped ``(N, C, T)``.
        sampling_rate: Sampling rate in Hz.
        nperseg_seconds: Welch segment length.
        eps: Numerical floor before log.

    Returns:
        Array shaped ``(N, C * n_bands)``.
    """
    nperseg = min(windows.shape[-1], int(round(sampling_rate * nperseg_seconds)))
    freqs, psd = welch(
        windows,
        fs=sampling_rate,
        nperseg=nperseg,
        axis=-1,
        scaling="density",
    )

    features = []
    for low, high in FACED_BANDS.values():
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            raise ValueError(f"No Welch bins in band {low}-{high} Hz")
        band_power = np.trapz(psd[..., mask], freqs[mask], axis=-1)
        features.append(np.log(band_power + eps))

    return np.stack(features, axis=-1).reshape(windows.shape[0], -1).astype(np.float32)


def build_feature_cache(
    data_dir: str | Path,
    cache_path: str | Path,
    limit_recordings: int | None = None,
) -> Path:
    """Extract log-bandpower features from FACED and save them as an NPZ cache."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    label_to_id = {label: i for i, label in enumerate(FACED_LABELS)}
    features = []
    labels = []
    subjects = []
    videos = []
    binary_labels = []

    recordings = list_recordings(data_dir)
    if limit_recordings is not None:
        recordings = recordings[:limit_recordings]

    for recording in recordings:
        rows = read_metadata(recording)
        row_offset = 0
        for chunk in iter_recording_chunks(recording):
            chunk_features = log_bandpower_features(chunk, recording.sampling_rate)
            chunk_rows = rows[row_offset: row_offset + len(chunk)]
            row_offset += len(chunk)

            features.append(chunk_features)
            labels.extend(label_to_id[row["emotion_label"]] for row in chunk_rows)
            subjects.extend([recording.subject] * len(chunk_rows))
            videos.extend(int(row["video_index"]) for row in chunk_rows)
            binary_labels.extend(row["binary_label"] for row in chunk_rows)

        if row_offset != len(rows):
            raise ValueError(
                f"Metadata/data length mismatch for {recording.path.name}: "
                f"used {row_offset} rows but found {len(rows)}"
            )

    X = np.concatenate(features, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    subject_ids = np.asarray(subjects)
    video_index = np.asarray(videos, dtype=np.int64)
    binary_label = np.asarray(binary_labels)

    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        subject_id=subject_ids,
        video_index=video_index,
        binary_label=binary_label,
        label_names=np.asarray(FACED_LABELS),
        band_names=np.asarray(list(FACED_BANDS.keys())),
    )
    return cache_path


def load_feature_cache(cache_path: str | Path) -> dict[str, np.ndarray]:
    """Load a FACED feature cache created by :func:`build_feature_cache`."""
    with np.load(cache_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
