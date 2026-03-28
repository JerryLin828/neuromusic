"""
Download DREAMER EEG emotion dataset from HuggingFace.

Downloads both DREAMERA (arousal) and DREAMERV (valence) datasets.
These are preprocessed by the MONSTER benchmark (Monash) using TorchEEG:
  - 14-channel EEG at 128 Hz from Emotiv EPOC headset
  - 2-second windows (256 samples), already filtered and cropped
  - Binary labels: arousal (low/high), valence (low/high)
  - 5-fold cross-validation splits by participant
  - 170,246 total samples from 23 subjects x 18 stimuli

Source: https://huggingface.co/datasets/monster-monash/DREAMERA
Paper: Katsigiannis & Ramzan, IEEE JBHI 2018 (DREAMER)
        Foumani et al., 2025 (MONSTER benchmark)

Usage:
    python data/scripts/download_dreamer.py
    python data/scripts/download_dreamer.py --config configs/default.yaml
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from src.utils.io import load_config, add_config_arg


def download_dreamer(output_dir: Path, cache_dir: str | None = None):
    """Download DREAMERA and DREAMERV datasets from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_HUB_CACHE"] = cache_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "arousal": {
            "repo": "monster-monash/DREAMERA",
            "files": {
                "X": "DREAMERA_X.npy",
                "y": "DREAMERA_y.npy",
                "metadata": "DREAMERA_metadata.npy",
            },
            "folds": [f"test_indices_fold_{i}.txt" for i in range(5)],
        },
        "valence": {
            "repo": "monster-monash/DREAMERV",
            "files": {
                "X": "DREAMERV_X.npy",
                "y": "DREAMERV_y.npy",
                "metadata": "DREAMERV_metadata.npy",
            },
            "folds": [f"test_indices_fold_{i}.txt" for i in range(5)],
        },
    }

    for dim_name, ds_info in datasets.items():
        dim_dir = output_dir / dim_name
        dim_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Downloading DREAMER {dim_name} from {ds_info['repo']}")
        print(f"{'='*60}")

        for local_name, remote_name in ds_info["files"].items():
            target = dim_dir / f"{local_name}.npy"
            if target.exists():
                print(f"  [skip] {target.name} already exists")
                continue
            print(f"  Downloading {remote_name}...")
            cached_path = hf_hub_download(
                repo_id=ds_info["repo"],
                filename=remote_name,
                repo_type="dataset",
                force_download=True,
            )
            shutil.copy2(cached_path, target)
            cached_size = Path(cached_path).stat().st_size
            target_size = target.stat().st_size
            if target_size != cached_size:
                raise RuntimeError(
                    f"Copy failed: {target} is {target_size} bytes "
                    f"but source is {cached_size} bytes"
                )
            print(f"  -> {target} ({target_size / 1e6:.1f} MB)")

        for fold_file in ds_info["folds"]:
            target = dim_dir / fold_file
            if target.exists():
                continue
            cached_path = hf_hub_download(
                repo_id=ds_info["repo"],
                filename=fold_file,
                repo_type="dataset",
            )
            shutil.copy2(cached_path, target)

        print(f"  Downloaded 5 fold split files")

    _verify(output_dir)


def _verify(output_dir: Path):
    """Verify the downloaded data."""
    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")

    for dim in ["arousal", "valence"]:
        dim_dir = output_dir / dim
        X = np.load(dim_dir / "X.npy")
        y = np.load(dim_dir / "y.npy")

        class_counts = dict(zip(*np.unique(y, return_counts=True)))
        print(f"\n  {dim}:")
        print(f"    X shape: {X.shape}  (samples, channels, timepoints)")
        print(f"    y shape: {y.shape}  classes: {class_counts}")
        print(f"    X dtype: {X.dtype}, y dtype: {y.dtype}")
        print(f"    X range: [{X[0].min():.2f}, {X[0].max():.2f}] (sample 0)")

        n_folds = sum(1 for f in dim_dir.glob("test_indices_fold_*.txt"))
        print(f"    Folds: {n_folds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DREAMER from HuggingFace")
    add_config_arg(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})

    download_dreamer(
        output_dir=Path(paths.get("data_dir", "data/raw/dreamer")),
        cache_dir=paths.get("hf_cache_dir"),
    )
