"""
CLI entry point for the Therapeutic Soundtrack Pipeline.

Usage:
    # With a DREAMER sample (no extra flags needed):
    python -m src.pipeline.run --dreamer-sample 42

    # With a .npy file:
    python -m src.pipeline.run --input sample_eeg.npy

    # Override dimension or output name:
    python -m src.pipeline.run --dreamer-sample 42 --dreamer-dim valence --name test1
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from src.pipeline.pipeline import TherapeuticSoundtrackPipeline
from src.utils.io import load_config, add_config_arg


def load_eeg_input(path: str) -> np.ndarray:
    """Load EEG data from a .npy or .npz file."""
    path = Path(path)

    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".npz":
        npz = np.load(path)
        return npz[list(npz.keys())[0]]
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            f"Use .npy (14, 256) for DREAMER format."
        )


def load_dreamer_sample(data_dir: str, dim: str, index: int) -> tuple[np.ndarray, int]:
    """Load a single sample from the downloaded DREAMER dataset."""
    dim_dir = Path(data_dir) / dim
    X = np.load(dim_dir / "X.npy", mmap_mode="r")
    y = np.load(dim_dir / "y.npy")

    if index >= len(X):
        raise IndexError(f"Sample index {index} out of range (max: {len(X)-1})")

    return X[index].copy(), int(y[index])


def main():
    parser = argparse.ArgumentParser(description="Therapeutic Soundtrack Generator")
    add_config_arg(parser)
    parser.add_argument("--input", type=str, default=None,
                        help="Path to EEG input file (.npy, .npz)")
    parser.add_argument("--dreamer-sample", type=int, default=None,
                        help="Load a specific sample from DREAMER dataset by index")
    parser.add_argument("--dreamer-dim", type=str, default="arousal",
                        choices=["arousal", "valence"])
    parser.add_argument("--name", type=str, default="output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    data_dir = paths.get("data_dir", "data/raw/dreamer")
    output_dir = Path(paths.get("output_dir", "outputs"))

    pipeline = TherapeuticSoundtrackPipeline.from_config(args.config)
    pipeline.output_dir = output_dir

    if args.input:
        eeg_data = load_eeg_input(args.input)
    elif args.dreamer_sample is not None:
        eeg_data, ground_truth = load_dreamer_sample(
            data_dir, args.dreamer_dim, args.dreamer_sample,
        )
        gt_label = "high" if ground_truth == 1 else "low"
        print(f"DREAMER sample {args.dreamer_sample}: "
              f"ground truth {args.dreamer_dim} = {gt_label}")
    else:
        parser.error(
            "Provide either --input <file> or --dreamer-sample <index>. "
            "No dummy data is used."
        )

    result = pipeline.generate(eeg_data)
    result.save(pipeline.output_dir, args.name)

    print(f"\n{'='*60}")
    print(f"Result Summary")
    print(f"{'='*60}")
    print(f"  Emotion:  {result.emotion.quadrant} "
          f"(V={result.emotion.valence:.2f}, A={result.emotion.arousal:.2f})")
    print(f"  Prompt:   {result.therapeutic_prompt[:100]}...")
    print(f"  Audio:    {len(result.audio)/result.sample_rate:.1f}s "
          f"at {result.sample_rate} Hz")
    print(f"  Saved to: {pipeline.output_dir}/")


if __name__ == "__main__":
    main()
