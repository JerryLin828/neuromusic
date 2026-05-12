"""
Download the FACED EEG emotion dataset mirror from Hugging Face.

Default source:
  - braindecode/faced

This Hugging Face mirror is a practical, windowed Zarr version of FACED:
  - 123 recordings
  - 26 EEG channels
  - 200 Hz
  - 19,217 windows/samples
  - BIDS-inspired metadata with events/channels JSON/TSV files
  - per-window metadata with emotion_label, binary_label, video_index, and
    self-report columns

The canonical EEGDash pointer is EEGDash/nm000112, but that route streams the
larger canonical OpenNeuro/NEMAR data through eegdash. For this project, the
braindecode/faced mirror is easier to cache under data/raw and use for model
development.

Usage:
    python data/scripts/download_faced.py
    python data/scripts/download_faced.py --output-dir data/raw/faced
    python data/scripts/download_faced.py --repo-id EEGDash/nm000112
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import add_config_arg, load_config


DEFAULT_REPO_ID = "braindecode/faced"
DEFAULT_OUTPUT_DIR = "data/raw/faced"
EXPECTED_BRAINDECODE_RECORDINGS = 123
EXPECTED_BRAINDECODE_WINDOWS = 19217
EXPECTED_BRAINDECODE_CHANNELS = 26
EXPECTED_BRAINDECODE_SAMPLING_RATE = 200.0


def _set_hf_cache(cache_dir: str | None) -> None:
    """Route Hugging Face cache to the configured cluster path if provided."""
    if not cache_dir:
        return
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir


def download_faced(
    output_dir: Path,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    cache_dir: str | None = None,
    force_download: bool = False,
) -> Path:
    """Download a FACED Hugging Face dataset snapshot into output_dir."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub: pip install huggingface_hub") from exc

    _set_hf_cache(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Downloading FACED dataset")
    print(f"  repo:       {repo_id}")
    print(f"  revision:   {revision or 'default'}")
    print(f"  output_dir: {output_dir}")
    print(f"{'=' * 60}")

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(output_dir),
        force_download=force_download,
    )

    snapshot_path = Path(snapshot_path)
    _write_manifest(output_dir, repo_id, revision, snapshot_path)
    _verify(output_dir, repo_id)
    return output_dir


def _write_manifest(
    output_dir: Path,
    repo_id: str,
    revision: str | None,
    snapshot_path: Path,
) -> None:
    """Write lightweight provenance for downstream loaders and reports."""
    manifest = {
        "dataset": "FACED",
        "repo_id": repo_id,
        "revision": revision,
        "snapshot_path": str(snapshot_path),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Default braindecode/faced mirror is a windowed Zarr dataset. "
            "Use EEGDash/nm000112 with eegdash for the larger canonical source."
        ),
    }
    with open(output_dir / "download_info.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _verify(output_dir: Path, repo_id: str) -> None:
    """Print a quick structural verification of the downloaded snapshot."""
    print(f"\n{'=' * 60}")
    print("Verification")
    print(f"{'=' * 60}")

    all_files = [p for p in output_dir.rglob("*") if p.is_file()]
    total_size = sum(p.stat().st_size for p in all_files)
    print(f"  Files: {len(all_files):,}")
    print(f"  Size:  {total_size / 1e9:.2f} GB")

    zarr_dirs = _payload_zarr_dirs(output_dir)
    cached_zarr_dirs = [
        p for p in output_dir.rglob("*.zarr")
        if p.is_dir() and ".cache" in p.parts
    ]
    if zarr_dirs:
        print("  Zarr datasets:")
        for path in zarr_dirs[:5]:
            print(f"    - {path.relative_to(output_dir)}")
        if len(zarr_dirs) > 5:
            print(f"    ... {len(zarr_dirs) - 5} more")
    else:
        print("  Zarr datasets: none found")
    if cached_zarr_dirs:
        print(f"  Cached Zarr mirrors: {len(cached_zarr_dirs):,} (ignored for payload checks)")

    expected_metadata = [
        "dataset_description.json",
        "participants.tsv",
        "download_info.json",
    ]
    for name in expected_metadata:
        matches = list(output_dir.rglob(name))
        status = "found" if matches else "missing"
        print(f"  {name}: {status}")

    event_files = list(output_dir.rglob("*_events.tsv"))
    channel_files = list(output_dir.rglob("*_channels.tsv"))
    print(f"  Event files:   {len(event_files):,}")
    print(f"  Channel files: {len(channel_files):,}")

    if repo_id == DEFAULT_REPO_ID and not zarr_dirs:
        raise RuntimeError(
            "Downloaded braindecode/faced but no .zarr dataset was found. "
            "The repository layout may have changed."
        )

    if repo_id == DEFAULT_REPO_ID and zarr_dirs:
        summary = summarize_braindecode_faced(output_dir)
        print("\n  Braindecode FACED payload:")
        print(f"    Zarr root:     {summary['zarr_root'].relative_to(output_dir)}")
        print(f"    Recordings:    {summary['n_recordings']:,}")
        print(f"    Subjects:      {summary['n_subjects']:,}")
        print(f"    Windows:       {summary['n_windows']:,}")
        print(f"    Channels:      {summary['n_channels']}")
        print(f"    Sampling rate: {summary['sampling_rate']} Hz")
        print(f"    Videos:        {len(summary['video_counts']):,}")
        print(f"    Classes:       {len(summary['emotion_counts']):,}")
        for label, count in sorted(summary["emotion_counts"].items()):
            print(f"      - {label}: {count:,}")

        _assert_braindecode_summary(summary)

    print("\nFACED download ready.")


def _payload_zarr_dirs(output_dir: Path) -> list[Path]:
    """Return non-cache Zarr payload directories."""
    return [
        p for p in output_dir.rglob("*.zarr")
        if p.is_dir() and ".cache" not in p.parts
    ]


def summarize_braindecode_faced(output_dir: Path) -> dict:
    """Summarize the local braindecode/faced Zarr payload without loading arrays."""
    zarr_dirs = _payload_zarr_dirs(output_dir)
    if not zarr_dirs:
        raise FileNotFoundError(
            f"No non-cache .zarr payload found under {output_dir}. "
            "Re-run python data/scripts/download_faced.py."
        )

    preferred = output_dir / "sourcedata" / "braindecode" / "dataset.zarr"
    zarr_root = preferred if preferred.exists() else zarr_dirs[0]
    recordings = sorted(
        [
            p for p in zarr_root.iterdir()
            if p.is_dir() and p.name.startswith("recording_")
        ],
        key=lambda p: int(p.name.split("_", 1)[1]),
    )

    emotion_counts: Counter[str] = Counter()
    binary_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    shape_counts: Counter[tuple[int, ...]] = Counter()
    subjects: set[str] = set()
    rows_per_recording: dict[str, int] = {}
    n_windows = 0
    n_channels: int | None = None
    sampling_rate: float | None = None

    for recording in recordings:
        group_json = recording / "zarr.json"
        data_json = recording / "data" / "zarr.json"
        metadata_tsv = recording / "metadata.tsv"
        if not group_json.exists() or not data_json.exists() or not metadata_tsv.exists():
            raise FileNotFoundError(
                f"Malformed FACED recording at {recording}: expected zarr.json, "
                "data/zarr.json, and metadata.tsv"
            )

        with open(group_json) as f:
            group_meta = json.load(f)
        attrs = group_meta.get("attributes", {})
        description = attrs.get("description", {})
        info = attrs.get("info", {})
        subject = str(description.get("subject", recording.name))
        subjects.add(subject)
        if sampling_rate is None:
            sampling_rate = float(info.get("sfreq", 0.0))

        with open(data_json) as f:
            data_meta = json.load(f)
        shape = tuple(int(v) for v in data_meta.get("shape", []))
        if len(shape) != 3:
            raise ValueError(f"Unexpected FACED array shape in {data_json}: {shape}")
        shape_counts[shape] += 1
        n_windows += shape[0]
        if n_channels is None:
            n_channels = shape[1]

        row_count = 0
        with open(metadata_tsv, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row_count += 1
                emotion_counts[row["emotion_label"]] += 1
                binary_counts[row["binary_label"]] += 1
                video_counts[row["video_index"]] += 1
        rows_per_recording[recording.name] = row_count
        if row_count != shape[0]:
            raise ValueError(
                f"Metadata/data length mismatch in {recording}: "
                f"{row_count} metadata rows vs {shape[0]} array windows"
            )

    return {
        "zarr_root": zarr_root,
        "n_recordings": len(recordings),
        "n_subjects": len(subjects),
        "n_windows": n_windows,
        "n_channels": n_channels,
        "sampling_rate": sampling_rate,
        "shape_counts": dict(shape_counts),
        "emotion_counts": dict(emotion_counts),
        "binary_counts": dict(binary_counts),
        "video_counts": dict(video_counts),
        "rows_per_recording": rows_per_recording,
    }


def _assert_braindecode_summary(summary: dict) -> None:
    """Raise if the local braindecode/faced payload is structurally unusable."""
    errors = []
    if summary["n_recordings"] != EXPECTED_BRAINDECODE_RECORDINGS:
        errors.append(
            f"expected {EXPECTED_BRAINDECODE_RECORDINGS} recordings, "
            f"found {summary['n_recordings']}"
        )
    if summary["n_subjects"] != EXPECTED_BRAINDECODE_RECORDINGS:
        errors.append(
            f"expected {EXPECTED_BRAINDECODE_RECORDINGS} subjects, "
            f"found {summary['n_subjects']}"
        )
    if summary["n_windows"] != EXPECTED_BRAINDECODE_WINDOWS:
        errors.append(
            f"expected {EXPECTED_BRAINDECODE_WINDOWS} windows, "
            f"found {summary['n_windows']}"
        )
    if summary["n_channels"] != EXPECTED_BRAINDECODE_CHANNELS:
        errors.append(
            f"expected {EXPECTED_BRAINDECODE_CHANNELS} channels, "
            f"found {summary['n_channels']}"
        )
    if summary["sampling_rate"] != EXPECTED_BRAINDECODE_SAMPLING_RATE:
        errors.append(
            f"expected {EXPECTED_BRAINDECODE_SAMPLING_RATE} Hz, "
            f"found {summary['sampling_rate']} Hz"
        )
    if not summary["emotion_counts"]:
        errors.append("no emotion labels found in Zarr metadata")
    if errors:
        raise RuntimeError("FACED verification failed: " + "; ".join(errors))

    if "Neutral" not in summary["emotion_counts"]:
        print(
            "\n  Note: this braindecode/faced mirror contains no Neutral windows. "
            "Use it as an 8-class emotion dataset, or fetch the canonical "
            "FACED/Synapse feature files if neutral is required."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FACED from Hugging Face")
    add_config_arg(parser)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to place the dataset snapshot (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision/commit/tag",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force Hugging Face to re-download files",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing FACED download without downloading files",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})

    output_dir = Path(args.output_dir)
    if args.verify_only:
        _verify(output_dir, args.repo_id)
    else:
        download_faced(
            output_dir=output_dir,
            repo_id=args.repo_id,
            revision=args.revision,
            cache_dir=paths.get("hf_cache_dir"),
            force_download=args.force_download,
        )


if __name__ == "__main__":
    main()
