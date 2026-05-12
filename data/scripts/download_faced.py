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
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import add_config_arg, load_config


DEFAULT_REPO_ID = "braindecode/faced"
DEFAULT_OUTPUT_DIR = "data/raw/faced"


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

    zarr_dirs = [p for p in output_dir.rglob("*.zarr") if p.is_dir()]
    if zarr_dirs:
        print("  Zarr datasets:")
        for path in zarr_dirs[:5]:
            print(f"    - {path.relative_to(output_dir)}")
        if len(zarr_dirs) > 5:
            print(f"    ... {len(zarr_dirs) - 5} more")
    else:
        print("  Zarr datasets: none found")

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

    print("\nFACED download ready.")


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})

    download_faced(
        output_dir=Path(args.output_dir),
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=paths.get("hf_cache_dir"),
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()
