"""Shared I/O and config utilities."""

import argparse
from pathlib import Path
import yaml
import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Add the standard --config argument to any argparse parser."""
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG})",
    )


def save_json(data: dict, path: str | Path):
    """Save a dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
