"""YAML configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads the YAML config. Relative paths are resolved against the repository root."""
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str) -> Path:
    """Resolves a config-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
