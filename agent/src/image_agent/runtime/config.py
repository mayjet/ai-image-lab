"""Load the checked-in YAML configuration without creating mutable global state."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def config_dir() -> Path:
    override = os.environ.get("IMAGE_AGENT_CONFIG_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "configs"


@lru_cache(maxsize=8)
def load_config(name: str) -> dict[str, Any]:
    path = config_dir() / name
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data
