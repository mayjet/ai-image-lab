"""Central project paths shared by scripts and notebooks."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
MODELS_ROOT = PROJECT_ROOT / "models"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CACHE_ROOT = PROJECT_ROOT / "cache"
PROMPTS_ROOT = PROJECT_ROOT / "prompts"
VENDOR_ROOT = PROJECT_ROOT / "vendor"


def env_path(name: str, default: Path) -> Path:
    """Return an absolute environment override or the project-relative default."""
    value = os.environ.get(name, "").strip()
    path = Path(value).expanduser() if value else default
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_dataset_root() -> Path:
    return env_path("AI_IMAGE_DATASET_ROOT", DATASETS_ROOT / "works")


def sd15_model_root() -> Path:
    return env_path("AI_IMAGE_SD15_MODEL_ROOT", MODELS_ROOT / "sd15")


def sd15_output_root() -> Path:
    return env_path("AI_IMAGE_SD15_OUTPUT_ROOT", OUTPUTS_ROOT / "sd15")


def sd_scripts_root() -> Path:
    return env_path("AI_IMAGE_SD_SCRIPTS_ROOT", VENDOR_ROOT / "sd-scripts")


def flux2_model_root() -> Path:
    return env_path("AI_IMAGE_FLUX2_MODEL_ROOT", MODELS_ROOT / "flux2")


def flux2_output_root() -> Path:
    return env_path("AI_IMAGE_FLUX2_OUTPUT_ROOT", OUTPUTS_ROOT / "flux2")


def diffusers_root() -> Path:
    return env_path("AI_IMAGE_DIFFUSERS_ROOT", VENDOR_ROOT / "diffusers")
