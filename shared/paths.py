"""Central project paths shared by scripts and notebooks."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
LORAS_ROOT = PROJECT_ROOT / "loras"
SD15_ROOT = PROJECT_ROOT / "sd15"
FLUX2_ROOT = PROJECT_ROOT / "flux2"


def env_path(name: str, default: Path) -> Path:
    """Return an absolute environment override or the project-relative default."""
    value = os.environ.get(name, "").strip()
    path = Path(value).expanduser() if value else default
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_dataset_root() -> Path:
    return env_path("AI_IMAGE_DATASET_ROOT", DATASETS_ROOT / "works")


def sd15_model_root() -> Path:
    return env_path(
        "AI_IMAGE_SD15_MODEL_ROOT",
        LORAS_ROOT / "sd15" / "stablediffusionapi--counterfeit-v30",
    )


def sd15_output_root() -> Path:
    return env_path("AI_IMAGE_SD15_OUTPUT_ROOT", SD15_ROOT / "outputs")


def sd_scripts_root() -> Path:
    return env_path("AI_IMAGE_SD_SCRIPTS_ROOT", SD15_ROOT / "vendor" / "sd-scripts")


def flux2_model_root() -> Path:
    return env_path(
        "AI_IMAGE_FLUX2_MODEL_ROOT",
        LORAS_ROOT / "flux2" / "black-forest-labs--flux.2-klein-base-4b",
    )


def flux2_output_root() -> Path:
    return env_path("AI_IMAGE_FLUX2_OUTPUT_ROOT", FLUX2_ROOT / "outputs")


def diffusers_root() -> Path:
    return env_path("AI_IMAGE_DIFFUSERS_ROOT", FLUX2_ROOT / "vendor" / "diffusers")
