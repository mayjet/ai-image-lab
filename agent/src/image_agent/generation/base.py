"""Abstract base for generation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GenerationRequest:
    """Everything a backend needs to generate an image."""

    job_id: str
    prompt: str
    negative_constraints: list[str] = field(default_factory=list)
    width: int = 512
    height: int = 512
    steps: int = 4
    seed: Optional[int] = None
    # Resolved paths to reference images (user_original or agent_derived)
    reference_image_paths: list[str] = field(default_factory=list)
    # Output directory where the backend writes results
    output_dir: str = "/workspace/agent/runs"
    # Backend-specific overrides
    extra: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Output from a backend run."""

    job_id: str
    success: bool
    output_paths: list[str] = field(default_factory=list)
    seed_used: Optional[int] = None
    backend: str = ""
    model_id: str = ""
    model_revision: Optional[str] = None
    vram_peak_mib: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


class GenerationBackend(ABC):
    """Interface that every generation backend must implement."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Execute a generation job.

        Must run model inference in a subprocess so that VRAM is fully released
        when the subprocess exits.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend dependencies are present."""
        ...
