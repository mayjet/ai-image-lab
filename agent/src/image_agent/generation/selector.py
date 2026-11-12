"""Backend selector with fallback logic."""

from __future__ import annotations

from typing import Optional

from image_agent.errors import BackendError
from image_agent.generation.base import GenerationBackend, GenerationRequest, GenerationResult
from image_agent.generation.mock_backend import MockBackend
from image_agent.generation.flux2_backend import Flux2Backend
from image_agent.generation.flux3_backend import Flux3Backend


class _Flux3WithFlux2Fallback(GenerationBackend):
    """Keep the experimental FLUX.3 path optional on 8 GB Turing GPUs."""

    def __init__(self, primary: Flux3Backend, fallback: Flux2Backend) -> None:
        self.primary = primary
        self.fallback = fallback

    @property
    def name(self) -> str:
        return "flux3"

    def is_available(self) -> bool:
        return self.primary.is_available() or self.fallback.is_available()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        reason = "FLUX.3 ComfyUI runtime is not installed"
        if self.primary.is_available():
            try:
                return self.primary.generate(request)
            except Exception as exc:
                reason = f"FLUX.3 failed: {exc}"
        result = self.fallback.generate(request)
        result.fallback_used = True
        result.fallback_reason = reason
        return result


def get_backend(
    name: str,
    fallback_to_mock: bool = False,
    **kwargs,
) -> GenerationBackend:
    """
    Return the named backend.

    If the requested backend is unavailable and fallback_to_mock is True,
    returns a MockBackend instead (useful for testing without GPU).
    """
    flux2 = Flux2Backend(**{k: v for k, v in kwargs.items() if _is_flux2_kwarg(k)})
    flux3 = Flux3Backend(**{k: v for k, v in kwargs.items() if _is_flux3_kwarg(k)})
    backends: dict[str, GenerationBackend] = {
        "flux2": flux2,
        "flux3": _Flux3WithFlux2Fallback(flux3, flux2),
        "mock": MockBackend(),
    }

    backend = backends.get(name)
    if backend is None:
        raise BackendError(f"Unknown backend {name!r}")

    if not backend.is_available():
        if fallback_to_mock:
            return MockBackend()
        raise BackendError(
            f"Backend {name!r} is not available in this environment. "
            "Install required dependencies or use backend='mock' for testing."
        )

    return backend


def _is_flux2_kwarg(key: str) -> bool:
    return key in {
        "model_id", "cache_dir", "timeout", "quantization", "offload",
        "guidance_scale", "max_sequence_length",
    }


def _is_flux3_kwarg(key: str) -> bool:
    return key in {
        "comfyui_dir", "workflow_path", "model_cache_dir", "timeout", "port",
    }
