"""Image generation backends public API."""

from image_agent.generation.base import GenerationBackend, GenerationRequest, GenerationResult
from image_agent.generation.mock_backend import MockBackend
from image_agent.generation.flux2_backend import Flux2Backend
from image_agent.generation.flux3_backend import Flux3Backend
from image_agent.generation.selector import get_backend

__all__ = [
    "GenerationBackend",
    "GenerationRequest",
    "GenerationResult",
    "MockBackend",
    "Flux2Backend",
    "Flux3Backend",
    "get_backend",
]
