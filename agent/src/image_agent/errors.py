"""Custom exception hierarchy for image_agent."""

from __future__ import annotations


class ImageAgentError(Exception):
    """Base class for all image_agent errors."""


# --- Asset / Input errors ---

class AssetError(ImageAgentError):
    """An asset file could not be loaded or validated."""


class DatasetEscapeError(AssetError):
    """A path resolution attempted to escape the dataset root."""


class MarkdownError(ImageAgentError):
    """The input Markdown is invalid or missing required content."""


# --- Planning errors ---

class PlannerError(ImageAgentError):
    """The planner (Qwen) returned an unusable result."""


class JSONExtractionError(PlannerError):
    """JSON could not be extracted from the model's raw output."""


class JSONRepairError(PlannerError):
    """JSON repair failed or exceeded the allowed attempt count."""


class PlanValidationError(PlannerError):
    """The extracted plan failed Pydantic or semantic validation."""


class JobGraphError(PlannerError):
    """The job graph is invalid (cycle, out-of-range counts, etc.)."""


class DuplicatePlanError(PlannerError):
    """The same plan hash has been seen before — stopping to avoid a loop."""


# --- Generation errors ---

class GenerationError(ImageAgentError):
    """An image generation step failed."""


class BackendError(GenerationError):
    """A specific backend (FLUX.2, FLUX.3, mock) raised an error."""


class SubprocessError(GenerationError):
    """A child process exited abnormally or timed out."""


class CudaOOMError(GenerationError):
    """CUDA out-of-memory during generation."""


class ModelLoadError(ImageAgentError):
    """A model could not be loaded (download failure, access denied, etc.)."""


class HuggingFaceDownloadError(ModelLoadError):
    """Hugging Face Hub download or access check failed."""


# --- ComfyUI errors ---

class ComfyUIError(ImageAgentError):
    """ComfyUI-related error."""


class ComfyUIReadinessTimeout(ComfyUIError):
    """ComfyUI did not become ready within the allowed time."""


class ComfyUIJobTimeout(ComfyUIError):
    """A ComfyUI workflow job did not complete within the allowed time."""


# --- Upscale errors ---

class UpscaleError(ImageAgentError):
    """Upscaling failed."""


class UpscalerOOMError(UpscaleError):
    """Upscaler ran out of VRAM."""


# --- Orchestration errors ---

class OrchestrationError(ImageAgentError):
    """The agent orchestration layer hit an unrecoverable state."""


class MaxStepsExceeded(OrchestrationError):
    """The graph step counter reached MAX_GRAPH_STEPS."""


class UserRequiredError(OrchestrationError):
    """The agent cannot proceed without user input."""


# --- Runtime errors ---

class RuntimeError_(ImageAgentError):
    """Runtime diagnostics or resource check failed."""


class DiskSpaceError(RuntimeError_):
    """Insufficient disk space."""


class RAMError(RuntimeError_):
    """Insufficient CPU RAM."""


# --- Tailscale errors (borrowed GPU mode only) ---

class TailscaleAuthTimeout(ImageAgentError):
    """Tailscale authentication did not complete within the allowed time."""
