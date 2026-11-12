"""
FLUX.2 (Diffusers) backend.

Runs inference in a subprocess so VRAM is released on subprocess exit.
No LoRA loading or training.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from image_agent.errors import BackendError, CudaOOMError, SubprocessError
from image_agent.generation.base import GenerationBackend, GenerationRequest, GenerationResult
from image_agent.runtime.logging import get_logger
from image_agent.runtime.subprocess_runner import run_subprocess
from image_agent.runtime.config import load_config

logger = get_logger(__name__)

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_TIMEOUT = 3600  # 60 minutes, including first model download


class Flux2Backend(GenerationBackend):
    """FLUX.2-klein-4B via Diffusers, subprocess-isolated."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        timeout: Optional[int] = None,
        quantization: Optional[str] = None,
        offload: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        max_sequence_length: Optional[int] = None,
    ) -> None:
        model_cfg = load_config("models.yaml").get("flux2", {})
        self.model_id = model_id or model_cfg.get("model_id", DEFAULT_MODEL_ID)
        self.revision = model_cfg.get("revision") or None
        self.cache_dir = cache_dir or "/workspace/agent/.runtime/models"
        self.timeout = timeout or int(model_cfg.get("timeout_seconds", DEFAULT_TIMEOUT))
        self.quantization = quantization or model_cfg.get("quantization", "bnb4")
        self.offload = offload or model_cfg.get("offload", "auto")
        self.guidance_scale = (
            guidance_scale if guidance_scale is not None else float(model_cfg.get("guidance_scale", 1.0))
        )
        self.max_sequence_length = max_sequence_length or int(
            model_cfg.get("max_sequence_length", 256)
        )

    @property
    def name(self) -> str:
        return "flux2"

    def is_available(self) -> bool:
        try:
            import diffusers  # noqa: F401
            import transformers  # noqa: F401
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = {
            "model_id": self.model_id,
            "revision": self.revision,
            "cache_dir": self.cache_dir,
            "quantization": self.quantization,
            "offload": self.offload,
            "guidance_scale": self.guidance_scale,
            "max_sequence_length": self.max_sequence_length,
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "seed": request.seed,
            "output_dir": request.output_dir,
            "job_id": request.job_id,
            "reference_image_paths": request.reference_image_paths[:4],
            "negative_constraints": request.negative_constraints,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(payload, f)
            payload_path = f.name

        try:
            worker = str(Path(__file__).parent / "_flux2_worker.py")
            env = dict(os.environ)
            env["HF_HOME"] = env.get("HF_HOME", "/workspace/agent/.runtime/cache/huggingface")
            env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

            result = run_subprocess(
                [sys.executable, worker, payload_path],
                timeout=self.timeout,
                env=env,
                log_tag="flux2_worker",
                allow_failure=True,
            )
        finally:
            try:
                os.unlink(payload_path)
            except OSError:
                pass

        if result.returncode == 2:
            raise CudaOOMError("FLUX.2 worker reported CUDA OOM")

        if result.returncode != 0:
            raise BackendError(
                f"FLUX.2 worker exited {result.returncode}.\n{result.stderr[-2000:]}"
            )

        try:
            out = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise BackendError(f"FLUX.2 worker output is not valid JSON: {result.stdout[:500]}")

        if out.get("status") == "oom":
            raise CudaOOMError(f"FLUX.2 OOM: {out.get('error')}")
        if out.get("status") != "ok":
            raise BackendError(f"FLUX.2 error: {out.get('error')}")

        return GenerationResult(
            job_id=request.job_id,
            success=True,
            output_paths=out.get("output_paths", []),
            seed_used=out.get("seed_used"),
            backend="flux2",
            model_id=self.model_id,
            model_revision=self.revision,
            vram_peak_mib=out.get("vram_peak_mib"),
            elapsed_seconds=out.get("elapsed_seconds"),
        )
