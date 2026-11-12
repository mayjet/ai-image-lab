"""
FLUX.3 Klein ComfyUI INT8 backend.

STATUS: EXPERIMENTAL
  - RTX 2080 (8 GB VRAM) compatibility is unverified.
  - Only INT8 ConvRot checkpoint files are supported.
  - Falls back to FLUX.2 on failure when fallback_backend is set.

License note:
  FLUX.3 weights are distributed under the FLUX Kommunity License,
  which differs from FLUX.2's Apache 2.0 license.
  Review the license at https://bfl.ai before use.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from image_agent.errors import BackendError, ComfyUIError, CudaOOMError
from image_agent.generation.base import GenerationBackend, GenerationRequest, GenerationResult
from image_agent.runtime.logging import get_logger
from image_agent.runtime.subprocess_runner import run_subprocess

logger = get_logger(__name__)


class Flux3Backend(GenerationBackend):
    """FLUX.3 Klein via ComfyUI INT8 ConvRot, subprocess-isolated.

    EXPERIMENTAL — not verified on RTX 2080.
    """

    def __init__(
        self,
        comfyui_dir: str = "/workspace/agent/.runtime/comfyui",
        workflow_path: str = "/workspace/agent/workflows/flux3/flux3_int8_basic.json",
        model_cache_dir: str = "/workspace/agent/.runtime/models",
        timeout: int = 1800,
        port: int = 18188,
    ) -> None:
        self.comfyui_dir = comfyui_dir
        self.workflow_path = workflow_path
        self.model_cache_dir = model_cache_dir
        self.timeout = timeout
        self.port = port

    @property
    def name(self) -> str:
        return "flux3"

    def is_available(self) -> bool:
        """Check if ComfyUI directory exists."""
        return Path(self.comfyui_dir).exists()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        # Patch the workflow with request parameters
        workflow = self._prepare_workflow(request)

        with tempfile.TemporaryDirectory() as tmpdir:
            workflow_tmp = str(Path(tmpdir) / "workflow.json")
            with open(workflow_tmp, "w", encoding="utf-8") as f:
                json.dump(workflow, f)

            payload = {
                "workflow_path": workflow_tmp,
                "output_dir": request.output_dir,
                "comfyui_dir": self.comfyui_dir,
                "port": self.port,
                "job_id": request.job_id,
            }
            payload_path = str(Path(tmpdir) / "payload.json")
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            worker = str(Path(__file__).parent / "_flux3_worker.py")
            env = dict(os.environ)
            env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

            result = run_subprocess(
                [sys.executable, worker, payload_path],
                timeout=self.timeout,
                env=env,
                log_tag="flux3_worker",
                allow_failure=True,
            )

        if result.returncode == 2:
            raise CudaOOMError("FLUX.3 worker reported CUDA OOM")
        if result.returncode == 3:
            raise ComfyUIError("ComfyUI job timed out")
        if result.returncode != 0:
            raise BackendError(
                f"FLUX.3 worker exited {result.returncode}.\n{result.stderr[-2000:]}"
            )

        try:
            out = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise BackendError(f"FLUX.3 output not JSON: {result.stdout[:500]}")

        if out.get("status") != "ok":
            raise BackendError(f"FLUX.3 error: {out.get('error')}")

        return GenerationResult(
            job_id=request.job_id,
            success=True,
            output_paths=out.get("output_paths", []),
            backend="flux3",
            model_id="FLUX.3-klein-int8-convrot",
            elapsed_seconds=out.get("elapsed_seconds"),
        )

    def _prepare_workflow(self, request: GenerationRequest) -> dict:
        """Load the base workflow and patch parameters for this request."""
        try:
            with open(self.workflow_path, encoding="utf-8") as f:
                workflow = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError(f"Cannot load FLUX.3 workflow {self.workflow_path}: {exc}") from exc

        # Patch prompt, seed, dimensions in ComfyUI API format
        # Node IDs depend on the specific workflow; use known keys if present
        for node_id, node in workflow.items():
            inputs = node.get("inputs", {})
            class_type = node.get("class_type", "")

            if class_type == "CLIPTextEncode" and "text" in inputs:
                inputs["text"] = request.prompt

            if class_type in ("KSampler", "KSamplerAdvanced"):
                if request.seed is not None:
                    inputs["seed"] = request.seed
                inputs["steps"] = request.steps

            if class_type == "EmptyLatentImage":
                inputs["width"] = request.width
                inputs["height"] = request.height

        return workflow
