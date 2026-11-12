"""
Qwen3-VL-4B-Instruct planner/critic runner (subprocess-based).

The Qwen model runs in a child process to ensure full VRAM release on exit.
The parent communicates via stdin/stdout using JSON.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

from image_agent.errors import (
    JSONExtractionError,
    JSONRepairError,
    PlannerError,
    SubprocessError,
)
from image_agent.planner.json_extractor import extract_json, repair_json_once
from image_agent.runtime.logging import get_logger
from image_agent.runtime.subprocess_runner import run_subprocess
from image_agent.runtime.config import load_config

logger = get_logger(__name__)

# Default Qwen model (can be overridden via config)
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_QWEN_TIMEOUT = 1800  # 30 minutes, including first model download


class QwenPlannerRunner:
    """
    Wrapper that runs Qwen inference in a subprocess.

    Usage
    -----
        runner = QwenPlannerRunner(model_id="Qwen/Qwen3-VL-4B-Instruct")
        raw_text = runner.run_planner(markdown=..., image_paths=[...])
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        timeout: Optional[int] = None,
        quantization: Optional[str] = None,
        compute_dtype: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        max_image_side: Optional[int] = None,
        max_images: Optional[int] = None,
    ) -> None:
        model_cfg = load_config("models.yaml").get("planner_critic", {})
        self.model_id = model_id or model_cfg.get("model_id", DEFAULT_QWEN_MODEL)
        self.revision = model_cfg.get("revision") or None
        self.cache_dir = cache_dir or "/workspace/agent/.runtime/models"
        self.timeout = timeout or int(model_cfg.get("timeout_seconds", DEFAULT_QWEN_TIMEOUT))
        self.quantization = quantization or model_cfg.get("quantization", "4bit")
        self.compute_dtype = compute_dtype or model_cfg.get("compute_dtype", "float16")
        self.max_new_tokens = max_new_tokens or int(model_cfg.get("max_new_tokens", 2048))
        self.max_image_side = max_image_side or int(model_cfg.get("max_image_side", 448))
        self.max_images = max_images or int(model_cfg.get("max_images", 4))
        self.gpu_max_memory_mib = int(model_cfg.get("gpu_max_memory_mib", 6200))
        self.cpu_max_memory_gib = int(model_cfg.get("cpu_max_memory_gib", 24))

    def run_planner(
        self,
        markdown: str,
        image_paths: Sequence[str],
        asset_id_map: Optional[dict[str, str]] = None,
        system_prompt: Optional[str] = None,
        extra_context: str = "",
    ) -> str:
        """
        Run Qwen planner on the given Markdown and images.

        Returns raw text output from the model.
        Raises SubprocessError or PlannerError on failure.
        """
        payload = {
            "mode": "planner",
            "model_id": self.model_id,
            "revision": self.revision,
            "cache_dir": self.cache_dir,
            "quantization": self.quantization,
            "compute_dtype": self.compute_dtype,
            "max_new_tokens": self.max_new_tokens,
            "max_image_side": self.max_image_side,
            "max_images": self.max_images,
            "gpu_max_memory_mib": self.gpu_max_memory_mib,
            "cpu_max_memory_gib": self.cpu_max_memory_gib,
            "markdown": markdown,
            "image_paths": list(image_paths),
            "asset_id_map": asset_id_map or {},
            "system_prompt": system_prompt or _PLANNER_SYSTEM_PROMPT,
            "extra_context": extra_context,
        }
        return self._call_worker(payload)

    def run_critic(
        self,
        markdown: str,
        reference_image_paths: Sequence[str],
        candidate_image_path: str,
        scene_plan_json: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Run Qwen critic on a generated candidate.

        Returns raw text output from the model.
        """
        payload = {
            "mode": "critic",
            "model_id": self.model_id,
            "revision": self.revision,
            "cache_dir": self.cache_dir,
            "quantization": self.quantization,
            "compute_dtype": self.compute_dtype,
            "max_new_tokens": 1024,
            "max_image_side": self.max_image_side,
            "max_images": self.max_images,
            "gpu_max_memory_mib": self.gpu_max_memory_mib,
            "cpu_max_memory_gib": self.cpu_max_memory_gib,
            "markdown": markdown,
            "reference_image_paths": list(reference_image_paths),
            "candidate_image_path": candidate_image_path,
            "scene_plan_json": scene_plan_json,
            "system_prompt": system_prompt or _CRITIC_SYSTEM_PROMPT,
        }
        return self._call_worker(payload)

    def _call_worker(self, payload: dict[str, Any]) -> str:
        """Serialize payload to temp file and invoke worker subprocess."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(payload, f)
            payload_path = f.name

        try:
            worker_script = str(Path(__file__).parent / "_qwen_worker.py")
            env = dict(os.environ)
            env["HF_HOME"] = env.get("HF_HOME", "/workspace/agent/.runtime/cache/huggingface")
            env["TRANSFORMERS_CACHE"] = env.get(
                "TRANSFORMERS_CACHE", "/workspace/agent/.runtime/cache/transformers"
            )
            env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            env["TRANSFORMERS_VERBOSITY"] = "error"
            result = run_subprocess(
                [sys.executable, worker_script, payload_path],
                timeout=self.timeout,
                env=env,
                log_tag="qwen_worker",
            )
            # Worker prints JSON result to stdout
            try:
                out = json.loads(result.stdout)
                if out.get("status") == "error":
                    raise PlannerError(f"Qwen worker error: {out.get('error')}")
                return out.get("text", "")
            except json.JSONDecodeError:
                # Fallback: treat raw stdout as text
                return result.stdout
        finally:
            try:
                os.unlink(payload_path)
            except OSError:
                pass


# ---- Prompt templates ----

_PLANNER_SYSTEM_PROMPT = """\
You are a visual scene planner for an anime-style image generation system.
Your job is to analyze the user's Markdown request and reference images, then produce a
JSON plan. You MUST NOT:
- execute code
- access files directly
- use shell commands
- produce fine-tuning or weight training instructions of any kind

Output ONLY a single JSON object with this structure:
{
  "subject": "...",
  "style": "...",
  "scene": "...",
  "composition": "...",
  "must_preserve": [...],
  "may_change": [...],
  "must_design": [...],
  "inferred": [...],
  "contradictions": [...],
  "uncertainty": [...],
  "evaluation_criteria": [...],
  "backend_preference": "flux2",
  "generation_caption": "Full scene caption for FLUX generation",
  "jobs": [
    {
      "id": "job_1",
      "type": "main_render",
      "purpose": "Main character illustration",
      "depends_on": [],
      "input_asset_ids": [],
      "backend": "flux2",
      "prompt": "...",
      "negative_constraints": [],
      "width": 512,
      "height": 512,
      "steps": 4,
      "seed": null
    }
  ]
}

Allowed job types: composition_draft, pose_draft, costume_sheet, item_sheet,
background_draft, main_render, localized_edit, upscale.
Only add auxiliary jobs when strictly needed. Never add more than 3 auxiliary jobs.
Refer to images by their asset_id only — never by file path.
Treat silhouette, body and object proportions, topology, pose geometry, costume panel
layout, emblem shape and placement, and relative spatial relationships as explicit
must_preserve constraints whenever the Markdown or reference images specify them.
Do not replace those constraints with generic style descriptions. Auxiliary jobs must
create references that materially improve the later main render.
"""

_CRITIC_SYSTEM_PROMPT = """\
You are a strict quality evaluator for anime-style generated images.
Compare the candidate image against the original Markdown request and reference images.
Output ONLY a single JSON object:
{
  "action": "accept|retry_prompt|retry_reference|localized_edit|user_required",
  "overall_score": 0.0,
  "confidence": 0.0,
  "face_hair_body": {"score": 0.0, "confidence": 0.0, "notes": ""},
  "costume_emblem_items": {"score": 0.0, "confidence": 0.0, "notes": ""},
  "art_style": {"score": 0.0, "confidence": 0.0, "notes": ""},
  "composition_pose": {"score": 0.0, "confidence": 0.0, "notes": ""},
  "background": {"score": 0.0, "confidence": 0.0, "notes": ""},
  "must_preserve_violations": [],
  "rationale": "Short explanation",
  "suggested_revision": "null or revision hint"
}
Be conservative: if you are uncertain, set confidence <= 0.5 and use user_required.
Evaluate silhouette, proportions, pose geometry, costume construction, emblem geometry,
and relative placement separately from color and art style. Describe any geometric
deviation precisely in suggested_revision so the next deterministic pass can act on it.
"""
