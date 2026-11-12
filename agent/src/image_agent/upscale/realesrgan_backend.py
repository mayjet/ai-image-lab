"""
Real-ESRGAN upscaler backend — subprocess-isolated.

Runs inference in a child process so VRAM is fully released on exit.
No LoRA. No GFPGAN face enhancement by default (would change character identity).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from image_agent.errors import UpscaleError, UpscalerOOMError
from image_agent.runtime.logging import get_logger
from image_agent.runtime.subprocess_runner import run_subprocess
from image_agent.runtime.config import load_config
from image_agent.schemas.evaluation import UpscaleResult

logger = get_logger(__name__)

# Target resolutions
_TARGETS: dict[tuple[int, int], tuple[int, int]] = {
    (768, 432): (1920, 1080),
    (432, 768): (1080, 1920),
    (512, 512): (2048, 2048),
}
DEFAULT_TIMEOUT = 600


class RealESRGANUpscaler:
    """
    Real-ESRGAN upscaler using subprocess isolation.

    Defaults to anime model; never loads GFPGAN face enhancement.
    """

    def __init__(
        self,
        anime_model: Optional[str] = None,
        general_model: Optional[str] = None,
        tile: Optional[int] = None,
        tile_padding: Optional[int] = None,
        fp16: Optional[bool] = None,
        model_cache_dir: str = "/workspace/agent/.runtime/models/realesrgan",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        cfg = load_config("models.yaml").get("upscaler", {})
        self.anime_model = anime_model or cfg.get("anime_model", "RealESRGAN_x4plus_anime_6B")
        self.general_model = general_model or cfg.get("general_model", "RealESRGAN_x4plus")
        self.tile = tile if tile is not None else int(cfg.get("tile", 256))
        self.tile_padding = (
            tile_padding if tile_padding is not None else int(cfg.get("tile_padding", 16))
        )
        self.fp16 = fp16 if fp16 is not None else bool(cfg.get("fp16", True))
        self.model_cache_dir = model_cache_dir
        self.timeout = timeout

    def upscale(
        self,
        source_asset_id: str,
        input_path: str,
        output_dir: str,
        use_anime_model: bool = True,
    ) -> UpscaleResult:
        """
        Upscale an image using Real-ESRGAN in a subprocess.

        Returns UpscaleResult; raises UpscaleError / UpscalerOOMError on failure.
        """
        input_p = Path(input_path)
        try:
            from PIL import Image
            with Image.open(input_path) as img:
                in_w, in_h = img.size
        except Exception as exc:
            raise UpscaleError(f"Cannot read input image {input_path}: {exc}") from exc

        target = _TARGETS.get((in_w, in_h))
        out_path = str(Path(output_dir) / f"{source_asset_id}_upscaled.png")

        payload = {
            "input_path": input_path,
            "output_path": out_path,
            "model_name": self.anime_model if use_anime_model else self.general_model,
            "tile": self.tile,
            "tile_padding": self.tile_padding,
            "fp16": self.fp16,
            "model_cache_dir": self.model_cache_dir,
        }
        if target:
            payload["target_width"] = target[0]
            payload["target_height"] = target[1]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(payload, f)
            payload_path = f.name

        try:
            worker = str(Path(__file__).parent / "_realesrgan_worker.py")
            result = run_subprocess(
                [sys.executable, worker, payload_path],
                timeout=self.timeout,
                log_tag="realesrgan_worker",
                allow_failure=True,
            )
        finally:
            try:
                os.unlink(payload_path)
            except OSError:
                pass

        if result.returncode == 2:
            raise UpscalerOOMError("Real-ESRGAN OOM even after tile reduction")
        if result.returncode != 0:
            raise UpscaleError(
                f"Real-ESRGAN worker failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )

        try:
            out = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise UpscaleError(f"Real-ESRGAN output not JSON: {result.stdout[:500]}")

        if out.get("status") == "oom":
            raise UpscalerOOMError(out.get("error", "OOM"))
        if out.get("status") != "ok":
            raise UpscaleError(f"Real-ESRGAN error: {out.get('error')}")

        output_ids = []
        for path_key in ("upscaled_path", "final_path"):
            p = out.get(path_key, "")
            if p and p not in output_ids:
                output_ids.append(p)

        return UpscaleResult(
            source_asset_id=source_asset_id,
            output_asset_ids=output_ids,
            backend="realesrgan",
            model_name=out.get("model_name", ""),
            input_width=out.get("input_width", in_w),
            input_height=out.get("input_height", in_h),
            output_width=out.get("output_width", 0),
            output_height=out.get("output_height", 0),
            tile_size=out.get("tile_used", self.tile),
            tile_padding=self.tile_padding,
            fp16=self.fp16,
            oom_fallback_used=out.get("oom_fallback_used", False),
            success=True,
        )
