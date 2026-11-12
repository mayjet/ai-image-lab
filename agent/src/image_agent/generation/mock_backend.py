"""
Mock generation backend for testing without a GPU.

Creates a solid-colour PNG so the pipeline can run end-to-end.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from image_agent.generation.base import GenerationBackend, GenerationRequest, GenerationResult


class MockBackend(GenerationBackend):
    """Test backend — no GPU required."""

    @property
    def name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        try:
            import PIL  # noqa: F401

            return True
        except ImportError:
            return False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        from PIL import Image, ImageDraw, ImageFont

        t0 = time.monotonic()
        out_dir = Path(request.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        seed = request.seed or 42
        # Simple deterministic colour based on seed
        r = (seed * 37) % 256
        g = (seed * 71) % 256
        b = (seed * 113) % 256

        img = Image.new("RGB", (request.width, request.height), color=(r, g, b))
        draw = ImageDraw.Draw(img)
        draw.text(
            (10, 10),
            f"[MOCK]\n{request.prompt[:60]}",
            fill=(255, 255, 255),
        )

        out_path = out_dir / f"{request.job_id}_mock.png"
        img.save(str(out_path), format="PNG")

        return GenerationResult(
            job_id=request.job_id,
            success=True,
            output_paths=[str(out_path)],
            seed_used=seed,
            backend="mock",
            model_id="mock",
            elapsed_seconds=time.monotonic() - t0,
        )
