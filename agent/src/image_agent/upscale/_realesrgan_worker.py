"""
Real-ESRGAN upscale worker — runs in a subprocess.

Receives a JSON payload file. Outputs JSON result to stdout.
Supports:
  - RealESRGAN_x4plus_anime_6B (default for anime)
  - RealESRGAN_x4plus (general)
  - fp16 on CUDA
  - tile retry on OOM (256 → 128)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def run_upscale(payload: dict) -> dict:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from PIL import Image
    import numpy as np
    import torch

    input_path = payload["input_path"]
    output_path = payload["output_path"]
    model_name = payload.get("model_name", "RealESRGAN_x4plus_anime_6B")
    tile = int(payload.get("tile", 256))
    tile_pad = int(payload.get("tile_padding", 16))
    fp16 = bool(payload.get("fp16", True))
    target_width = payload.get("target_width")
    target_height = payload.get("target_height")
    model_cache_dir = payload.get("model_cache_dir", "/workspace/agent/.runtime/models/realesrgan")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model_cache_dir).mkdir(parents=True, exist_ok=True)

    # Model config
    if model_name == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
        model_url = (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
            "RealESRGAN_x4plus_anime_6B.pth"
        )
    else:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )
        netscale = 4
        model_url = (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
            "RealESRGAN_x4plus.pth"
        )

    t0 = time.monotonic()
    half = fp16 and torch.cuda.is_available()

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_url,
        dni_weight=None,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=0,
        half=half,
        gpu_id=0 if torch.cuda.is_available() else None,
    )

    input_img = Image.open(input_path).convert("RGB")
    input_w, input_h = input_img.size

    img_array = np.array(input_img)[:, :, ::-1]  # RGB -> BGR for basicsr

    oom_fallback = False
    try:
        output_array, _ = upsampler.enhance(img_array, outscale=netscale)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and tile > 128:
            # One retry with smaller tile
            oom_fallback = True
            upsampler.tile_size = 128
            output_array, _ = upsampler.enhance(img_array, outscale=netscale)
        else:
            raise

    output_img = Image.fromarray(output_array[:, :, ::-1])  # BGR -> RGB
    out_w, out_h = output_img.size

    # Resize/crop to target if specified
    final_path = output_path
    if target_width and target_height:
        output_img = _fit_to_target(output_img, int(target_width), int(target_height))
        final_path = output_path.replace(".png", "_fullhd.png")
        output_img.save(final_path, format="PNG")
        final_w, final_h = output_img.size
    else:
        final_w, final_h = out_w, out_h

    # Save upscaled (before target resize)
    upscaled_path = output_path
    Image.fromarray(output_array[:, :, ::-1]).save(upscaled_path, format="PNG")

    elapsed = time.monotonic() - t0

    vram_peak = None
    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "status": "ok",
        "upscaled_path": upscaled_path,
        "final_path": final_path if (target_width and target_height) else upscaled_path,
        "input_width": input_w,
        "input_height": input_h,
        "output_width": final_w,
        "output_height": final_h,
        "oom_fallback_used": oom_fallback,
        "tile_used": 128 if oom_fallback else tile,
        "elapsed_seconds": elapsed,
        "vram_peak_mib": vram_peak,
        "model_name": model_name,
    }


def _fit_to_target(img: Any, target_w: int, target_h: int) -> Any:
    """Scale and centre-crop to exact target dimensions."""
    from PIL import Image

    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "error": "No payload path"}))
        sys.exit(1)

    payload_path = sys.argv[1]
    try:
        with open(payload_path, encoding="utf-8") as f:
            payload = json.load(f)
        result = run_upscale(payload)
        print(json.dumps(result))
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(json.dumps({"status": "oom", "error": str(exc)}))
            sys.exit(2)
        print(json.dumps({"status": "error", "error": str(exc)}))
        sys.exit(1)
    except Exception as exc:
        import traceback
        print(json.dumps({"status": "error", "error": str(exc), "traceback": traceback.format_exc()}))
        sys.exit(1)


if __name__ == "__main__":
    main()
