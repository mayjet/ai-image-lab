"""
FLUX.2 (black-forest-labs/FLUX.2-klein-4B) inference worker.

Runs as a subprocess. Receives a JSON payload file path as argv[1].
Writes output images to the specified directory and prints JSON result.

RTX 2080 config (VRAM 8GB):
  - 4bit quantization (bitsandbytes NF4)
  - FP16 compute
  - sequential CPU offload
  - VAE tiling + slicing
  - batch size 1
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional


def _load_pipeline(
    model_id: str,
    cache_dir: str,
    quantization: str,
    compute_dtype: str,
    offload: str,
    revision: str | None,
) -> Any:
    import torch
    from diffusers import AutoModel, BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from diffusers import Flux2KleinPipeline
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from transformers import Qwen3ForCausalLM

    dtype = torch.float16 if compute_dtype == "float16" else torch.float32
    component_kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "revision": revision,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    transformer = None
    text_encoder = None
    # RTX 2080 SUPER exposes about 7786 MiB. Keep headroom for CUDA context,
    # activations, VAE decode, and fragmentation; use host RAM for the rest.
    max_memory = {0: "6200MiB", "cpu": "24GiB"}

    if quantization in ("bnb4", "4bit"):
        transformer_quant = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        text_quant = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        transformer = AutoModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=transformer_quant,
            device_map="auto",
            max_memory=max_memory,
            **component_kwargs,
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            model_id,
            subfolder="text_encoder",
            quantization_config=text_quant,
            device_map="auto",
            max_memory=max_memory,
            **component_kwargs,
        )

    pipeline_kwargs: dict[str, Any] = {}
    if quantization in ("bnb4", "4bit"):
        pipeline_kwargs.update(device_map="balanced", max_memory=max_memory)

    pipe = Flux2KleinPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=text_encoder,
        cache_dir=cache_dir,
        revision=revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
        **pipeline_kwargs,
    )

    # Quantized components retain Accelerate's dispatch hooks. Applying a second
    # offload mechanism would corrupt that device map.
    if quantization not in ("bnb4", "4bit") and offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif quantization not in ("bnb4", "4bit") and offload == "model":
        pipe.enable_model_cpu_offload()

    # VAE tiling and slicing to reduce peak VRAM
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    return pipe


def run_generation(payload: dict) -> dict:
    import torch

    model_id = payload["model_id"]
    revision = payload.get("revision")
    cache_dir = payload.get("cache_dir", "/workspace/agent/.runtime/models")
    quantization = payload.get("quantization", "bnb4")
    compute_dtype = payload.get("compute_dtype", "float16")
    offload = payload.get("offload", "sequential")
    prompt = payload["prompt"]
    width = int(payload.get("width", 512))
    height = int(payload.get("height", 512))
    steps = int(payload.get("steps", 4))
    guidance_scale = float(payload.get("guidance_scale", 1.0))
    seed = payload.get("seed")
    max_sequence_length = int(payload.get("max_sequence_length", 256))
    output_dir = payload["output_dir"]
    job_id = payload["job_id"]
    reference_paths = payload.get("reference_image_paths", [])[:4]
    negative_constraints = payload.get("negative_constraints", [])

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    pipe = _load_pipeline(model_id, cache_dir, quantization, compute_dtype, offload, revision)

    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is not None:
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        actual_seed = int(seed)
    else:
        actual_seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=generator_device).manual_seed(actual_seed)

    vram_before = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / (1024 * 1024)

    if negative_constraints:
        prompt = f"{prompt}\nAvoid: {', '.join(str(item) for item in negative_constraints)}"

    call_kwargs = dict(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        max_sequence_length=max_sequence_length,
    )
    if reference_paths:
        from PIL import Image

        references = [Image.open(path).convert("RGB") for path in reference_paths]
        call_kwargs["image"] = references

    output = pipe(**call_kwargs)

    vram_peak = None
    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

    img = output.images[0]
    out_path = str(Path(output_dir) / f"{job_id}.png")
    img.save(out_path, format="PNG")

    elapsed = time.monotonic() - t0
    return {
        "status": "ok",
        "output_paths": [out_path],
        "seed_used": actual_seed,
        "vram_peak_mib": vram_peak,
        "elapsed_seconds": elapsed,
        "model_id": model_id,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "error": "No payload path"}))
        sys.exit(1)

    payload_path = sys.argv[1]
    try:
        with open(payload_path, encoding="utf-8") as f:
            payload = json.load(f)
        result = run_generation(payload)
        print(json.dumps(result))
    except MemoryError as exc:
        print(json.dumps({"status": "oom", "error": str(exc)}))
        sys.exit(2)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(json.dumps({"status": "oom", "error": str(exc)}))
            sys.exit(2)
        raise
    except Exception as exc:
        import traceback
        print(json.dumps({"status": "error", "error": str(exc), "traceback": traceback.format_exc()}))
        sys.exit(1)


if __name__ == "__main__":
    main()
