"""
Qwen inference worker — runs in a subprocess.

stdin: path to a JSON payload file
stdout: JSON {"status": "ok", "text": "..."}
stderr: JSON error details on failure

This module is intentionally isolated from the parent package's schemas to keep
the subprocess environment minimal.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _load_model_and_processor(
    model_id: str,
    cache_dir: str,
    quantization: str,
    compute_dtype: str,
    revision: str | None,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    dtype = torch.float16 if compute_dtype == "float16" else torch.float32

    hf_kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "revision": revision,
        "attn_implementation": "sdpa",
    }

    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        hf_kwargs["quantization_config"] = bnb_config
        hf_kwargs["device_map"] = "auto"
        hf_kwargs["max_memory"] = {0: "7000MiB", "cpu": "24GiB"}
    else:
        hf_kwargs["device_map"] = "auto"
        hf_kwargs["max_memory"] = {0: "7000MiB", "cpu": "24GiB"}

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **hf_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_id, cache_dir=cache_dir, revision=revision
    )
    return model, processor


def _prepare_messages_planner(
    markdown: str,
    image_paths: list[str],
    max_image_side: int,
    max_images: int,
    system_prompt: str,
    extra_context: str,
    asset_id_map: dict[str, str],
) -> list[dict]:
    from PIL import Image

    content: list[dict] = []

    # Add images (capped)
    for i, img_path in enumerate(image_paths[:max_images]):
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if max(w, h) > max_image_side:
                scale = max_image_side / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            aid = asset_id_map.get(img_path, f"ref_{i}")
            content.append({"type": "text", "text": f"[Reference image: {aid}]"})
            content.append({"type": "image", "image": img})
        except Exception:
            pass

    # Add Markdown text
    text_block = f"User request (Markdown):\n\n{markdown}"
    if extra_context:
        text_block += f"\n\nAdditional context:\n{extra_context}"
    content.append({"type": "text", "text": text_block})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _prepare_messages_critic(
    markdown: str,
    reference_image_paths: list[str],
    candidate_image_path: str,
    scene_plan_json: str,
    max_image_side: int,
    max_images: int,
    system_prompt: str,
) -> list[dict]:
    from PIL import Image

    content: list[dict] = []

    for i, img_path in enumerate(reference_image_paths[:max_images - 1]):
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if max(w, h) > max_image_side:
                scale = max_image_side / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            content.append({"type": "text", "text": f"[Reference image {i + 1}]"})
            content.append({"type": "image", "image": img})
        except Exception:
            pass

    try:
        cand = Image.open(candidate_image_path).convert("RGB")
        w, h = cand.size
        if max(w, h) > max_image_side:
            scale = max_image_side / max(w, h)
            cand = cand.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        content.append({"type": "text", "text": "[Candidate image to evaluate]"})
        content.append({"type": "image", "image": cand})
    except Exception:
        pass

    text_block = (
        f"Original request:\n{markdown}\n\n"
        f"Scene plan summary:\n{scene_plan_json[:1000]}\n\n"
        "Evaluate the candidate image above."
    )
    content.append({"type": "text", "text": text_block})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def run_inference(payload: dict) -> str:
    import torch

    model_id = payload["model_id"]
    revision = payload.get("revision")
    cache_dir = payload["cache_dir"]
    quantization = payload.get("quantization", "4bit")
    compute_dtype = payload.get("compute_dtype", "float16")
    max_new_tokens = payload.get("max_new_tokens", 2048)
    max_image_side = payload.get("max_image_side", 448)
    max_images = payload.get("max_images", 4)
    mode = payload.get("mode", "planner")
    system_prompt = payload.get("system_prompt", "")

    model, processor = _load_model_and_processor(
        model_id, cache_dir, quantization, compute_dtype, revision
    )

    if mode == "planner":
        image_paths = payload.get("image_paths", [])
        asset_id_map: dict[str, str] = payload.get("asset_id_map", {})
        messages = _prepare_messages_planner(
            markdown=payload["markdown"],
            image_paths=image_paths,
            max_image_side=max_image_side,
            max_images=max_images,
            system_prompt=system_prompt,
            extra_context=payload.get("extra_context", ""),
            asset_id_map=asset_id_map,
        )
    else:
        messages = _prepare_messages_critic(
            markdown=payload["markdown"],
            reference_image_paths=payload.get("reference_image_paths", []),
            candidate_image_path=payload.get("candidate_image_path", ""),
            scene_plan_json=payload.get("scene_plan_json", ""),
            max_image_side=max_image_side,
            max_images=max_images,
            system_prompt=system_prompt,
        )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = outputs[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def main() -> None:
    if len(sys.argv) < 2:
        result = {"status": "error", "error": "No payload path given"}
        print(json.dumps(result), file=sys.stderr)
        sys.exit(1)

    payload_path = sys.argv[1]
    try:
        with open(payload_path, encoding="utf-8") as f:
            payload = json.load(f)
        text = run_inference(payload)
        print(json.dumps({"status": "ok", "text": text}))
    except Exception as exc:
        import traceback

        print(
            json.dumps(
                {
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
