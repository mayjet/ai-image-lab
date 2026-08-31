"""Generate or edit an image with FLUX.2 [klein] and an optional LoRA."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.paths import flux2_output_root
from flux2.prompt_markdown import (
    LoraSpec,
    ReferenceSpec,
    compose_prompt,
    parse_markdown,
    resolve_loras,
    resolve_project_path,
    resolve_references,
    setting_float,
    setting_int,
)

INFERENCE_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
MEMORY_MODES = (
    "auto",
    "gpu",
    "model-offload",
    "sequential-offload",
    "group-offload",
    "disk-offload",
    "offload",  # backward-compatible alias for model-offload
)
QUANTIZATION_MODES = ("auto", "none", "bnb4")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--prompt")
    prompt_source.add_argument("--prompt-md", type=Path, help="prompts/prompt1.md形式のMarkdown")
    parser.add_argument("--model-id", default=INFERENCE_MODEL_ID)
    parser.add_argument("--lora", help="ローカルsafetensors、フォルダ、またはHub ID")
    parser.add_argument("--no-lora", action="store_true", help="Markdown内のLoRA指定を無視する")
    parser.add_argument("--lora-weight-name")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--input-image", type=Path, help="指定時は画像編集")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-reference-images", type=int)
    parser.add_argument("--reference-max-side", type=int, help="参照画像を読み込み時に縮小する長辺")
    parser.add_argument("--memory-mode", choices=MEMORY_MODES)
    parser.add_argument("--quantization", choices=QUANTIZATION_MODES)
    parser.add_argument(
        "--offload-dir",
        type=Path,
        help="disk-offload用ディレクトリ（高速なSSDを推奨）",
    )
    parser.add_argument("--no-vae-tiling", action="store_true")
    parser.add_argument("--no-vae-slicing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="メモリ安全判定を無視")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.lora_scale < 0:
        parser.error("--lora-scaleは0以上にしてください")
    return args


def available_ram_gib() -> float:
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return page_size * available_pages / 1024**3


def select_memory_mode(requested: str, vram_gib: float, ram_gib: float) -> str:
    if requested == "offload":
        return "model-offload"
    if requested != "auto":
        return requested
    if vram_gib >= 15:
        return "gpu"
    if vram_gib >= 4 and ram_gib >= 20:
        return "sequential-offload"
    raise RuntimeError(
        f"利用可能VRAM {vram_gib:.1f} GiB / RAM {ram_gib:.1f} GiBでは安全に読み込めません。"
        " 8GB級GPUではoffload用の空きCPU RAM 20 GiB以上が目安です"
    )


def select_quantization(requested: str, vram_gib: float) -> str:
    if requested != "auto":
        return requested
    return "bnb4" if vram_gib < 15 else "none"


def inference_dtype(torch, compute_capability_major: int):
    # Ampere以降はbf16、Turing以前はfp16を使う。
    return torch.bfloat16 if compute_capability_major >= 8 else torch.float16


def output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (flux2_output_root() / "generation" / f"flux2-{stamp}-seed{args.seed}.png").resolve()


def load_pipeline(
    args: argparse.Namespace,
    mode: str,
    quantization: str,
    dtype,
    loras: list[LoraSpec],
):
    import torch
    from diffusers import Flux2KleinPipeline

    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": args.local_files_only,
    }
    if quantization == "bnb4":
        try:
            from diffusers.quantizers import PipelineQuantizationConfig
        except ImportError as error:
            raise RuntimeError(
                "4bit量子化にはbitsandbytes対応のdiffusers/transformersが必要です"
            ) from error
        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": dtype,
            },
            components_to_quantize=["transformer", "text_encoder"],
        )
    # 低VRAM時は量子化時の一時VRAM peakも避けるため、最初からCPUへ読み込む。
    kwargs["device_map"] = "cuda" if mode == "gpu" else "cpu"
    pipe = Flux2KleinPipeline.from_pretrained(args.model_id, **kwargs)
    if mode != "gpu":
        # CPU device mapを外してからstatefulなoffload hookを設定する。
        pipe.reset_device_map()
    adapter_names = []
    adapter_weights = []
    for index, lora_spec in enumerate(loras):
        lora = Path(lora_spec.source).expanduser()
        source = str(lora.resolve()) if lora.exists() else lora_spec.source
        load_kwargs = {"low_cpu_mem_usage": True}
        if lora.is_file():
            source = str(lora.resolve().parent)
            load_kwargs["weight_name"] = lora.name
        elif args.lora_weight_name and len(loras) == 1:
            load_kwargs["weight_name"] = args.lora_weight_name
        adapter_name = f"md_{index}_{lora_spec.role}"
        pipe.load_lora_weights(source, adapter_name=adapter_name, **load_kwargs)
        adapter_names.append(adapter_name)
        adapter_weights.append(lora_spec.weight)
    if adapter_names:
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    # LoRAのロード後にstatefulなoffload hookを入れる。
    if not args.no_vae_tiling:
        pipe.vae.enable_tiling()
    if not args.no_vae_slicing:
        pipe.vae.enable_slicing()
    if mode == "model-offload":
        pipe.enable_model_cpu_offload()
    elif mode == "sequential-offload":
        pipe.enable_sequential_cpu_offload()
    elif mode in {"group-offload", "disk-offload"}:
        group_kwargs = {
            "onload_device": torch.device("cuda"),
            "offload_device": torch.device("cpu"),
            "offload_type": "block_level",
            "num_blocks_per_group": 1,
            "use_stream": False,
            "low_cpu_mem_usage": True,
        }
        if mode == "disk-offload":
            args.offload_dir.mkdir(parents=True, exist_ok=True)
            group_kwargs["offload_to_disk_path"] = str(args.offload_dir)
        pipe.enable_group_offload(**group_kwargs)
    return pipe


def resolve_configuration(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, list[ReferenceSpec], list[LoraSpec]]:
    document = parse_markdown(args.prompt_md) if args.prompt_md else None
    max_references = args.max_reference_images
    if max_references is None:
        max_references = setting_int(document, "max_reference_images", 1) if document else 1
    if max_references < 0 or max_references > 4:
        raise RuntimeError("参照画像数は0〜4枚にしてください")
    args.max_reference_images = max_references
    references = resolve_references(document, PROJECT_ROOT, max_references) if document else []
    if args.input_image:
        if max_references == 0:
            raise RuntimeError("--input-image使用時は参照画像数を1以上にしてください")
        input_path = args.input_image.expanduser().resolve()
        if not input_path.is_file():
            raise RuntimeError(f"入力画像がありません: {input_path}")
        existing = next((item for item in references if item.path == input_path), None)
        if existing is None:
            references.insert(0, ReferenceSpec(path=input_path, roles=("source_image",)))
            references = references[:max_references]
    loras = resolve_loras(document, PROJECT_ROOT) if document and not args.no_lora else []
    if args.lora:
        lora_path = Path(args.lora).expanduser()
        source = str(lora_path.resolve()) if lora_path.exists() else args.lora
        loras.append(LoraSpec(role="cli", source=source, weight=args.lora_scale))
    args.prompt = compose_prompt(document, references) if document else args.prompt
    args.width = args.width if args.width is not None else (setting_int(document, "width", 512) if document else 512)
    args.height = args.height if args.height is not None else (setting_int(document, "height", 512) if document else 512)
    args.steps = args.steps if args.steps is not None else (setting_int(document, "steps", 4) if document else 4)
    args.seed = args.seed if args.seed is not None else (setting_int(document, "seed", 42) if document else 42)
    if args.guidance_scale is None:
        args.guidance_scale = setting_float(document, "guidance_scale", 1.0) if document else 1.0
    if args.max_sequence_length is None:
        args.max_sequence_length = (
            setting_int(document, "max_sequence_length", 256) if document else 256
        )
    args.memory_mode = args.memory_mode or (document.settings.get("memory_mode", "auto") if document else "auto")
    args.quantization = args.quantization or (document.settings.get("quantization", "auto") if document else "auto")
    if args.offload_dir is None:
        setting = document.settings.get("offload_dir") if document else None
        args.offload_dir = (
            resolve_project_path(setting, PROJECT_ROOT)
            if setting
            else PROJECT_ROOT / "cache" / "flux2-offload"
        )
    if args.reference_max_side is None:
        args.reference_max_side = setting_int(document, "reference_max_side", 512) if document else 512
    if args.memory_mode not in MEMORY_MODES:
        raise RuntimeError(f"settings.memory_mode が不正です: {args.memory_mode}")
    if args.quantization not in QUANTIZATION_MODES:
        raise RuntimeError(f"settings.quantization が不正です: {args.quantization}")
    if args.output is None and document and document.settings.get("output"):
        args.output = resolve_project_path(document.settings["output"], PROJECT_ROOT)
    if args.width < 256 or args.height < 256 or args.width % 16 or args.height % 16:
        raise RuntimeError("width/heightは256以上かつ16の倍数にしてください")
    if args.width * args.height // 256 > 4096:
        raise RuntimeError("(width * height) / 256 は4096以下にしてください")
    if args.steps < 1:
        raise RuntimeError("stepsは1以上にしてください")
    if args.max_sequence_length < 64 or args.max_sequence_length > 512:
        raise RuntimeError("max_sequence_lengthは64〜512にしてください")
    if args.reference_max_side < 256:
        raise RuntimeError("reference_max_sideは256以上にしてください")
    return args, references, loras


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args, references, loras = resolve_configuration(args)
    target = output_path(args)
    config = vars(args).copy()
    config["output"] = str(target)
    config["input_image"] = str(args.input_image) if args.input_image else None
    config["prompt_md"] = str(args.prompt_md) if args.prompt_md else None
    config["references"] = [
        {"path": str(reference.path), "roles": list(reference.roles)} for reference in references
    ]
    config["loras"] = [vars(lora) for lora in loras]
    print(json.dumps(config, ensure_ascii=False, indent=2, default=str))
    if args.dry_run:
        return 0
    # CUDA allocator初期化前に設定する。予約領域の断片化によるOOMを減らす。
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    props = torch.cuda.get_device_properties(0)
    vram_gib = props.total_memory / 1024**3
    free_vram_gib = torch.cuda.mem_get_info(0)[0] / 1024**3
    ram_gib = available_ram_gib()
    if (os.environ.get("AI_IMAGE_PLATFORM") == "l4t" or os.uname().machine == "aarch64") and ram_gib < 16 and not args.force:
        raise RuntimeError(f"Jetsonの利用可能共有RAM {ram_gib:.1f} GiBではモデル取得前に停止します")
    try:
        mode = select_memory_mode(args.memory_mode, free_vram_gib, ram_gib)
    except RuntimeError:
        if not args.force:
            raise
        mode = "sequential-offload" if args.memory_mode == "auto" else args.memory_mode
    quantization = select_quantization(args.quantization, free_vram_gib)
    dtype = inference_dtype(torch, props.major)
    if free_vram_gib < 10 and quantization == "none" and not args.force:
        raise RuntimeError(
            f"空きVRAM {free_vram_gib:.1f} GiBでは非量子化モデルは危険です。"
            " --quantization bnb4を使うか、意図的に試す場合だけ--forceを指定してください"
        )
    print(
        f"memory_mode={mode}, quantization={quantization}, dtype={dtype}, "
        f"GPU={props.name}, VRAM={vram_gib:.1f} GiB, free_VRAM={free_vram_gib:.1f} GiB, "
        f"available_RAM={ram_gib:.1f} GiB"
    )
    torch.cuda.reset_peak_memory_stats()
    pipe = load_pipeline(args, mode, quantization, dtype, loras)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    call = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "generator": generator,
        "max_sequence_length": args.max_sequence_length,
    }
    if references:
        reference_images = []
        for reference in references:
            with Image.open(reference.path) as source_image:
                loaded = source_image.convert("RGB")
                loaded.thumbnail(
                    (args.reference_max_side, args.reference_max_side),
                    Image.Resampling.LANCZOS,
                )
                reference_images.append(loaded)
        call["image"] = reference_images[0] if len(reference_images) == 1 else reference_images
    with torch.inference_mode():
        image = pipe(**call).images[0]
    runtime_memory = {
        "peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "peak_process_ram_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, pnginfo=None)
    target.with_suffix(".json").write_text(
        json.dumps(
            {
                **config,
                "memory_mode_selected": mode,
                "quantization_selected": quantization,
                "dtype_selected": str(dtype),
                "hardware": {
                    "gpu": props.name,
                    "vram_total_gib": round(vram_gib, 3),
                    "vram_free_before_load_gib": round(free_vram_gib, 3),
                    "ram_available_before_load_gib": round(ram_gib, 3),
                },
                "runtime_memory": runtime_memory,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"memory_peak={json.dumps(runtime_memory, ensure_ascii=False)}")
    print(f"saved: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
