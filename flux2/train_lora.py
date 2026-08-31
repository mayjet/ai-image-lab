"""FLUX.2 [klein] 4B DreamBooth LoRA launcher with low-memory defaults.

The actual training loop is Hugging Face Diffusers' maintained example.  This
wrapper validates the machine and dataset, builds a reproducible command, and
keeps memory-heavy validation disabled during training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.paths import default_dataset_root, diffusers_root, flux2_model_root, flux2_output_root

TRAIN_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TRAINER_RELATIVE = Path("examples/dreambooth/train_dreambooth_lora_flux2_klein.py")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-dir", type=Path, help="学習画像を直下に置いたフォルダ")
    parser.add_argument("--character", help="datasets/works/<character>/lora を使う省略指定")
    parser.add_argument("--instance-prompt", required=True, help="例: ZK_CHAR, one anime girl")
    parser.add_argument("--run-name", default="flux2-klein-lora")
    parser.add_argument("--model-id", default=TRAIN_MODEL_ID)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trainer", type=Path, default=diffusers_root() / TRAINER_RELATIVE)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--checkpointing-steps", type=int, default=250)
    parser.add_argument("--checkpoints-total-limit", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--fp8", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--no-8bit-adam", action="store_true")
    parser.add_argument("--offload", action="store_true", help="VRAMを減らす代わりにCPU RAMを多く使う")
    parser.add_argument("--no-cache-latents", action="store_true")
    parser.add_argument("--resume", default=None, metavar="CHECKPOINT", help="例: latest")
    parser.add_argument("--train", action="store_true", help="実際に学習を開始する")
    parser.add_argument("--dry-run", action="store_true", help="検証してコマンドだけ表示")
    parser.add_argument("--force-unsupported", action="store_true", help="Jetson/小容量GPUの停止判定を無視")
    args = parser.parse_args(argv)
    if bool(args.instance_dir) == bool(args.character):
        parser.error("--instance-dir と --character のどちらか一方を指定してください")
    if args.resolution < 256 or args.resolution % 16:
        parser.error("--resolution は256以上かつ16の倍数にしてください")
    for name in ("steps", "checkpointing_steps", "checkpoints_total_limit", "rank", "lora_alpha", "gradient_accumulation"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} は1以上にしてください")
    if not args.train and not args.dry_run:
        parser.error("安全のため --dry-run または --train を指定してください")
    return args


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_instance_dir(args: argparse.Namespace) -> Path:
    path = args.instance_dir or (default_dataset_root() / args.character / "lora")
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"学習画像フォルダがありません: {path}")
    images = image_files(path)
    if not images:
        raise RuntimeError(f"学習画像がありません: {path}")
    return path


def cuda_info(required: bool) -> tuple[bool, tuple[int, int] | None, float | None]:
    try:
        import torch
    except ImportError:
        if required:
            raise RuntimeError("PyTorchがありません。flux2-cudaコンテナ内で実行してください")
        return False, None, None
    available = torch.cuda.is_available()
    if required and not available:
        raise RuntimeError("CUDAが利用できません。flux2-cudaコンテナ内で実行してください")
    if not available:
        return False, None, None
    props = torch.cuda.get_device_properties(0)
    return True, torch.cuda.get_device_capability(0), props.total_memory / 1024**3


def fp8_enabled(mode: str, cuda_available: bool, capability: tuple[int, int] | None) -> bool:
    supported = cuda_available and capability is not None and capability >= (8, 9)
    if mode == "on" and not supported:
        raise RuntimeError("FP8学習にはcompute capability 8.9以上が必要です")
    return supported if mode == "auto" else mode == "on"


def build_command(
    args: argparse.Namespace,
    instance_dir: Path,
    source_dir: Path | None = None,
) -> tuple[list[str], dict[str, object]]:
    execute = args.train and not args.dry_run
    if execute:
        cuda_available, capability, vram_gib = cuda_info(required=True)
    else:
        cuda_available, capability, vram_gib = False, None, None
    if execute and (os.environ.get("AI_IMAGE_PLATFORM") == "l4t" or os.uname().machine == "aarch64") and not args.force_unsupported:
        raise RuntimeError("JetsonでFLUX.2 LoRA学習は実行しません。x86のflux2-cuda環境を使ってください")
    if execute and vram_gib is not None and vram_gib < 22 and not args.force_unsupported:
        raise RuntimeError(f"VRAM {vram_gib:.1f} GiBでは既定学習を開始しません（推奨24 GiB以上）")
    trainer = args.trainer.expanduser().resolve()
    if not trainer.is_file() and execute:
        raise RuntimeError(f"公式trainerがありません: {trainer}\n先に flux2-cuda コンテナを起動してください")
    output_dir = (args.output_dir or (flux2_model_root() / args.run_name)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    use_fp8 = fp8_enabled(args.fp8, cuda_available, capability) if execute else args.fp8 == "on"
    command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", "1", "--num_machines", "1", "--dynamo_backend", "no",
        str(trainer),
        "--pretrained_model_name_or_path", args.model_id,
        "--instance_data_dir", str(instance_dir),
        "--instance_prompt", args.instance_prompt,
        "--output_dir", str(output_dir),
        "--resolution", str(args.resolution),
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", str(args.gradient_accumulation),
        "--max_train_steps", str(args.steps),
        "--checkpointing_steps", str(args.checkpointing_steps),
        "--checkpoints_total_limit", str(args.checkpoints_total_limit),
        "--learning_rate", str(args.learning_rate),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--rank", str(args.rank),
        "--lora_alpha", str(args.lora_alpha),
        "--mixed_precision", args.mixed_precision,
        "--guidance_scale", "1.0",
        "--gradient_checkpointing",
        "--skip_final_inference",
        "--dataloader_num_workers", "0",
        "--report_to", "tensorboard",
        "--logging_dir", str(flux2_output_root() / "training" / "logs" / args.run_name),
        "--seed", str(args.seed),
    ]
    if not args.no_cache_latents:
        command.append("--cache_latents")
    if args.offload:
        command.append("--offload")
    if not args.no_8bit_adam:
        command.extend(["--optimizer", "AdamW", "--use_8bit_adam"])
    if use_fp8:
        command.append("--do_fp8_training")
    if args.resume:
        command.extend(["--resume_from_checkpoint", args.resume])
    metadata = {
        "model": args.model_id,
        "instance_dir": str(source_dir or instance_dir),
        "runtime_instance_dir": str(instance_dir),
        "images": len(image_files(instance_dir)),
        "instance_prompt": args.instance_prompt,
        "output_dir": str(output_dir),
        "vram_gib": vram_gib,
        "compute_capability": capability,
        "fp8": use_fp8,
        "command": command,
    }
    return command, metadata


def image_only_dataset(source: Path, run_name: str):
    """Create a temporary image-only view; the official trainer opens every entry."""
    runtime_root = flux2_output_root() / "runtime_datasets"
    runtime_root.mkdir(parents=True, exist_ok=True)
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-.") or "flux2"
    temporary = tempfile.TemporaryDirectory(prefix=f"{safe_prefix}-", dir=runtime_root)
    runtime = Path(temporary.name)
    for image in image_files(source):
        target = runtime / image.name
        try:
            target.symlink_to(image.resolve())
        except OSError:
            shutil.copy2(image, target)
    return temporary, runtime


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = resolve_instance_dir(args)
    temporary, instance_dir = image_only_dataset(source_dir, args.run_name)
    try:
        command, metadata = build_command(args, instance_dir, source_dir)
        print(json.dumps({key: value for key, value in metadata.items() if key != "command"}, ensure_ascii=False, indent=2))
        print("\ncommand:\n" + shlex.join(command))
        if args.dry_run or not args.train:
            return 0
        output_dir = Path(str(metadata["output_dir"]))
        (output_dir / "launch.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        subprocess.run(command, env=env, check=True)
    finally:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
