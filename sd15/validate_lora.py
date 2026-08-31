"""学習済みキャラクターLoRAを同一条件で生成比較する。

引数なしでは datasets/works の全キャラクター・全項目の最終重みを検証する。
Jetsonでは docker/run_l4t.sh sd15 で起動したコンテナ内から実行する。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.paths import default_dataset_root, sd15_model_root, sd15_output_root

try:
    import tomllib
except ImportError:  # Python 3.10向けfallback（Jetson学習環境は3.12）
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_DATASET_ROOT = default_dataset_root()
DEFAULT_MODEL_ROOT = sd15_model_root()
DEFAULT_MODEL_ID = "stablediffusionapi/counterfeit-v30"
FOLDERS = ["lora", "portrait", "anime", "game", "illust", "face"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, normal quality, blurry, bad anatomy, bad hands, "
    "extra fingers, missing fingers, extra limbs, multiple girls, text, watermark, logo"
)
STEP_PATTERN = re.compile(r"-step(\d+)$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--characters", nargs="*", help="省略時は全キャラクター")
    parser.add_argument("--folders", nargs="*", choices=FOLDERS, help="省略時は全項目")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.6, 0.8, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--checkpoints", action="store_true", help="途中重みも生成比較する")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--cpu-offload", action="store_true", help="低メモリ向け。生成は遅くなる")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=sd15_output_root() / "validation",
    )
    parser.add_argument("--run-name", help="省略時は実行日時")
    parser.add_argument("--dry-run", action="store_true", help="対象と生成件数だけ確認する")
    args = parser.parse_args(argv)
    if args.width < 64 or args.height < 64 or args.width % 8 or args.height % 8:
        parser.error("width/heightは64以上かつ8の倍数にしてください")
    if args.steps < 1:
        parser.error("stepsは1以上にしてください")
    if not args.scales or any(scale < 0 for scale in args.scales):
        parser.error("scalesは0以上を指定してください")
    return args


def list_images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def find_characters(dataset_root: Path, selected: list[str] | None) -> dict[str, Path]:
    if not dataset_root.is_dir():
        raise RuntimeError(f"データセットがありません: {dataset_root}")
    requested = set(selected or [])
    found = {
        path.name: path
        for path in sorted(dataset_root.iterdir())
        if path.is_dir()
        and (not requested or path.name in requested)
        and any((path / folder).is_dir() for folder in FOLDERS)
    }
    missing = requested - set(found)
    if missing:
        raise RuntimeError("存在しないキャラクター: " + ", ".join(sorted(missing)))
    if not found:
        raise RuntimeError(f"対象キャラクターがありません: {dataset_root}")
    return found


def load_character_config(character_dir: Path) -> dict[str, Any]:
    path = character_dir / "character.toml"
    if path.is_file():
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    else:
        config = {}
    if not isinstance(config, dict):
        config = {}
    return config


def character_name(character_dir: Path, config: dict[str, Any]) -> str:
    fallback = " ".join(part.capitalize() for part in character_dir.name.replace("_", "-").split("-") if part)
    return str(config.get("name") or fallback).strip()


def as_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def prompt_for(character_dir: Path, folder: str) -> str:
    config = load_character_config(character_dir)
    folders = config.get("folders", {})
    folder_tags = as_tags(folders.get(folder)) if isinstance(folders, dict) else []
    tags = [
        character_name(character_dir, config),
        *as_tags(config.get("identity_tags", ["High school girl"])),
        *folder_tags,
        "solo", "looking at viewer", "simple background", "masterpiece", "best quality",
    ]
    result, seen = [], set()
    for raw in tags:
        tag = " ".join(str(raw).strip().split())
        if tag and tag.casefold() not in seen:
            result.append(tag)
            seen.add(tag.casefold())
    return ", ".join(result)


def weight_paths(
    character_dir: Path,
    folder: str,
    checkpoints: bool,
    model_root: Path,
    dataset_name: str,
) -> list[Path]:
    weight_dir = model_root / dataset_name / character_dir.name
    final = weight_dir / f"{folder}.safetensors"
    paths: list[Path] = []
    if checkpoints:
        paths.extend(sorted((weight_dir / "checkpoints").glob(f"{folder}-step*.safetensors"), key=weight_step))
    if final.is_file():
        paths.append(final)
    if paths:
        return paths
    # Transitional fallback for datasets that have not moved folder_loras yet.
    legacy_dir = character_dir / "folder_loras"
    legacy_final = legacy_dir / f"{character_dir.name}-{folder}.safetensors"
    if checkpoints:
        paths.extend(sorted(legacy_dir.glob(f"{character_dir.name}-{folder}-step*.safetensors"), key=weight_step))
    if legacy_final.is_file():
        paths.append(legacy_final)
    return paths


def weight_step(path: Path) -> int:
    match = STEP_PATTERN.search(path.stem)
    return int(match.group(1)) if match else sys.maxsize


def weight_label(path: Path) -> str:
    step = weight_step(path)
    return f"step{step:08d}" if step != sys.maxsize else "final"


def build_jobs(args: argparse.Namespace, characters: dict[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    folders = args.folders or FOLDERS
    jobs, missing = [], []
    for character, root in characters.items():
        for folder in folders:
            weights = weight_paths(
                root, folder, args.checkpoints, args.model_root, args.dataset_root.name
            )
            if not weights:
                missing.append({"character": character, "folder": folder, "status": "missing_weight"})
                continue
            for weight in weights:
                for seed in args.seeds:
                    for scale in args.scales:
                        jobs.append({
                            "character": character, "root": root, "folder": folder,
                            "weight": weight, "weight_label": weight_label(weight),
                            "seed": seed, "scale": scale,
                        })
    return jobs, missing


def select_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAを利用できません")
    return requested


def load_pipeline(args: argparse.Namespace, device: str):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        local_files_only=args.local_files_only,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    if args.cpu_offload:
        if device != "cuda":
            raise RuntimeError("--cpu-offloadはCUDA利用時だけ指定できます")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    return pipe


def release_memory(device: str) -> None:
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()


def make_grid(items: list[tuple[Path, str]], output: Path, columns: int = 3) -> None:
    if not items:
        return
    thumb_w, thumb_h, label_h = 320, 320, 42
    columns = max(1, min(columns, len(items)))
    rows = math.ceil(len(items) / columns)
    canvas = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(items):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_w + (thumb_w - image.width) // 2
        y = (index // columns) * (thumb_h + label_h) + (thumb_h - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text(((index % columns) * thumb_w + 6, y + thumb_h + 4), label[:52], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def write_reports(run_dir: Path, config: dict[str, Any], results: list[dict[str, Any]]) -> None:
    (run_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "status", "character", "folder", "weight_label", "weight", "scale", "seed",
        "seconds", "cuda_peak_mb", "image", "prompt", "error",
    ]
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def run(args: argparse.Namespace) -> Path | None:
    characters = find_characters(args.dataset_root, args.characters)
    jobs, missing = build_jobs(args, characters)
    baseline_count = len(characters) * len(args.seeds)
    print("対象キャラ:", list(characters))
    print("対象項目:", args.folders or FOLDERS)
    print(f"生成予定: baseline={baseline_count}, LoRA={len(jobs)}, missing={len(missing)}")
    for item in missing:
        print(f"[missing] {item['character']}/{item['folder']}")
    if args.dry_run:
        for job in jobs:
            print(
                f"[job] {job['character']}/{job['folder']} {job['weight_label']} "
                f"scale={job['scale']} seed={job['seed']}"
            )
        return None

    import torch
    device = select_device(args.device)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"出力先が空ではありません: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({
        "dataset_root": str(args.dataset_root.resolve()), "output_dir": str(args.output_dir),
        "characters_resolved": list(characters), "folders_resolved": args.folders or FOLDERS,
        "device_resolved": device,
    })
    results: list[dict[str, Any]] = list(missing)
    write_reports(run_dir, config, results)
    try:
        pipe = load_pipeline(args, device)
    except Exception as exc:
        error = {
            "status": "pipeline_error", "character": "", "folder": "", "weight_label": "",
            "weight": "", "scale": "", "seed": "", "prompt": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
        results.append(error)
        write_reports(run_dir, config, results)
        print(f"[pipeline_error] {error['error']}")
        print(f"RESULT_DIR={run_dir.resolve()}")
        raise

    # Baselineはキャラクター名を含む同一条件で、LoRA適用との差を見るために生成する。
    for character, root in characters.items():
        prompt = prompt_for(root, "lora")
        for seed in args.seeds:
            record: dict[str, Any] = {
                "status": "pending", "character": character, "folder": "baseline",
                "weight_label": "none", "weight": "", "scale": 0.0, "seed": seed,
                "prompt": prompt, "error": "",
            }
            try:
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                generator = torch.Generator(device=device).manual_seed(seed)
                image = pipe(
                    prompt=prompt, negative_prompt=args.negative_prompt,
                    width=args.width, height=args.height, num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale, clip_skip=2, generator=generator,
                ).images[0]
                output = run_dir / character / "baseline" / f"baseline_seed{seed}.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                image.save(output)
                record.update({
                    "status": "ok", "seconds": round(time.perf_counter() - started, 3),
                    "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else None,
                    "image": str(output.resolve()),
                })
                print(f"[ok] {character}/baseline seed={seed} -> {output}")
            except Exception as exc:
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                print(f"[error] {character}/baseline: {record['error']}")
            results.append(record)
            write_reports(run_dir, config, results)
            release_memory(device)

    for job in jobs:
        prompt = prompt_for(job["root"], job["folder"])
        record = {
            "status": "pending", "character": job["character"], "folder": job["folder"],
            "weight_label": job["weight_label"], "weight": str(job["weight"].resolve()),
            "scale": job["scale"], "seed": job["seed"], "prompt": prompt, "error": "",
        }
        adapter = "validation_adapter"
        try:
            pipe.load_lora_weights(str(job["weight"]), adapter_name=adapter)
            pipe.set_adapters([adapter], adapter_weights=[job["scale"]])
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            generator = torch.Generator(device=device).manual_seed(job["seed"])
            image = pipe(
                prompt=prompt, negative_prompt=args.negative_prompt,
                width=args.width, height=args.height, num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale, clip_skip=2, generator=generator,
            ).images[0]
            filename = (
                f"{job['character']}_{job['folder']}_{job['weight_label']}_"
                f"scale{job['scale']:g}_seed{job['seed']}.png"
            )
            output = run_dir / job["character"] / job["folder"] / filename
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output)
            record.update({
                "status": "ok", "seconds": round(time.perf_counter() - started, 3),
                "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else None,
                "image": str(output.resolve()),
            })
            print(
                f"[ok] {job['character']}/{job['folder']} {job['weight_label']} "
                f"scale={job['scale']} seed={job['seed']} -> {output}"
            )
        except Exception as exc:
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] {job['character']}/{job['folder']}: {record['error']}")
        finally:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
            release_memory(device)
        results.append(record)
        write_reports(run_dir, config, results)

    for character in characters:
        character_results = [r for r in results if r.get("character") == character and r.get("status") == "ok"]
        for folder in ["baseline", *(args.folders or FOLDERS)]:
            folder_results = [r for r in character_results if r.get("folder") == folder]
            items = [(Path(r["image"]), f"{r['weight_label']} s={r['scale']:g} seed={r['seed']}") for r in folder_results]
            make_grid(items, run_dir / character / "grids" / f"{folder}.png", columns=len(args.scales) if folder != "baseline" else len(args.seeds))
        summary = []
        for folder in ["baseline", *(args.folders or FOLDERS)]:
            candidates = [r for r in character_results if r.get("folder") == folder]
            if candidates:
                preferred = min(candidates, key=lambda r: (r.get("weight_label") != "final", abs(float(r.get("scale", 0)) - 0.8), r.get("seed", 0)))
                summary.append((Path(preferred["image"]), folder))
        make_grid(summary, run_dir / character / "grids" / "summary.png", columns=len(summary))

    summaries = []
    for character in characters:
        path = run_dir / character / "grids" / "summary.png"
        if path.is_file():
            summaries.append((path, character))
    make_grid(summaries, run_dir / "grids" / "all_characters_summary.png", columns=1)
    write_reports(run_dir, config, results)
    print(f"RESULT_DIR={run_dir.resolve()}")
    return run_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
