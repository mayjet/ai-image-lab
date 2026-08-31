"""SD1.5向け、フォルダ単位のキャラクターLoRA学習。

character.tomlを正本として共通captionを生成し、準備済みfaceを含めて学習する。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import toml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.paths import default_dataset_root, sd15_model_root, sd15_output_root, sd_scripts_root

DATASET_ROOT = default_dataset_root()
MODEL_ROOT = sd15_model_root()
MODEL_ID = "stablediffusionapi/counterfeit-v30"
IS_SDXL = False
TRAIN_FOLDERS = ["lora", "portrait", "anime", "game", "illust", "face"]

TRAIN_RESOLUTION = 512
BATCH_SIZE = 1
LORA_RANK = 32
LORA_ALPHA = 32
LEARNING_RATE = 1e-4
TRAIN_SEED = 42
STEPS_PER_IMAGE = 30
MIN_TRAIN_STEPS = 300
MAX_TRAIN_STEPS = 1200
SAVE_EVERY_N_STEPS = 300
USE_8BIT_ADAM = False
LOG_WITH = "tensorboard"

SD_SCRIPTS_DIR = sd_scripts_root()
OUTPUT_DIR = sd15_output_root() / "training"
RUNTIME_DATASET_DIR = OUTPUT_DIR / "runtime_datasets"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
COMMON_CAPTION_FILE = "_common_caption.txt"
CHARACTER_CONFIG_FILE = "character.toml"
DEFAULT_IDENTITY_TAGS = ["High school girl"]
FOLDER_CAPTION_SUFFIXES = {
    "lora": "",
    "portrait": "full body, character reference",
    "anime": "anime screenshot, anime style",
    "game": "game artwork",
    "illust": "illustration",
    "face": "face focus, close-up",
}

def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def character_token(character_dir: Path) -> str:
    # Dataset directory names remain URL/shell friendly, captions use a natural full name.
    return " ".join(part.capitalize() for part in character_dir.name.replace("_", "-").split("-") if part)


def join_tags(*texts: str) -> str:
    result, seen = [], set()
    for text in texts:
        for raw in text.replace("\n", ",").split(","):
            tag = " ".join(raw.strip().split())
            key = tag.casefold()
            if tag and key not in seen:
                result.append(tag)
                seen.add(key)
    return ", ".join(result)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def default_character_config(character_dir: Path) -> dict:
    return {
        "schema_version": 1,
        "name": character_token(character_dir),
        "folder": character_dir.name,
        "unit": "",
        "identity_tags": list(DEFAULT_IDENTITY_TAGS),
        "folders": {
            name: [tag.strip() for tag in suffix.split(",") if tag.strip()]
            for name, suffix in FOLDER_CAPTION_SUFFIXES.items()
        },
    }


def load_character_config(character_dir: Path) -> dict:
    path = character_dir / CHARACTER_CONFIG_FILE
    if not path.exists():
        config = default_character_config(character_dir)
        with path.open("w", encoding="utf-8") as stream:
            toml.dump(config, stream)
        print(f"[caption] created config: {path}")
        return config
    try:
        config = toml.load(path)
    except Exception as exc:
        raise RuntimeError(f"character.tomlを読めません: {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"character.tomlの形式が不正です: {path}")
    return config


def config_tags(value: object, location: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise RuntimeError(f"captionタグは文字列の配列で指定してください: {location}")


def render_common_caption(character_dir: Path, folder_name: str, config: dict) -> str:
    name = str(config.get("name") or character_token(character_dir)).strip()
    if not name:
        raise RuntimeError(f"キャラクター名が空です: {character_dir / CHARACTER_CONFIG_FILE}")
    identity_tags = config_tags(
        config.get("identity_tags", DEFAULT_IDENTITY_TAGS),
        f"{character_dir.name}.identity_tags",
    )
    folders = config.get("folders", {})
    if folders is not None and not isinstance(folders, dict):
        raise RuntimeError(f"[folders]の形式が不正です: {character_dir / CHARACTER_CONFIG_FILE}")
    folder_tags = config_tags(
        (folders or {}).get(
            folder_name,
            [tag.strip() for tag in FOLDER_CAPTION_SUFFIXES[folder_name].split(",") if tag.strip()],
        ),
        f"{character_dir.name}.folders.{folder_name}",
    )
    return join_tags(name, ", ".join(identity_tags), ", ".join(folder_tags))


def write_common_caption(character_dir: Path, folder_name: str, config: dict) -> Path:
    folder = character_dir / folder_name
    path = folder / COMMON_CAPTION_FILE
    value = render_common_caption(character_dir, folder_name, config)
    old_value = read_text(path)
    if old_value != value:
        path.write_text(value + "\n", encoding="utf-8")
        action = "updated" if old_value else "created"
        print(f"[caption] {action}: {path} -> {value}")
    return path


def find_character_dirs(selected: set[str] | None = None) -> dict[str, Path]:
    found = {}
    if not DATASET_ROOT.exists():
        return found
    for path in sorted(DATASET_ROOT.iterdir()):
        if path.is_dir() and (not selected or path.name in selected):
            if any((path / name).is_dir() for name in TRAIN_FOLDERS if name != "face"):
                found[path.name] = path
    return found


def prepare_common_captions(character_dirs: dict[str, Path]) -> None:
    for character_dir in character_dirs.values():
        config = load_character_config(character_dir)
        for folder_name in TRAIN_FOLDERS:
            # face/ is owned by anime_face_collect.py and is not created by training setup.
            if (character_dir / folder_name).is_dir():
                write_common_caption(character_dir, folder_name, config)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_folder(character_dir: Path, folder_name: str) -> list[str]:
    folder, errors = character_dir / folder_name, []
    images = list_images(folder)
    if not images:
        return ["画像がありません"]
    common, token = read_text(folder / COMMON_CAPTION_FILE), character_token(character_dir)
    if not common:
        errors.append(f"{COMMON_CAPTION_FILE} がないか空です")
    elif token.casefold() not in common.casefold():
        errors.append(f"共通captionにキャラ名 `{token}` がありません")
    if folder_name == "lora":
        stems = {}
        for image in images:
            stems.setdefault(image.stem.casefold(), []).append(image.name)
        for names in stems.values():
            if len(names) > 1:
                errors.append("同じstemの画像があります: " + ", ".join(names))
        for image in images:
            if not read_text(image.with_suffix(".txt")):
                errors.append(f"個別captionがありません: {image.stem}.txt")
    return errors


def validate_dataset(
    character_dirs: dict[str, Path],
    selected: set[str] | None,
) -> bool:
    ok = True
    for character, root in character_dirs.items():
        for folder_name in TRAIN_FOLDERS:
            if selected and folder_name not in selected:
                continue
            if not list_images(root / folder_name):
                print(f"[validate] {character}/{folder_name}: ERROR (画像なし)")
                ok = False
                continue
            errors = validate_folder(root, folder_name)
            print(f"[validate] {character}/{folder_name}: {'ERROR' if errors else 'OK'} (images={len(list_images(root / folder_name))})")
            for error in errors:
                print("  -", error)
            ok &= not errors
    return ok


def build_runtime_lora_dataset(character_dir: Path) -> Path:
    source, runtime = character_dir / "lora", RUNTIME_DATASET_DIR / character_dir.name / "lora"
    shutil.rmtree(runtime, ignore_errors=True)
    runtime.mkdir(parents=True, exist_ok=True)
    common = read_text(source / COMMON_CAPTION_FILE)
    for image in list_images(source):
        target = runtime / image.name
        try:
            target.symlink_to(image.resolve())
        except OSError:
            shutil.copy2(image, target)
        (runtime / f"{image.stem}.txt").write_text(
            join_tags(common, read_text(image.with_suffix(".txt"))) + "\n", encoding="utf-8"
        )
    return runtime


def build_runtime_common_dataset(character_dir: Path, folder_name: str) -> Path:
    """Expose images without any stray per-image captions; class_tokens supplies the caption."""
    source = character_dir / folder_name
    runtime = RUNTIME_DATASET_DIR / character_dir.name / folder_name
    shutil.rmtree(runtime, ignore_errors=True)
    runtime.mkdir(parents=True, exist_ok=True)
    for image in list_images(source):
        target = runtime / image.name
        try:
            target.symlink_to(image.resolve())
        except OSError:
            shutil.copy2(image, target)
    return runtime


def build_dataset_config(character_dir: Path, folder_name: str) -> tuple[dict, Path]:
    source = character_dir / folder_name
    image_dir = (
        build_runtime_lora_dataset(character_dir)
        if folder_name == "lora"
        else build_runtime_common_dataset(character_dir, folder_name)
    )
    subset = {"image_dir": str(image_dir.resolve()), "num_repeats": 1}
    if folder_name != "lora":
        subset["class_tokens"] = read_text(source / COMMON_CAPTION_FILE)
    return ({
        "general": {"shuffle_caption": True, "caption_extension": ".txt", "keep_tokens": 1},
        "datasets": [{
            "resolution": TRAIN_RESOLUTION, "batch_size": BATCH_SIZE,
            "enable_bucket": True, "min_bucket_reso": 256, "max_bucket_reso": 512,
            "bucket_reso_steps": 64, "subsets": [subset],
        }],
    }, image_dir)


def optimizer_type() -> str:
    import torch

    if torch.cuda.is_available() and USE_8BIT_ADAM:
        try:
            import bitsandbytes  # noqa: F401
            return "AdamW8bit"
        except Exception as exc:
            print(f"bitsandbytes unavailable; AdamWへfallback: {exc}")
    return "AdamW"


def train_steps(image_count: int) -> int:
    return max(MIN_TRAIN_STEPS, min(MAX_TRAIN_STEPS, image_count * STEPS_PER_IMAGE))


def release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def training_fingerprint(character_dir: Path, folder_name: str, steps: int) -> str:
    folder = character_dir / folder_name
    payload = {
        "model": MODEL_ID,
        "sdxl": IS_SDXL,
        "folder": folder_name,
        "common_caption": read_text(folder / COMMON_CAPTION_FILE),
        "images": [],
        "training": {
            "resolution": TRAIN_RESOLUTION,
            "batch_size": BATCH_SIZE,
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "learning_rate": LEARNING_RATE,
            "seed": TRAIN_SEED,
            "steps": steps,
            "save_every": SAVE_EVERY_N_STEPS,
        },
    }
    for image in list_images(folder):
        item = {"name": image.name, "sha256": sha256_file(image)}
        if folder_name == "lora":
            item["caption"] = read_text(image.with_suffix(".txt"))
        payload["images"].append(item)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stream_command(command: list[str], env: dict[str, str]) -> None:
    print("実行コマンド:", " ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)


def attempt_train(command: list[str], output: Path, name: str, env: dict[str, str]) -> bool:
    try:
        stream_command(command, env)
        return output.is_file() and output.stat().st_size > 0
    except subprocess.CalledProcessError as exc:
        print(f"[{name}] 学習失敗 (exit={exc.returncode})")
        return False


def train_one(character_dir: Path, folder_name: str, force: bool = False, dry_run: bool = False) -> bool:
    images = list_images(character_dir / folder_name)
    if not images:
        return True
    output_dir = MODEL_ROOT / DATASET_ROOT.name / character_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"{character_dir.name}-{folder_name}"
    output = output_dir / f"{folder_name}.safetensors"
    metadata_path = output_dir / f"{folder_name}.train.json"
    steps = train_steps(len(images))
    fingerprint = training_fingerprint(character_dir, folder_name, steps)
    old_metadata = {}
    if metadata_path.exists():
        try:
            old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    if output.exists() and old_metadata.get("fingerprint") == fingerprint and not force:
        print(f"[{name}] 同一データ・設定の完成済みモデル -> skip")
        return True
    config, runtime_dir = build_dataset_config(character_dir, folder_name)
    config_dir = OUTPUT_DIR / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{name}.toml"
    with config_path.open("w", encoding="utf-8") as stream:
        toml.dump(config, stream)
    script = "sdxl_train_network.py" if IS_SDXL else "train_network.py"
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch", "--num_processes", "1",
        "--num_machines", "1", "--mixed_precision", "fp16", "--dynamo_backend", "no",
        "--num_cpu_threads_per_process", "1", str(SD_SCRIPTS_DIR / script),
        "--pretrained_model_name_or_path", MODEL_ID, "--dataset_config", str(config_path),
        "--output_dir", str(output_dir), "--output_name", folder_name, "--save_model_as", "safetensors",
        "--network_module", "networks.lora", "--network_dim", str(LORA_RANK),
        "--network_alpha", str(LORA_ALPHA), "--learning_rate", str(LEARNING_RATE),
        "--lr_scheduler", "cosine_with_restarts", "--max_train_steps", str(steps),
        "--mixed_precision", "fp16", "--save_precision", "fp16",
        "--optimizer_type", optimizer_type(), "--clip_skip", "1" if IS_SDXL else "2",
        "--seed", str(TRAIN_SEED), "--cache_latents", "--cache_latents_to_disk", "--sdpa",
        "--gradient_checkpointing", "--lowram", "--max_data_loader_n_workers", "1",
        "--save_every_n_steps", str(SAVE_EVERY_N_STEPS), "--log_with", LOG_WITH,
        "--logging_dir", str(OUTPUT_DIR / "logs"), "--log_tracker_name", name,
    ]
    print(f"\n[{name}] images={len(images)}, steps={steps}, checkpoint_every={SAVE_EVERY_N_STEPS}, dataset={runtime_dir}")
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        shutil.rmtree(RUNTIME_DATASET_DIR / character_dir.name / folder_name, ignore_errors=True)
        return True
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    ok = attempt_train(cmd, output, name, env)
    if not ok:
        no_cache = [x for x in cmd if x not in {"--cache_latents", "--cache_latents_to_disk"}]
        print(f"[{name}] cache_latentsなしで再試行")
        ok = attempt_train(no_cache, output, name, env)
    if ok:
        checkpoint_dir = output_dir / "checkpoints"
        for checkpoint in output_dir.glob(f"{folder_name}-step*.safetensors"):
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(checkpoint), checkpoint_dir / checkpoint.name)
        metadata = {
            "name": name,
            "fingerprint": fingerprint,
            "model": MODEL_ID,
            "folder": folder_name,
            "images": len(images),
            "steps": steps,
            "save_every_n_steps": SAVE_EVERY_N_STEPS,
            "dataset_config": str(config_path.resolve()),
            "output": str(output.resolve()),
            "command": cmd,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    release_cuda()
    shutil.rmtree(RUNTIME_DATASET_DIR / character_dir.name / folder_name, ignore_errors=True)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--characters", nargs="*")
    parser.add_argument("--folders", nargs="*", choices=TRAIN_FOLDERS)
    parser.add_argument("--prepare", action="store_true", help="共通caption準備のみ")
    parser.add_argument("--validate", action="store_true", help="データを検証して終了")
    parser.add_argument("--train", action="store_true", help="準備済みデータだけを検証して学習")
    parser.add_argument("--force", action="store_true", help="同じデータ・設定でも再学習")
    parser.add_argument("--dry-run", action="store_true", help="設定とコマンドを作るが学習しない")
    return parser.parse_args()


def main() -> None:
    global DATASET_ROOT, MODEL_ROOT, OUTPUT_DIR, RUNTIME_DATASET_DIR
    args = parse_args()
    DATASET_ROOT = args.dataset_root.expanduser().resolve()
    MODEL_ROOT = args.model_root.expanduser().resolve()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    RUNTIME_DATASET_DIR = OUTPUT_DIR / "runtime_datasets"
    if not (args.prepare or args.validate or args.train):
        raise RuntimeError("実行内容を指定してください: --prepare / --validate / --train")
    characters = find_character_dirs(set(args.characters) if args.characters else None)
    folders = set(args.folders) if args.folders else None
    if not characters:
        raise RuntimeError(f"キャラフォルダが見つかりません: {DATASET_ROOT}")
    print("対象キャラ:", list(characters))
    print("対象フォルダ:", sorted(folders or set(TRAIN_FOLDERS)))
    if args.prepare or args.train:
        prepare_common_captions(characters)
    if args.prepare and not (args.validate or args.train):
        return
    if args.train:
        missing_faces = [
            name for name, root in characters.items()
            if not (root / "face").is_dir() or not list_images(root / "face")
        ]
        if missing_faces:
            raise RuntimeError(
                "faceデータがないため学習を開始しません。先に anime_face_collect.py を実行してください: "
                + ", ".join(missing_faces)
            )
    if (args.validate or args.train) and not validate_dataset(characters, folders):
        raise RuntimeError("データ検証に失敗しました")
    if args.validate and not args.train:
        return
    if args.train:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDAが利用できません。Jetsonコンテナ内で実行してください。")
        (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
        failures = []
        for character, root in characters.items():
            for folder_name in TRAIN_FOLDERS:
                if (not folders or folder_name in folders) and not train_one(
                    root, folder_name, args.force, args.dry_run
                ):
                    failures.append(f"{character}/{folder_name}")
        if failures:
            raise RuntimeError("学習失敗: " + ", ".join(failures))


if __name__ == "__main__":
    main()
