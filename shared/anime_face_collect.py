"""アニメ画像からキャラクターごとのface学習データを事前生成する。

Jetsonのメモリを学習へ持ち越さないため、このスクリプトは単独で実行して終了する。
face/ と lora/ は入力にせず、複数顔の大きさが近い画像は保存しない。
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.paths import default_dataset_root, sd15_output_root

# Keep the command simple while preserving the Python 3.10/CUDA-isolated runtime.
FACE_DETECT_PYTHON = Path("/opt/face-detector/bin/python")
RUNTIME_MARKER = "ANIME_FACE_COLLECT_ISOLATED"
if os.environ.get(RUNTIME_MARKER) != "1":
    if not FACE_DETECT_PYTHON.is_file():
        raise RuntimeError(
            "顔検出用Pythonがありません。Jetson用Dockerイメージを再ビルドし、"
            "コンテナ内で実行してください"
        )
    environment = os.environ.copy()
    environment[RUNTIME_MARKER] = "1"
    os.execve(
        str(FACE_DETECT_PYTHON),
        [str(FACE_DETECT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )

import toml
from PIL import Image, ImageOps

DATASET_ROOT = default_dataset_root()
SOURCE_FOLDERS = ["anime", "game", "illust", "portrait"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
COMMON_CAPTION_FILE = "_common_caption.txt"

OUTPUT_DIR = sd15_output_root() / "face_collect"
FACE_DETECT_SCRIPT = Path("/opt/face-detector/dghs_runner.py")
FACE_MODEL_LEVEL = "n"
FACE_MODEL_VERSION = "v1.4"
FACE_CONFIDENCE = 0.30
FACE_IOU_THRESHOLD = 0.60
FACE_MIN_BOX_SIZE = 64
FACE_CROP_SCALE = 1.60
FACE_OUTPUT_SIZE = 512
FACE_DETECT_MAX_EDGE = 2048
FACE_CHUNK_SIZE = 16
FACE_SECONDARY_AREA_RATIO = 0.70
FACE_DHASH_DISTANCE = 4

# An empty token produces an invalid ``Authorization: Bearer `` header in
# huggingface_hub/httpx.  docker/run_l4t.sh intentionally accepts an unset
# HF_TOKEN, so omit empty authentication variables before model downloads.
HF_TOKEN_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def detector_environment(mode: str | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    for variable in HF_TOKEN_VARIABLES:
        if not environment.get(variable, "").strip():
            environment.pop(variable, None)
    if mode is not None:
        environment["ONNX_MODE"] = mode
    return environment


def list_images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def character_name(character_dir: Path) -> str:
    return " ".join(
        part.capitalize()
        for part in character_dir.name.replace("_", "-").split("-")
        if part
    )


def face_common_caption(character_dir: Path) -> str:
    """Use the same character.toml source of truth as train_lora.py."""
    config_path = character_dir / "character.toml"
    config = toml.load(config_path) if config_path.is_file() else {}
    name = str(config.get("name") or character_name(character_dir)).strip()
    raw_identity = config.get("identity_tags", ["High school girl"])
    identity = [raw_identity] if isinstance(raw_identity, str) else list(raw_identity)
    folders = config.get("folders", {})
    raw_face = folders.get("face", ["face focus", "close-up"]) if isinstance(folders, dict) else []
    face_tags = [raw_face] if isinstance(raw_face, str) else list(raw_face)
    result, seen = [], set()
    for tag in [name, *identity, *face_tags]:
        normalized = " ".join(str(tag).strip().split())
        if normalized and normalized.casefold() not in seen:
            result.append(normalized)
            seen.add(normalized.casefold())
    return ", ".join(result)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dhash(image: Image.Image, size: int = 8) -> int:
    gray = ImageOps.grayscale(image).resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for y in range(size):
        row = y * (size + 1)
        for x in range(size):
            value = (value << 1) | int(pixels[row + x] > pixels[row + x + 1])
    return value


def detector_settings(mode: str) -> dict:
    return {
        "execution_provider": mode,
        "model_level": FACE_MODEL_LEVEL,
        "model_version": FACE_MODEL_VERSION,
        "confidence": FACE_CONFIDENCE,
        "iou": FACE_IOU_THRESHOLD,
        "min_box_size": FACE_MIN_BOX_SIZE,
        "crop_scale": FACE_CROP_SCALE,
        "output_size": FACE_OUTPUT_SIZE,
        "detect_max_edge": FACE_DETECT_MAX_EDGE,
        "secondary_area_ratio": FACE_SECONDARY_AREA_RATIO,
    }


def read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def write_manifest(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def expanded_square_box(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(x2 - x1, y2 - y1) * FACE_CROP_SCALE, width, height)
    left = min(max(0, round(center_x - side / 2)), max(0, round(width - side)))
    top = min(max(0, round(center_y - side / 2)), max(0, round(height - side)))
    return int(left), int(top), int(round(left + side)), int(round(top + side))


def detect_chunk(paths: list[Path], mode: str) -> dict[str, dict]:
    if not FACE_DETECT_PYTHON.is_file() or not FACE_DETECT_SCRIPT.is_file():
        raise RuntimeError("顔検出環境がありません。Jetson用Dockerイメージを再ビルドしてください")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="chunk-", dir=OUTPUT_DIR) as raw_tmp:
        temporary_dir = Path(raw_tmp)
        detector_paths: list[Path] = []
        mapping: dict[str, tuple[Path, float, float]] = {}
        for index, source_path in enumerate(paths):
            try:
                with Image.open(source_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    original_width, original_height = image.size
                    image.thumbnail((FACE_DETECT_MAX_EDGE, FACE_DETECT_MAX_EDGE), Image.Resampling.LANCZOS)
                    detector_path = temporary_dir / f"{index:04d}.jpg"
                    image.save(detector_path, quality=90)
                    detector_paths.append(detector_path)
                    mapping[str(detector_path.resolve())] = (
                        source_path,
                        original_width / image.width,
                        original_height / image.height,
                    )
            except Exception as exc:
                results[str(source_path.resolve())] = {"error": f"prepare: {type(exc).__name__}: {exc}"}
        if not detector_paths:
            return results
        request_path = temporary_dir / "request.json"
        response_path = temporary_dir / "response.json"
        request_path.write_text(
            json.dumps([str(path.resolve()) for path in detector_paths]) + "\n", encoding="utf-8"
        )
        command = [
            str(FACE_DETECT_PYTHON), str(FACE_DETECT_SCRIPT),
            "--input", str(request_path.resolve()), "--output", str(response_path.resolve()),
            "--level", FACE_MODEL_LEVEL, "--version", FACE_MODEL_VERSION,
            "--confidence", str(FACE_CONFIDENCE), "--iou", str(FACE_IOU_THRESHOLD),
        ]
        subprocess.run(command, check=True, env=detector_environment(mode))
        raw_results = json.loads(response_path.read_text(encoding="utf-8"))
        for detector_path, item in raw_results.items():
            source_path, scale_x, scale_y = mapping[detector_path]
            converted = dict(item)
            converted["detections"] = [
                {
                    **detection,
                    "bbox": [
                        round(detection["bbox"][0] * scale_x),
                        round(detection["bbox"][1] * scale_y),
                        round(detection["bbox"][2] * scale_x),
                        round(detection["bbox"][3] * scale_y),
                    ],
                }
                for detection in item.get("detections", [])
            ]
            results[str(source_path.resolve())] = converted
    return results


def select_detector_mode(requested: str) -> str:
    cpu_command = [
        str(FACE_DETECT_PYTHON),
        "-c",
        "import onnxruntime; from imgutils.detect import detect_faces; "
        "assert 'CPUExecutionProvider' in onnxruntime.get_available_providers(); "
        "print('face detector CPU ready:', onnxruntime.__version__)",
    ]
    gpu_command = [
        str(FACE_DETECT_PYTHON),
        "-c",
        "import ctypes, pathlib, onnxruntime; from imgutils.detect import detect_faces; "
        "providers = onnxruntime.get_available_providers(); "
        "assert 'CUDAExecutionProvider' in providers, "
        "f'CUDAExecutionProvider is unavailable: {providers}'; "
        "cuda_lib = pathlib.Path(onnxruntime.__file__).parent / 'capi' / "
        "'libonnxruntime_providers_cuda.so'; ctypes.CDLL(str(cuda_lib)); "
        "print('face detector GPU ready:', onnxruntime.__version__, providers)",
    ]
    if requested in {"auto", "gpu"}:
        gpu_result = subprocess.run(gpu_command, check=False, env=detector_environment())
        if gpu_result.returncode == 0:
            return "gpu"
        if gpu_result.returncode in {-signal.SIGINT, 128 + signal.SIGINT}:
            raise KeyboardInterrupt
        if requested == "gpu":
            raise RuntimeError("--device gpu が指定されましたがCUDA顔検出環境を利用できません")
        print("[device] GPUを利用できないためCPUへ切り替えます")
    try:
        subprocess.run(cpu_command, check=True, env=detector_environment())
    except subprocess.CalledProcessError as exc:
        if exc.returncode in {-signal.SIGINT, 128 + signal.SIGINT}:
            raise KeyboardInterrupt from exc
        raise RuntimeError(
            "CPUを含む顔検出環境を読み込めません。Dockerイメージを再ビルドしてください"
        ) from exc
    return "cpu"


def detect_chunked(paths: list[Path], mode: str) -> dict[str, dict]:
    results: dict[str, dict] = {}

    def run(chunk: list[Path]) -> None:
        try:
            results.update(detect_chunk(chunk, mode))
        except subprocess.CalledProcessError as exc:
            if exc.returncode in {-signal.SIGINT, 128 + signal.SIGINT}:
                raise KeyboardInterrupt from exc
            # A SIGKILL can be caused by memory pressure. Only that case benefits from splitting.
            if exc.returncode not in {-signal.SIGKILL, 128 + signal.SIGKILL}:
                raise RuntimeError(
                    f"顔検出プロセスが異常終了しました (exit={exc.returncode})"
                ) from exc
            if len(chunk) == 1:
                results[str(chunk[0].resolve())] = {
                    "error": "detector_process: killed while processing one image"
                }
            else:
                middle = len(chunk) // 2
                run(chunk[:middle])
                run(chunk[middle:])
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"顔検出プロセスの入出力に失敗しました: {exc}") from exc

    for start in range(0, len(paths), FACE_CHUNK_SIZE):
        chunk = paths[start:start + FACE_CHUNK_SIZE]
        print(f"[detect] {start + 1}-{start + len(chunk)}/{len(paths)}")
        run(chunk)
    return results


def collect_character(
    character_dir: Path, detector_mode: str, rebuild: bool = False
) -> dict[str, int]:
    face_dir = character_dir / "face"
    face_dir.mkdir(parents=True, exist_ok=True)
    caption = face_dir / COMMON_CAPTION_FILE
    caption.write_text(face_common_caption(character_dir) + "\n", encoding="utf-8")
    manifest_path = character_dir / "face_manifest.json"
    old_by_source = {
        str(record.get("source")): record
        for record in read_manifest(manifest_path)
        if record.get("source")
    }
    settings = detector_settings(detector_mode)
    records: list[dict] = []
    pending: list[tuple[str, Path, str]] = []
    for folder_name in SOURCE_FOLDERS:
        for source_path in list_images(character_dir / folder_name):
            relative = str(source_path.relative_to(character_dir))
            source_hash = sha256_file(source_path)
            old = old_by_source.get(relative)
            output_path = character_dir / old["output"] if old and old.get("output") else None
            unchanged = (
                not rebuild
                and old is not None
                and old.get("source_sha256") == source_hash
                and old.get("settings") == settings
                and (
                    old.get("status") == "discarded"
                    or (
                        old.get("status") == "accepted"
                        and output_path is not None
                        and output_path.is_file()
                        and old.get("output_sha256") == sha256_file(output_path)
                    )
                )
            )
            if unchanged:
                records.append(old)
            else:
                pending.append((folder_name, source_path, source_hash))
    known_hashes = [int(record["dhash"], 16) for record in records if record.get("dhash")]
    detected = detect_chunked([path for _, path, _ in pending], detector_mode) if pending else {}
    accepted = discarded = errors = 0
    for folder_name, source_path, source_hash in pending:
        relative = str(source_path.relative_to(character_dir))
        base = {"source": relative, "source_sha256": source_hash, "settings": settings}
        try:
            item = detected.get(str(source_path.resolve()), {})
            if item.get("error"):
                raise RuntimeError(item["error"])
            faces = []
            for detection in item.get("detections", []):
                box = tuple(int(value) for value in detection["bbox"])
                if min(box[2] - box[0], box[3] - box[1]) >= FACE_MIN_BOX_SIZE:
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    faces.append((area, box, float(detection["confidence"])))
            faces.sort(reverse=True, key=lambda value: value[0])
            if not faces:
                records.append({**base, "status": "discarded", "reason": "no_valid_face"})
                discarded += 1
                continue
            if len(faces) > 1 and faces[1][0] >= faces[0][0] * FACE_SECONDARY_AREA_RATIO:
                records.append({
                    **base, "status": "discarded", "reason": "ambiguous_multiple_faces",
                    "largest_area": faces[0][0], "second_area": faces[1][0],
                })
                discarded += 1
                continue
            _, box, confidence = faces[0]
            with Image.open(source_path) as opened:
                source = ImageOps.exif_transpose(opened).convert("RGB")
                crop = source.crop(expanded_square_box(box, source.width, source.height))
                crop = ImageOps.fit(crop, (FACE_OUTPUT_SIZE, FACE_OUTPUT_SIZE), Image.Resampling.LANCZOS)
            dhash = image_dhash(crop)
            if any((dhash ^ known).bit_count() <= FACE_DHASH_DISTANCE for known in known_hashes):
                records.append({**base, "status": "discarded", "reason": "duplicate"})
                discarded += 1
                continue
            output_path = face_dir / f"{folder_name}__{source_path.stem}__face00.png"
            crop.save(output_path, optimize=True)
            known_hashes.append(dhash)
            records.append({
                **base, "status": "accepted", "reason": "largest_face",
                "bbox": list(box), "confidence": confidence,
                "output": str(output_path.relative_to(character_dir)),
                "output_sha256": sha256_file(output_path), "dhash": f"{dhash:016x}",
            })
            accepted += 1
        except Exception as exc:
            records.append({**base, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})
            errors += 1
        finally:
            write_manifest(manifest_path, records)
    return {
        "accepted": accepted,
        "discarded": discarded,
        "errors": errors,
        "total_faces": len(list_images(face_dir)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--characters", nargs="*", help="例: character-a。省略時は全キャラ")
    parser.add_argument("--rebuild", action="store_true", help="全入力画像を再判定")
    parser.add_argument(
        "--device", choices=["auto", "gpu", "cpu"], default="auto",
        help="既定autoはGPUを優先し、利用不可ならCPUへ切り替える",
    )
    parser.add_argument("--check", action="store_true", help="顔検出環境だけ確認して終了")
    return parser.parse_args()


def main() -> None:
    global DATASET_ROOT, OUTPUT_DIR
    args = parse_args()
    DATASET_ROOT = args.dataset_root.expanduser().resolve()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    if args.check:
        detector_mode = select_detector_mode(args.device)
        print(f"[device] selected: {detector_mode}")
        print(f"[runtime] detector python: {sys.version.split()[0]}")
        return
    selected = set(args.characters or [])
    characters = [
        path for path in sorted(DATASET_ROOT.iterdir())
        if path.is_dir() and (not selected or path.name in selected)
    ] if DATASET_ROOT.is_dir() else []
    if not characters:
        raise RuntimeError(f"対象キャラクターが見つかりません: {DATASET_ROOT}")
    found_names = {path.name for path in characters}
    missing = selected - found_names
    if missing:
        raise RuntimeError("存在しないキャラクター: " + ", ".join(sorted(missing)))
    detector_mode = select_detector_mode(args.device)
    print(f"[device] selected: {detector_mode}")
    total_errors = 0
    for character_dir in characters:
        print(f"[character] {character_dir.name}")
        result = collect_character(character_dir, detector_mode, args.rebuild)
        print("[result]", result)
        total_errors += result["errors"]
    if total_errors:
        raise RuntimeError(
            f"{total_errors}件の画像処理に失敗しました。"
            "face_manifest.json の reason を確認して再実行してください"
        )
    print("顔データの準備が完了しました。このプロセスはここで終了します。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n顔収集を中断しました。次回実行時は未処理画像から再開します。")
        raise SystemExit(130)
