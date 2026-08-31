"""Parse the repository's Markdown prompt format for FLUX.2 generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REFERENCE_ORDER = ("source_image", "identity", "pose", "costume", "style")


@dataclass(frozen=True)
class LoraSpec:
    role: str
    source: str
    weight: float


@dataclass(frozen=True)
class ReferenceSpec:
    path: Path
    roles: tuple[str, ...]


@dataclass(frozen=True)
class FluxMarkdownPrompt:
    path: Path
    settings: dict[str, str]
    references: dict[str, str]
    lora: dict[str, str]
    generation_preferences: str
    positive: str
    negative: str
    rules: str


def _key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip():
            values[key.strip()] = value.strip()
    return values


def parse_markdown(path: Path) -> FluxMarkdownPrompt:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Markdownプロンプトがありません: {path}")
    sections: dict[str, list[str]] = {}
    current = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("## "):
            current = raw[3:].strip().lower()
            sections.setdefault(current, [])
        elif current:
            sections[current].append(raw)

    def section(name: str) -> str:
        return "\n".join(sections.get(name, [])).strip()

    positive = section("positive")
    if not positive:
        raise RuntimeError(f"## positive が空です: {path}")
    return FluxMarkdownPrompt(
        path=path,
        settings=_key_values(section("settings")),
        references=_key_values(section("references")),
        lora=_key_values(section("lora")),
        generation_preferences=section("generation_preferences"),
        positive=positive,
        negative=section("negative"),
        rules=section("rules"),
    )


def resolve_project_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def resolve_references(
    document: FluxMarkdownPrompt,
    project_root: Path,
    max_images: int = 4,
) -> list[ReferenceSpec]:
    if max_images < 0:
        raise ValueError("max_images must not be negative")
    if max_images == 0:
        return []
    paths: dict[Path, list[str]] = {}
    for role in REFERENCE_ORDER:
        value = document.references.get(role, "").strip()
        if not value:
            continue
        source = resolve_project_path(value, project_root)
        if source.is_dir():
            candidates = sorted(
                item for item in source.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not candidates:
                raise RuntimeError(f"参照画像がないフォルダです: {source}")
        elif source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
            candidates = [source]
        else:
            raise RuntimeError(f"参照画像またはフォルダがありません: {source}")
        for candidate in candidates:
            resolved = candidate.resolve()
            paths.setdefault(resolved, [])
            if role not in paths[resolved]:
                paths[resolved].append(role)
            if len(paths) >= max_images:
                break
        if len(paths) >= max_images:
            break
    return [ReferenceSpec(path=path, roles=tuple(roles)) for path, roles in paths.items()]


def resolve_loras(document: FluxMarkdownPrompt, project_root: Path) -> list[LoraSpec]:
    raw_roles = document.lora.get("use_roles", "").strip()
    if not raw_roles:
        return []
    specs: list[LoraSpec] = []
    for role in (item.strip() for item in raw_roles.split(",")):
        if not role:
            continue
        source = document.lora.get(role, "").strip()
        if not source:
            raise RuntimeError(f"LoRA role `{role}` のパスがありません")
        if source.startswith(("./", "../", "/", "~")):
            source_path = resolve_project_path(source, project_root)
            if not source_path.exists():
                raise RuntimeError(f"LoRAがありません: {source_path}")
            source = str(source_path)
        if "/sd15/" in source.replace("\\", "/"):
            raise RuntimeError(f"SD1.5 LoRAはFLUX.2で使用できません: {source}")
        try:
            weight = float(document.lora.get(f"{role}_weight", "1.0"))
        except ValueError as exc:
            raise RuntimeError(f"{role}_weight が数値ではありません") from exc
        if weight < 0:
            raise RuntimeError(f"{role}_weight は0以上にしてください")
        specs.append(LoraSpec(role=role, source=source, weight=weight))
    return specs


def compose_prompt(document: FluxMarkdownPrompt, references: list[ReferenceSpec]) -> str:
    parts = ["Main generation request:\n" + document.positive]
    if document.generation_preferences:
        parts.append("Visual style and generation preferences:\n" + document.generation_preferences)
    if references:
        lines = []
        for index, reference in enumerate(references, start=1):
            roles = ", ".join(reference.roles)
            lines.append(f"Reference image {index} is for these roles only: {roles}.")
        parts.append("Reference image role mapping:\n" + "\n".join(lines))
    if document.rules:
        parts.append("Rules that must be followed:\n" + document.rules)
    if document.negative:
        parts.append("Avoid all of the following:\n" + document.negative)
    return "\n\n".join(parts)


def setting_int(document: FluxMarkdownPrompt, name: str, default: int) -> int:
    value = document.settings.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"settings.{name} が整数ではありません: {value}") from exc


def setting_float(document: FluxMarkdownPrompt, name: str, default: float) -> float:
    value = document.settings.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"settings.{name} が数値ではありません: {value}") from exc
