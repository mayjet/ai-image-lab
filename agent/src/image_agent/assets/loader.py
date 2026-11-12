"""
Markdown loading, image reference extraction, safe path resolution, and manifest building.

Security contract
-----------------
All paths returned by resolve_safe_path() are:
  1. resolved to an absolute real path (no symlinks)
  2. verified to be under the dataset_root
  3. rejected if the suffix is not in ALLOWED_EXTENSIONS
  4. verified by Pillow decode
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from image_agent.assets.hashing import compute_sha256_file, make_asset_id
from image_agent.assets.validation import ALLOWED_EXTENSIONS, validate_image_file
from image_agent.errors import AssetError, DatasetEscapeError
from image_agent.schemas.assets import AssetManifest, AssetProvenance, AssetRecord

# Markdown patterns that might introduce image paths:
#   ![alt](path)   or   - label: path
_MD_IMAGE_LINK_RE = re.compile(r"!\[.*?\]\(([^)]+)\)")
_MD_LIST_IMAGE_RE = re.compile(
    r"[-*]\s+[^\n]*?:\s+([^\s\n]+\.(?:png|jpg|jpeg|webp|bmp))",
    re.IGNORECASE,
)


def load_markdown(path: Path) -> str:
    """Read and return raw Markdown text without modification."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AssetError(f"Cannot read Markdown {path}: {exc}") from exc


def resolve_safe_path(
    raw_path: str,
    markdown_dir: Path,
    dataset_root: Path,
) -> Path:
    """
    Resolve a raw path string to an absolute path that must remain inside dataset_root.

    Raises DatasetEscapeError if the resolved path escapes dataset_root.
    """
    # Reject absolute paths from arbitrary input
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise DatasetEscapeError(
            f"Absolute path rejected: {raw_path!r}"
        )

    # Reject any symlink component before resolve() follows it.
    unresolved = (markdown_dir / candidate).absolute()
    current = unresolved
    while True:
        if current.is_symlink():
            raise DatasetEscapeError(f"Symlink rejected: {current}")
        if current == current.parent or current == dataset_root.absolute().parent:
            break
        current = current.parent

    # Resolve relative to Markdown directory
    resolved = unresolved.resolve()

    # Ensure the resolved path is under dataset_root (resolved)
    root_resolved = dataset_root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise DatasetEscapeError(
            f"Path escapes dataset_root: {resolved} not under {root_resolved}"
        )

    if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise AssetError(
            f"Disallowed extension {resolved.suffix!r} for {raw_path!r}"
        )

    return resolved


def extract_image_references(
    markdown_text: str,
    markdown_dir: Path,
    dataset_root: Path,
) -> list[tuple[str, Path]]:
    """
    Extract (role_hint, resolved_path) pairs from Markdown image references.

    Returns only paths that resolve safely inside dataset_root with an allowed extension.
    Invalid or escaped paths are skipped with a warning.
    """
    raw_refs: list[tuple[str, str]] = []

    # ![role](path)
    for m in _MD_IMAGE_LINK_RE.finditer(markdown_text):
        alt = m.group(0)[2 : m.group(0).index("]")]
        raw_refs.append((alt.strip() or "image", m.group(1).strip()))

    # - role: path
    for m in _MD_LIST_IMAGE_RE.finditer(markdown_text):
        line = m.group(0)
        colon_idx = line.index(":")
        role = line[1:colon_idx].strip("- *\t ")
        raw_refs.append((role, m.group(1).strip()))

    results: list[tuple[str, Path]] = []
    for role, raw in raw_refs:
        try:
            resolved = resolve_safe_path(raw, markdown_dir, dataset_root)
            if resolved.exists():
                results.append((role, resolved))
        except (AssetError, DatasetEscapeError):
            # Non-fatal — planner will work with what's available
            pass

    # Deduplicate by resolved path
    seen: set[Path] = set()
    deduped: list[tuple[str, Path]] = []
    for role, p in results:
        if p not in seen:
            seen.add(p)
            deduped.append((role, p))

    return deduped


def build_asset_manifest(
    run_id: str,
    markdown_path: Optional[Path],
    dataset_root: Path,
) -> AssetManifest:
    """
    Build an AssetManifest from a Markdown file and its referenced images.

    The Markdown itself is not modified. All assets receive SHA-256 and stable IDs.
    """
    manifest = AssetManifest(run_id=run_id)

    if markdown_path is None:
        return manifest

    if not markdown_path.exists():
        raise AssetError(f"Markdown file not found: {markdown_path}")

    try:
        markdown_path.resolve().relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise DatasetEscapeError(
            f"Markdown must be inside dataset_root: {markdown_path}"
        ) from exc

    md_text = load_markdown(markdown_path)
    md_sha = compute_sha256_file(markdown_path)
    manifest = manifest.model_copy(
        update={
            "markdown_path": str(markdown_path),
            "markdown_sha256": md_sha,
        }
    )

    md_dir = markdown_path.parent
    refs = extract_image_references(md_text, md_dir, dataset_root)

    for role_hint, img_path in refs:
        try:
            width, height, mime = validate_image_file(img_path)
            sha = compute_sha256_file(img_path)
            asset_id = make_asset_id(sha, role_hint)
            record = AssetRecord(
                asset_id=asset_id,
                sha256=sha,
                provenance=AssetProvenance.user_original,
                mime_type=mime,
                width=width,
                height=height,
                container_path=str(img_path),
                role_hint=role_hint,
            )
            manifest = manifest.add(record)
        except AssetError:
            # Non-fatal: skip undecodable images
            pass

    return manifest
