"""SHA-256 utilities and stable asset_id generation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def compute_sha256(data: bytes) -> str:
    """Return lowercase hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_sha256_file(path: Path) -> str:
    """Return lowercase hex SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_asset_id(sha256: str, role_hint: str = "") -> str:
    """
    Stable, URL-safe asset identifier.

    Format: sha256[:12] + "_" + slug(role_hint)[:16]
    Example: "a3f7c2d10b4e_face"
    """
    prefix = sha256[:12]
    slug = _slugify(role_hint)[:16] if role_hint else "asset"
    return f"{prefix}_{slug}"


def _slugify(text: str) -> str:
    """Convert arbitrary text to a lowercase ASCII slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    text = text.strip("_")
    return text or "asset"
