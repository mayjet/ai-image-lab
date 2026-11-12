"""Image file validation using Pillow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from image_agent.errors import AssetError

# Fixed set of permitted image extensions (lowercase)
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/bmp"}
)


def validate_image_file(path: Path) -> Tuple[int, int, str]:
    """
    Decode and validate an image file with Pillow.

    Returns (width, height, mime_type).
    Raises AssetError on failure.
    """
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as e:
        raise AssetError(f"Pillow not available: {e}") from e

    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise AssetError(
            f"Disallowed image extension {suffix!r}. "
            f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        raise AssetError(f"Image verification failed for {path}: {exc}") from exc

    # Re-open after verify (verify closes and may corrupt the object)
    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            fmt = img.format or ""
    except Exception as exc:
        raise AssetError(f"Image decode failed for {path}: {exc}") from exc

    mime = _format_to_mime(fmt)
    if mime not in ALLOWED_MIME_TYPES:
        raise AssetError(
            f"Decoded MIME type {mime!r} not allowed. Allowed: {sorted(ALLOWED_MIME_TYPES)}"
        )

    return width, height, mime


def _format_to_mime(fmt: str) -> str:
    mapping = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "WEBP": "image/webp",
        "BMP": "image/bmp",
    }
    return mapping.get(fmt.upper(), f"image/{fmt.lower()}")
