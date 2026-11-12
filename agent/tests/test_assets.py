"""
Tests for the assets module.
GPU NOT required.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from image_agent.assets.hashing import compute_sha256, compute_sha256_file, make_asset_id
from image_agent.assets.loader import (
    build_asset_manifest,
    extract_image_references,
    load_markdown,
    resolve_safe_path,
)
from image_agent.errors import AssetError, DatasetEscapeError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dataset(tmp_path: Path):
    """Create a temporary dataset directory with a small PNG image."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    # Create a minimal 1x1 PNG
    (dataset / "face.png").write_bytes(_minimal_png())
    (dataset / "sub").mkdir()
    (dataset / "sub" / "body.png").write_bytes(_minimal_png())

    return dataset


def _minimal_png() -> bytes:
    """Return bytes of a valid 1x1 RGB PNG."""
    try:
        from PIL import Image
        import io

        img = Image.new("RGB", (1, 1), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Fallback: hand-crafted 1x1 PNG bytes
        return bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108020000"
            "0090wc3d980000000c4944415478016360f8cfc000000002000157e3"
            "5be50000000049454e44ae426082".replace("w", "")
        )


# ---------------------------------------------------------------------------
# SHA-256 / asset_id
# ---------------------------------------------------------------------------


def test_compute_sha256_bytes():
    data = b"hello world"
    expected = hashlib.sha256(data).hexdigest()
    assert compute_sha256(data) == expected


def test_compute_sha256_file(tmp_path: Path):
    f = tmp_path / "test.bin"
    f.write_bytes(b"test content")
    result = compute_sha256_file(f)
    assert len(result) == 64
    assert result == hashlib.sha256(b"test content").hexdigest()


def test_sha256_hash_stability(tmp_path: Path):
    """Same file content → same hash."""
    f = tmp_path / "stable.bin"
    f.write_bytes(b"stable")
    assert compute_sha256_file(f) == compute_sha256_file(f)


def test_make_asset_id_format():
    sha = "a" * 64
    aid = make_asset_id(sha, "顔")
    assert aid.startswith("aaaaaaaaaaaa_")
    assert len(aid) <= 32


def test_make_asset_id_no_hint():
    sha = "b" * 64
    aid = make_asset_id(sha)
    assert "_asset" in aid


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_resolve_safe_path_ok(tmp_dataset: Path):
    result = resolve_safe_path("face.png", tmp_dataset, tmp_dataset)
    assert result == (tmp_dataset / "face.png").resolve()


def test_resolve_safe_path_subdir(tmp_dataset: Path):
    result = resolve_safe_path("sub/body.png", tmp_dataset, tmp_dataset)
    assert result == (tmp_dataset / "sub" / "body.png").resolve()


def test_resolve_safe_path_dotdot_escape(tmp_dataset: Path):
    with pytest.raises(DatasetEscapeError):
        resolve_safe_path("../etc/passwd", tmp_dataset, tmp_dataset)


def test_resolve_safe_path_absolute(tmp_dataset: Path):
    with pytest.raises(DatasetEscapeError):
        resolve_safe_path("/etc/passwd", tmp_dataset, tmp_dataset)


def test_resolve_safe_path_disallowed_extension(tmp_dataset: Path):
    with pytest.raises(AssetError):
        resolve_safe_path("face.txt", tmp_dataset, tmp_dataset)


def test_resolve_safe_path_symlink(tmp_dataset: Path, tmp_path: Path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(_minimal_png())
    link = tmp_dataset / "link.png"
    os.symlink(str(outside), str(link))
    with pytest.raises(DatasetEscapeError):
        resolve_safe_path("link.png", tmp_dataset, tmp_dataset)


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------


def test_load_markdown(tmp_path: Path):
    md = tmp_path / "test.md"
    md.write_text("# Hello\n日本語テスト\n", encoding="utf-8")
    text = load_markdown(md)
    assert "Hello" in text
    assert "日本語" in text


def test_load_markdown_japanese(tmp_path: Path):
    md = tmp_path / "jp.md"
    content = "# キャラクター設定\n\n顔と髪は変更しない。"
    md.write_text(content, encoding="utf-8")
    assert load_markdown(md) == content


def test_load_markdown_english(tmp_path: Path):
    md = tmp_path / "en.md"
    content = "# Character request\n\nDraw Alice in a snowy mountain."
    md.write_text(content, encoding="utf-8")
    assert load_markdown(md) == content


def test_load_markdown_chinese(tmp_path: Path):
    md = tmp_path / "zh.md"
    content = "# 场景设定\n\n用低角度，保持原来的动漫画风。"
    md.write_text(content, encoding="utf-8")
    assert load_markdown(md) == content


def test_load_markdown_mixed(tmp_path: Path):
    md = tmp_path / "mixed.md"
    content = "# 雪山\nDraw in anime style.\n使用低角度视角。"
    md.write_text(content, encoding="utf-8")
    text = load_markdown(md)
    assert "雪山" in text
    assert "anime" in text
    assert "低角度" in text


def test_extract_image_references_markdown_links(tmp_dataset: Path):
    md_text = "![顔](face.png)\n![全身](sub/body.png)"
    refs = extract_image_references(md_text, tmp_dataset, tmp_dataset)
    paths = [str(p.name) for _, p in refs]
    assert "face.png" in paths
    assert "body.png" in paths


def test_extract_image_references_list_format(tmp_dataset: Path):
    md_text = "- 顔: face.png\n- 全身: sub/body.png"
    refs = extract_image_references(md_text, tmp_dataset, tmp_dataset)
    assert len(refs) >= 1


def test_extract_image_references_escape_filtered(tmp_dataset: Path):
    md_text = "![bad](../etc/passwd)"
    refs = extract_image_references(md_text, tmp_dataset, tmp_dataset)
    assert len(refs) == 0


def test_extract_image_references_nonexistent_filtered(tmp_dataset: Path):
    md_text = "![missing](missing.png)"
    refs = extract_image_references(md_text, tmp_dataset, tmp_dataset)
    assert len(refs) == 0


# ---------------------------------------------------------------------------
# AssetManifest
# ---------------------------------------------------------------------------


def test_build_asset_manifest(tmp_dataset: Path, tmp_path: Path):
    md = tmp_dataset / "request.md"
    md.write_text("# Test\n\n- 顔: face.png\n", encoding="utf-8")
    manifest = build_asset_manifest("test_run", md, tmp_dataset)
    assert manifest.run_id == "test_run"
    assert manifest.markdown_sha256 is not None
    assert len(manifest.assets) >= 1
    asset = manifest.assets[0]
    assert asset.sha256 and len(asset.sha256) == 64
    assert asset.asset_id
    assert asset.provenance.value == "user_original"


def test_asset_manifest_by_id(tmp_dataset: Path):
    md = tmp_dataset / "request.md"
    md.write_text("- 顔: face.png\n", encoding="utf-8")
    manifest = build_asset_manifest("r1", md, tmp_dataset)
    if manifest.assets:
        aid = manifest.assets[0].asset_id
        assert manifest.by_id(aid) is not None
        assert manifest.by_id("nonexistent") is None


def test_build_asset_manifest_no_images(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    md = dataset / "text_only.md"
    md.write_text("# No images here", encoding="utf-8")
    manifest = build_asset_manifest("r2", md, dataset)
    assert len(manifest.assets) == 0
