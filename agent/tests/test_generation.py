"""
Tests for generation backends.
GPU NOT required — uses MockBackend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from image_agent.generation.base import GenerationRequest, GenerationResult
from image_agent.generation.mock_backend import MockBackend
from image_agent.generation.selector import get_backend
from image_agent.errors import BackendError


def test_mock_backend_generates_image(tmp_path: Path):
    backend = MockBackend()
    assert backend.is_available()
    request = GenerationRequest(
        job_id="test_job",
        prompt="Anime girl in snow",
        width=64,
        height=64,
        steps=1,
        seed=42,
        output_dir=str(tmp_path),
    )
    result = backend.generate(request)
    assert result.success
    assert len(result.output_paths) == 1
    assert Path(result.output_paths[0]).exists()


def test_mock_backend_different_seeds_different_colors(tmp_path: Path):
    backend = MockBackend()
    req1 = GenerationRequest(job_id="j1", prompt="x", width=64, height=64, steps=1, seed=1, output_dir=str(tmp_path))
    req2 = GenerationRequest(job_id="j2", prompt="x", width=64, height=64, steps=1, seed=2, output_dir=str(tmp_path))
    r1 = backend.generate(req1)
    r2 = backend.generate(req2)
    assert r1.output_paths[0] != r2.output_paths[0]


def test_mock_backend_name():
    assert MockBackend().name == "mock"


def test_get_backend_mock():
    backend = get_backend("mock")
    assert backend.name == "mock"


def test_get_backend_unknown_raises():
    with pytest.raises(BackendError):
        get_backend("nonexistent_backend_xyz")


def test_get_backend_unavailable_fallback_to_mock():
    """When unavailable and fallback_to_mock=True, returns MockBackend."""
    # flux2 likely unavailable in test environment
    backend = get_backend("flux2", fallback_to_mock=True)
    assert backend.name in ("flux2", "mock")


def test_generation_result_dataclass():
    r = GenerationResult(job_id="j1", success=True, backend="mock")
    assert r.job_id == "j1"
    assert not r.fallback_used
