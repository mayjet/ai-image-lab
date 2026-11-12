"""
Tests for the planner module.
GPU NOT required — uses pre-constructed dicts instead of real Qwen.
"""

from __future__ import annotations

import pytest

from image_agent.errors import (
    JSONExtractionError,
    JSONRepairError,
    JobGraphError,
    PlanValidationError,
)
from image_agent.planner.json_extractor import extract_json, repair_json_once
from image_agent.planner.plan_builder import build_scene_plan_from_response
from image_agent.schemas.assets import AssetManifest, AssetProvenance, AssetRecord


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def test_extract_json_from_code_block():
    text = 'Here is the plan:\n```json\n{"key": "value"}\n```'
    result = extract_json(text)
    assert result == {"key": "value"}


def test_extract_json_from_bare_brace():
    text = 'some text {"answer": 42} more text'
    result = extract_json(text)
    assert result["answer"] == 42


def test_extract_json_no_json_raises():
    with pytest.raises(JSONExtractionError):
        extract_json("There is no JSON here")


def test_extract_json_invalid_json_raises():
    text = "```json\n{invalid json}\n```"
    with pytest.raises(JSONExtractionError):
        extract_json(text)


def test_repair_json_once_trailing_comma():
    text = '{"a": 1, "b": [1, 2,]}'
    result = repair_json_once(text)
    assert result["a"] == 1


def test_repair_json_once_python_true_false():
    text = '{"flag": True, "other": False, "val": None}'
    result = repair_json_once(text)
    assert result["flag"] is True
    assert result["other"] is False
    assert result["val"] is None


def test_repair_json_once_max_one_attempt():
    """repair_json_once does not loop — raises on irrecoverable JSON."""
    text = "this is {completely { broken } json"
    with pytest.raises(JSONRepairError):
        repair_json_once(text)


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------


def _make_manifest(run_id: str = "r1") -> AssetManifest:
    return AssetManifest(run_id=run_id)


def _basic_plan_dict(**overrides) -> dict:
    base = {
        "subject": "Alice",
        "style": "anime",
        "scene": "snowy mountain",
        "composition": "full body",
        "must_preserve": ["face", "hair"],
        "may_change": ["sleeve"],
        "must_design": ["winter coat"],
        "inferred": ["blue eyes"],
        "contradictions": [],
        "uncertainty": [],
        "evaluation_criteria": ["face matches reference"],
        "backend_preference": "flux2",
        "generation_caption": "Anime girl Alice in snowy mountain, full body, blue eyes",
        "jobs": [
            {
                "id": "j1",
                "type": "main_render",
                "purpose": "Main character",
                "depends_on": [],
                "input_asset_ids": [],
                "backend": "flux2",
                "prompt": "anime girl in snow",
                "negative_constraints": [],
                "width": 512,
                "height": 512,
                "steps": 4,
                "seed": None,
            }
        ],
    }
    base.update(overrides)
    return base


def test_build_scene_plan_basic():
    plan = build_scene_plan_from_response(
        raw_dict=_basic_plan_dict(),
        source_markdown="# Test",
        manifest=_make_manifest(),
    )
    assert plan.generation_caption != ""
    assert plan.plan_hash
    assert len(plan.jobs.jobs) == 1
    assert not plan.jobs.has_cycle()


def test_build_scene_plan_no_caption_raises():
    d = _basic_plan_dict(generation_caption="")
    with pytest.raises(PlanValidationError):
        build_scene_plan_from_response(d, "# Test", _make_manifest())


def test_build_scene_plan_too_many_aux_jobs():
    jobs = []
    for i in range(5):  # 4 aux + 1 main = 5 aux > MAX_AUXILIARY_JOBS=3
        jobs.append({
            "id": f"j{i}",
            "type": "pose_draft",
            "purpose": f"aux {i}",
            "depends_on": [],
            "input_asset_ids": [],
            "backend": "flux2",
            "prompt": "x",
            "width": 512,
            "height": 512,
            "steps": 4,
        })
    d = _basic_plan_dict(jobs=jobs)
    with pytest.raises(JobGraphError):
        build_scene_plan_from_response(d, "# Test", _make_manifest())


def test_build_scene_plan_cycle_raises():
    jobs = [
        {
            "id": "j1",
            "type": "pose_draft",
            "purpose": "x",
            "depends_on": ["j2"],
            "input_asset_ids": [],
            "backend": "flux2",
            "prompt": "x",
            "width": 512,
            "height": 512,
            "steps": 4,
        },
        {
            "id": "j2",
            "type": "main_render",
            "purpose": "x",
            "depends_on": ["j1"],
            "input_asset_ids": [],
            "backend": "flux2",
            "prompt": "x",
            "width": 512,
            "height": 512,
            "steps": 4,
        },
    ]
    d = _basic_plan_dict(jobs=jobs)
    with pytest.raises((JobGraphError, PlanValidationError)):
        build_scene_plan_from_response(d, "# Test", _make_manifest())


def test_build_scene_plan_unknown_dep_raises():
    jobs = [
        {
            "id": "j1",
            "type": "main_render",
            "purpose": "x",
            "depends_on": ["nonexistent"],
            "input_asset_ids": [],
            "backend": "flux2",
            "prompt": "x",
            "width": 512,
            "height": 512,
            "steps": 4,
        }
    ]
    d = _basic_plan_dict(jobs=jobs)
    with pytest.raises((JobGraphError, PlanValidationError)):
        build_scene_plan_from_response(d, "# Test", _make_manifest())


def test_build_scene_plan_invalid_asset_ids_filtered():
    """Unknown asset IDs in input_asset_ids are silently dropped."""
    manifest = _make_manifest()
    jobs = [
        {
            "id": "j1",
            "type": "main_render",
            "purpose": "x",
            "depends_on": [],
            "input_asset_ids": ["nonexistent_asset_id"],
            "backend": "flux2",
            "prompt": "x",
            "width": 512,
            "height": 512,
            "steps": 4,
        }
    ]
    d = _basic_plan_dict(jobs=jobs)
    plan = build_scene_plan_from_response(d, "# Test", manifest)
    # Unknown IDs should be dropped
    assert plan.jobs.jobs[0].input_asset_ids == []


def test_build_scene_plan_hash_stable():
    d = _basic_plan_dict()
    plan1 = build_scene_plan_from_response(d, "# Test", _make_manifest())
    plan2 = build_scene_plan_from_response(d, "# Test", _make_manifest())
    assert plan1.plan_hash == plan2.plan_hash


def test_build_scene_plan_hash_changes_with_caption():
    d1 = _basic_plan_dict(generation_caption="Caption A")
    d2 = _basic_plan_dict(generation_caption="Caption B")
    plan1 = build_scene_plan_from_response(d1, "# Test", _make_manifest())
    plan2 = build_scene_plan_from_response(d2, "# Test", _make_manifest())
    assert plan1.plan_hash != plan2.plan_hash


def test_scene_plan_json_repaired_flag():
    d = _basic_plan_dict()
    plan = build_scene_plan_from_response(d, "# Test", _make_manifest(), json_repaired=True)
    assert plan.json_repaired is True
