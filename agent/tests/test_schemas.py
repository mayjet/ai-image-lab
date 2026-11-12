"""
Tests for all Pydantic schemas.
GPU NOT required.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from image_agent.schemas import (
    AgentState,
    AssetManifest,
    AssetProvenance,
    AssetRecord,
    BackendName,
    CriticAction,
    CriticResult,
    ExecutionMetrics,
    GenerationJob,
    ItemScore,
    JobGraph,
    JobStatus,
    JobType,
    RunStatus,
    RunStatusEnum,
    ScenePlan,
    ConstraintSet,
    UpscaleResult,
)


# ---------------------------------------------------------------------------
# AssetRecord
# ---------------------------------------------------------------------------


def test_asset_record_valid():
    r = AssetRecord(
        asset_id="abc123_face",
        sha256="a" * 64,
        provenance=AssetProvenance.user_original,
        mime_type="image/png",
        width=512,
        height=512,
        container_path="/workspace/dataset/face.png",
    )
    assert r.sha256 == "a" * 64


def test_asset_record_invalid_sha256():
    with pytest.raises(ValidationError):
        AssetRecord(
            asset_id="x",
            sha256="not-hex",
            provenance=AssetProvenance.user_original,
            mime_type="image/png",
        )


def test_asset_provenance_values():
    for v in ["user_original", "agent_inferred", "agent_derived", "generated_candidate"]:
        p = AssetProvenance(v)
        assert p.value == v


# ---------------------------------------------------------------------------
# GenerationJob
# ---------------------------------------------------------------------------


def test_generation_job_valid():
    job = GenerationJob(
        type=JobType.main_render,
        purpose="Main character render",
        backend=BackendName.flux2,
        prompt="An anime girl in the snow",
        width=512,
        height=512,
        steps=4,
    )
    assert job.status == JobStatus.pending
    assert job.type == JobType.main_render


def test_generation_job_invalid_width():
    with pytest.raises(ValidationError):
        GenerationJob(
            type=JobType.main_render,
            purpose="x",
            width=63,  # below minimum
            height=512,
            steps=4,
        )


def test_generation_job_width_not_multiple_of_8():
    with pytest.raises(ValidationError):
        GenerationJob(
            type=JobType.main_render,
            purpose="x",
            width=513,  # not multiple of 8
            height=512,
            steps=4,
        )


def test_all_job_types():
    for jt in JobType:
        job = GenerationJob(type=jt, purpose="test", width=512, height=512, steps=4)
        assert job.type == jt


# ---------------------------------------------------------------------------
# JobGraph
# ---------------------------------------------------------------------------


def test_job_graph_no_cycle():
    j1 = GenerationJob(id="j1", type=JobType.pose_draft, purpose="x", width=512, height=512, steps=4)
    j2 = GenerationJob(id="j2", type=JobType.main_render, purpose="x", depends_on=["j1"], width=512, height=512, steps=4)
    g = JobGraph(jobs=[j1, j2])
    assert not g.has_cycle()
    order = g.topological_order()
    assert order.index("j1") < order.index("j2")


def test_job_graph_cycle_detected():
    j1 = GenerationJob(id="j1", type=JobType.pose_draft, purpose="x", depends_on=["j2"], width=512, height=512, steps=4)
    j2 = GenerationJob(id="j2", type=JobType.main_render, purpose="x", depends_on=["j1"], width=512, height=512, steps=4)
    g = JobGraph(jobs=[j1, j2])
    assert g.has_cycle()
    with pytest.raises(ValueError):
        g.topological_order()


def test_job_graph_empty():
    g = JobGraph(jobs=[])
    assert not g.has_cycle()
    assert g.topological_order() == []


# ---------------------------------------------------------------------------
# ScenePlan
# ---------------------------------------------------------------------------


def test_scene_plan_valid():
    plan = ScenePlan(
        plan_hash="a" * 64,
        source_markdown="# Test",
        asset_manifest_id="run_1",
        generation_caption="An anime character in the snow",
    )
    assert plan.plan_hash == "a" * 64


def test_scene_plan_cycle_raises():
    j1 = GenerationJob(id="j1", type=JobType.pose_draft, purpose="x", depends_on=["j2"], width=512, height=512, steps=4)
    j2 = GenerationJob(id="j2", type=JobType.main_render, purpose="x", depends_on=["j1"], width=512, height=512, steps=4)
    with pytest.raises(ValidationError):
        ScenePlan(
            plan_hash="b" * 64,
            source_markdown="# Test",
            asset_manifest_id="run_1",
            generation_caption="caption",
            jobs=JobGraph(jobs=[j1, j2]),
        )


# ---------------------------------------------------------------------------
# CriticResult
# ---------------------------------------------------------------------------


def test_critic_result_valid():
    r = CriticResult(
        candidate_asset_id="abc",
        plan_id="plan_1",
        action=CriticAction.accept,
        overall_score=0.85,
        confidence=0.9,
        rationale="Looks good",
    )
    assert r.action == CriticAction.accept


def test_critic_result_actions():
    for action in CriticAction:
        r = CriticResult(
            candidate_asset_id="x",
            plan_id="p1",
            action=action,
            overall_score=0.5,
            confidence=0.5,
        )
        assert r.action == action


def test_item_score_bounds():
    with pytest.raises(ValidationError):
        ItemScore(dimension="x", score=1.5, confidence=0.5)  # score > 1
    with pytest.raises(ValidationError):
        ItemScore(dimension="x", score=0.5, confidence=-0.1)  # confidence < 0


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------


def test_agent_state_defaults():
    state = AgentState(run_id="r1", run_dir="/tmp/r1")
    assert state.graph_step_count == 0
    assert state.status == RunStatusEnum.pending
    assert state.backend == "flux2"


def test_agent_state_serializable():
    state = AgentState(run_id="r1", run_dir="/tmp/r1")
    d = state.model_dump(mode="json")
    state2 = AgentState.model_validate(d)
    assert state2.run_id == state.run_id


# ---------------------------------------------------------------------------
# UpscaleResult
# ---------------------------------------------------------------------------


def test_upscale_result_valid():
    ur = UpscaleResult(
        source_asset_id="src_1",
        output_asset_ids=["out_1", "out_2"],
        backend="realesrgan",
        input_width=512,
        input_height=512,
        output_width=2048,
        output_height=2048,
        success=True,
    )
    assert ur.success


# ---------------------------------------------------------------------------
# ExecutionMetrics
# ---------------------------------------------------------------------------


def test_execution_metrics_optional_fields():
    m = ExecutionMetrics()
    assert m.vram_peak_mib is None
    assert m.elapsed_seconds is None
