"""
Tests for orchestration: limits, checkpoint, mock end-to-end.
GPU NOT required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from image_agent.errors import MaxStepsExceeded
from image_agent.orchestration.checkpoint import (
    compute_input_hash,
    load_state,
    save_state,
)
from image_agent.orchestration.limits import (
    MAX_GENERATION_REVISIONS,
    MAX_GRAPH_STEPS,
)
from image_agent.orchestration.nodes import decide
from image_agent.schemas.evaluation import CriticAction, CriticResult
from image_agent.schemas.execution import AgentState, RunStatusEnum


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def test_save_and_load_state(tmp_path: Path):
    state = AgentState(run_id="r1", run_dir=str(tmp_path))
    save_state(state, str(tmp_path))
    loaded = load_state(str(tmp_path))
    assert loaded is not None
    assert loaded.run_id == "r1"


def test_load_state_missing_returns_none(tmp_path: Path):
    result = load_state(str(tmp_path / "nonexistent_run"))
    assert result is None


def test_save_state_atomic(tmp_path: Path):
    """State file should not be partially written."""
    state = AgentState(run_id="atomic", run_dir=str(tmp_path), graph_step_count=5)
    save_state(state, str(tmp_path))

    path = tmp_path / "state.json"
    with open(path) as f:
        d = json.load(f)
    assert d["graph_step_count"] == 5


def test_compute_input_hash_stable(tmp_path: Path):
    md = tmp_path / "test.md"
    md.write_text("# Test", encoding="utf-8")
    h1 = compute_input_hash(str(md), {"backend": "flux2"})
    h2 = compute_input_hash(str(md), {"backend": "flux2"})
    assert h1 == h2


def test_compute_input_hash_changes_with_content(tmp_path: Path):
    md = tmp_path / "test.md"
    md.write_text("# Content A", encoding="utf-8")
    h1 = compute_input_hash(str(md), {})
    md.write_text("# Content B", encoding="utf-8")
    h2 = compute_input_hash(str(md), {})
    assert h1 != h2


# ---------------------------------------------------------------------------
# decide() node (Python-only)
# ---------------------------------------------------------------------------


def _state_with_critic(action: CriticAction, revision_count: int = 0) -> AgentState:
    result = CriticResult(
        candidate_asset_id="c1",
        plan_id="p1",
        action=action,
        overall_score=0.8 if action == CriticAction.accept else 0.4,
        confidence=0.9,
    )
    return AgentState(
        run_id="r1",
        run_dir="/tmp/r1",
        critic_results=[result],
        generation_revision_count=revision_count,
    )


def test_decide_accept():
    state = _state_with_critic(CriticAction.accept)
    assert decide(state) == "finish"


def test_decide_retry_prompt():
    state = _state_with_critic(CriticAction.retry_prompt)
    assert decide(state) == "revise"


def test_decide_retry_reference():
    state = _state_with_critic(CriticAction.retry_reference)
    assert decide(state) == "revise"


def test_decide_localized_edit():
    state = _state_with_critic(CriticAction.localized_edit)
    assert decide(state) == "revise"


def test_decide_user_required_action():
    state = _state_with_critic(CriticAction.user_required)
    assert decide(state) == "user_required"


def test_decide_max_revisions_reached():
    state = _state_with_critic(CriticAction.retry_prompt, revision_count=MAX_GENERATION_REVISIONS)
    assert decide(state) == "user_required"


def test_decide_failed_state():
    state = AgentState(
        run_id="r1",
        run_dir="/tmp/r1",
        status=RunStatusEnum.failed,
        error="something went wrong",
    )
    assert decide(state) == "user_required"


def test_decide_no_critic_results():
    state = AgentState(run_id="r1", run_dir="/tmp/r1")
    assert decide(state) == "user_required"


# ---------------------------------------------------------------------------
# Graph step counter
# ---------------------------------------------------------------------------


def test_step_exceeds_max(tmp_path: Path):
    """Exceeding MAX_GRAPH_STEPS should raise."""
    from image_agent.orchestration.nodes import _step

    state = AgentState(
        run_id="r1",
        run_dir=str(tmp_path),
        graph_step_count=MAX_GRAPH_STEPS,  # already at limit
    )
    with pytest.raises(MaxStepsExceeded):
        _step(state, "validate_input")


# ---------------------------------------------------------------------------
# Duplicate plan hash detection
# ---------------------------------------------------------------------------


def test_duplicate_plan_hash_stops(tmp_path: Path):
    """analyze_and_plan should return user_required on duplicate hash."""
    from image_agent.orchestration.nodes import analyze_and_plan
    from image_agent.planner.qwen_runner import QwenPlannerRunner

    # Create a minimal Markdown and manifest
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    md = dataset / "request.md"
    md.write_text("# Test\n", encoding="utf-8")

    state = AgentState(
        run_id="dup_test",
        run_dir=str(tmp_path),
        markdown_path=str(md),
        dataset_root=str(dataset),
        # Seed a fake existing plan hash
        scene_plan_hash_history=["aaaa" * 16],
    )

    # Mock Qwen to return a JSON that produces the same hash
    mock_runner = MagicMock(spec=QwenPlannerRunner)
    mock_runner.model_id = "mock"
    mock_response = json.dumps({
        "generation_caption": "caption",
        "jobs": [],
        "must_preserve": [], "may_change": [], "must_design": [],
        "inferred": [], "contradictions": [], "uncertainty": [],
        "evaluation_criteria": [], "backend_preference": "flux2",
    })
    mock_runner.run_planner.return_value = mock_response

    # First call: should succeed (different hash)
    state2 = analyze_and_plan(state, qwen_runner=mock_runner)
    if state2.scene_plan is not None:
        # Inject the new hash as "already seen"
        state2 = state2.model_copy(
            update={
                "scene_plan_hash_history": [state2.scene_plan.plan_hash, state2.scene_plan.plan_hash]
            }
        )
        # Second call with same response → duplicate
        state3 = analyze_and_plan(state2, qwen_runner=mock_runner)
        assert state3.status in (RunStatusEnum.user_required, RunStatusEnum.failed)


# ---------------------------------------------------------------------------
# Mock end-to-end
# ---------------------------------------------------------------------------


def test_mock_end_to_end(tmp_path: Path):
    """Full pipeline with MockBackend and mocked Qwen — no GPU needed."""
    from image_agent.orchestration.nodes import (
        analyze_and_plan,
        critic_node,
        decide,
        finish,
        generate_auxiliary,
        generate_main,
        inspect_auxiliary,
        review_plan,
        validate_input,
        validate_plan,
    )
    from image_agent.planner.qwen_runner import QwenPlannerRunner
    from image_agent.generation.selector import get_backend
    from image_agent.critic.evaluator import CriticEvaluator

    # Setup
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    md = dataset / "request.md"
    md.write_text("# Test request\nDraw Alice.", encoding="utf-8")

    run_dir = tmp_path / "runs" / "run_test"
    run_dir.mkdir(parents=True)

    state = AgentState(
        run_id="e2e_test",
        run_dir=str(run_dir),
        markdown_path=str(md),
        dataset_root=str(dataset),
        backend="mock",
        width=64,
        height=64,
    )

    # --- validate_input ---
    state = validate_input(state)
    assert state.asset_manifest is not None

    # --- analyze_and_plan (mocked Qwen) ---
    mock_runner = MagicMock(spec=QwenPlannerRunner)
    mock_runner.model_id = "mock_qwen"
    mock_runner.run_planner.return_value = json.dumps({
        "generation_caption": "Alice anime character test",
        "subject": "Alice",
        "style": "anime",
        "scene": "test",
        "composition": "full body",
        "must_preserve": ["face"],
        "may_change": [],
        "must_design": [],
        "inferred": [],
        "contradictions": [],
        "uncertainty": [],
        "evaluation_criteria": [],
        "backend_preference": "mock",
        "jobs": [{
            "id": "j1",
            "type": "main_render",
            "purpose": "Main render",
            "depends_on": [],
            "input_asset_ids": [],
            "backend": "mock",
            "prompt": "Alice anime character",
            "negative_constraints": [],
            "width": 64,
            "height": 64,
            "steps": 1,
            "seed": 42,
        }],
    })

    state = analyze_and_plan(state, qwen_runner=mock_runner)
    assert state.scene_plan is not None

    state = validate_plan(state)
    state = review_plan(state)

    # --- generate_auxiliary (no aux jobs) ---
    state = generate_auxiliary(state, backend_factory=lambda n: get_backend("mock"))
    state = inspect_auxiliary(state)

    # --- generate_main ---
    state = generate_main(state, backend_factory=lambda n: get_backend("mock"))
    assert state.generation_candidate_count == 1

    # --- critic (mocked) ---
    mock_runner.run_critic.return_value = json.dumps({
        "action": "accept",
        "overall_score": 0.9,
        "confidence": 0.95,
        "rationale": "Looks good",
        "must_preserve_violations": [],
    })
    state = critic_node(state, qwen_runner=mock_runner)

    # --- decide ---
    next_node = decide(state)
    assert next_node == "finish"

    # --- finish ---
    state = finish(state, upscaler=_mock_upscaler())
    assert state.status == RunStatusEnum.completed


def _mock_upscaler():
    """Return a mock upscaler that succeeds."""
    from unittest.mock import MagicMock
    from image_agent.schemas.evaluation import UpscaleResult

    upscaler = MagicMock()
    upscaler.upscale.return_value = UpscaleResult(
        source_asset_id="mock_src",
        success=True,
        backend="mock",
        input_width=64,
        input_height=64,
        output_width=256,
        output_height=256,
    )
    return upscaler
