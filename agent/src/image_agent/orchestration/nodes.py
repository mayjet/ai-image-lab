"""
LangGraph node implementations.

Each node is a plain Python function. Domain logic is NOT coupled to LangGraph —
only the function signature (AgentState -> AgentState) ties to the graph.

State transition:
  validate_input
  -> analyze_and_plan
  -> validate_plan
  -> review_plan
  -> generate_auxiliary
  -> inspect_auxiliary
  -> generate_main
  -> critic
  -> decide  (Python-only logic)
     -> finish | revise | user_required
"""

from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from pathlib import Path
from typing import Optional

from image_agent.assets.loader import build_asset_manifest, load_markdown
from image_agent.errors import (
    AssetError,
    BackendError,
    CudaOOMError,
    DuplicatePlanError,
    ImageAgentError,
    MaxStepsExceeded,
    PlannerError,
    UpscaleError,
    UpscalerOOMError,
    UserRequiredError,
)
from image_agent.orchestration.checkpoint import save_state
from image_agent.orchestration.limits import (
    MAX_AUXILIARY_JOBS,
    MAX_GENERATION_CANDIDATES,
    MAX_GENERATION_REVISIONS,
    MAX_GRAPH_STEPS,
    MAX_REFERENCES_PER_JOB,
    MAX_UPSCALE_ATTEMPTS,
)
from image_agent.planner.json_extractor import extract_json, repair_json_once
from image_agent.planner.plan_builder import build_scene_plan_from_response
from image_agent.planner.qwen_runner import QwenPlannerRunner
from image_agent.runtime.logging import get_logger, write_jsonl
from image_agent.schemas.assets import AssetManifest, AssetProvenance, AssetRecord
from image_agent.schemas.evaluation import CriticAction
from image_agent.schemas.execution import AgentState, RunStatusEnum
from image_agent.schemas.planning import GenerationJob, JobStatus, JobType

logger = get_logger(__name__)


def _step(state: AgentState, node_name: str) -> AgentState:
    """Increment step counter and check hard limit."""
    state = state.model_copy(
        update={
            "graph_step_count": state.graph_step_count + 1,
            "current_node": node_name,
        }
    )
    if state.graph_step_count > MAX_GRAPH_STEPS:
        raise MaxStepsExceeded(f"MAX_GRAPH_STEPS ({MAX_GRAPH_STEPS}) exceeded")
    save_state(state, state.run_dir)
    return state


def _log_jsonl(state: AgentState, event: str, **extra) -> None:
    write_jsonl(
        Path(state.run_dir) / "execution_log.jsonl",
        {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "run_id": state.run_id,
            "node": state.current_node,
            "event": event,
            **extra,
        },
    )


# ---------------------------------------------------------------------------
# Node: validate_input
# ---------------------------------------------------------------------------

def validate_input(state: AgentState) -> AgentState:
    state = _step(state, "validate_input")
    _log_jsonl(state, "node_start")

    if not state.markdown_path:
        state = state.model_copy(
            update={
                "status": RunStatusEnum.failed,
                "error": "No markdown_path provided",
            }
        )
        return state

    dataset_root = Path(state.dataset_root)
    md_path = Path(state.markdown_path)

    try:
        manifest = build_asset_manifest(state.run_id, md_path, dataset_root)
    except AssetError as exc:
        state = state.model_copy(
            update={"status": RunStatusEnum.failed, "error": str(exc)}
        )
        return state

    # Save asset manifest
    _atomic_json_write(
        Path(state.run_dir) / "asset_manifest.json",
        manifest.model_dump(mode="json"),
    )
    # Copy original Markdown (read-only; don't modify original)
    try:
        md_text = load_markdown(md_path)
        orig_dest = Path(state.run_dir) / "original_prompt.md"
        orig_dest.write_text(md_text, encoding="utf-8")
    except Exception:
        pass

    state = state.model_copy(
        update={
            "asset_manifest": manifest,
            "status": RunStatusEnum.running,
        }
    )
    save_state(state, state.run_dir)
    _log_jsonl(state, "node_done", asset_count=len(manifest.assets))
    return state


# ---------------------------------------------------------------------------
# Node: analyze_and_plan
# ---------------------------------------------------------------------------

def analyze_and_plan(state: AgentState, qwen_runner: Optional[QwenPlannerRunner] = None) -> AgentState:
    state = _step(state, "analyze_and_plan")
    _log_jsonl(state, "node_start")

    manifest: AssetManifest = state.asset_manifest  # type: ignore[assignment]
    md_path = Path(state.markdown_path)  # type: ignore[arg-type]
    md_text = load_markdown(md_path)

    # Collect user_original image paths (manifest may be None if validate_input was skipped)
    image_paths = []
    asset_id_map: dict[str, str] = {}
    if manifest is not None:
        image_paths = [
            a.container_path
            for a in manifest.assets
            if a.provenance == AssetProvenance.user_original and a.container_path
        ]
        asset_id_map = {
            a.container_path: a.asset_id
            for a in manifest.assets
            if a.provenance == AssetProvenance.user_original and a.container_path
        }

    runner = qwen_runner or QwenPlannerRunner()
    try:
        raw_text = runner.run_planner(
            markdown=md_text,
            image_paths=image_paths,
            asset_id_map=asset_id_map,
        )
    except (PlannerError, Exception) as exc:
        state = state.model_copy(
            update={"status": RunStatusEnum.failed, "error": f"Planner failed: {exc}"}
        )
        return state

    # Parse and validate
    raw_dict = None
    json_repaired = False
    from image_agent.errors import JSONExtractionError, JSONRepairError

    try:
        raw_dict = extract_json(raw_text)
    except JSONExtractionError:
        pass

    if raw_dict is None:
        from image_agent.orchestration.limits import MAX_PLAN_REPAIRS
        if state.plan_repair_count >= MAX_PLAN_REPAIRS:
            state = state.model_copy(
                update={
                    "status": RunStatusEnum.user_required,
                    "user_message": "Planner output could not be parsed after repair attempt",
                    "error": "JSON repair exhausted",
                }
            )
            return state
        try:
            raw_dict = repair_json_once(raw_text)
            json_repaired = True
            state = state.model_copy(update={"plan_repair_count": state.plan_repair_count + 1})
        except JSONRepairError as exc:
            state = state.model_copy(
                update={
                    "status": RunStatusEnum.user_required,
                    "user_message": "Planner returned unparseable output. Please retry or simplify the request.",
                    "error": str(exc),
                }
            )
            return state

    # Build ScenePlan
    try:
        plan = build_scene_plan_from_response(
            raw_dict=raw_dict,
            source_markdown=md_text,
            manifest=manifest,
            planner_model=runner.model_id,
            raw_response=raw_text,
            json_repaired=json_repaired,
        )
    except (PlannerError, Exception) as exc:
        state = state.model_copy(
            update={"status": RunStatusEnum.failed, "error": f"Plan build failed: {exc}"}
        )
        return state

    # Check for duplicate plan hash
    if plan.plan_hash in state.scene_plan_hash_history:
        state = state.model_copy(
            update={
                "status": RunStatusEnum.user_required,
                "user_message": "Planner produced a duplicate plan. The request may be ambiguous.",
                "error": f"Duplicate plan hash: {plan.plan_hash[:16]}",
            }
        )
        return state

    state = state.model_copy(
        update={
            "scene_plan": plan,
            "scene_plan_hash_history": state.scene_plan_hash_history + [plan.plan_hash],
        }
    )
    _atomic_json_write(
        Path(state.run_dir) / "scene_plan.json",
        plan.model_dump(mode="json"),
    )
    save_state(state, state.run_dir)
    _log_jsonl(state, "node_done", plan_hash=plan.plan_hash[:16])
    return state


# ---------------------------------------------------------------------------
# Node: validate_plan
# ---------------------------------------------------------------------------

def validate_plan(state: AgentState) -> AgentState:
    state = _step(state, "validate_plan")
    _log_jsonl(state, "node_start")

    plan = state.scene_plan
    if plan is None:
        state = state.model_copy(
            update={"status": RunStatusEnum.failed, "error": "No plan to validate"}
        )
        return state

    # Check plan constraints and report contradictions
    contradictions = plan.constraints.contradictions
    if contradictions:
        _log_jsonl(state, "plan_contradictions", items=contradictions)

    # Check generation caption
    if not plan.generation_caption:
        state = state.model_copy(
            update={
                "status": RunStatusEnum.user_required,
                "user_message": "Planner did not produce a generation caption. Please retry.",
            }
        )
        return state

    _log_jsonl(state, "node_done")
    return state


# ---------------------------------------------------------------------------
# Node: review_plan
# ---------------------------------------------------------------------------

def review_plan(state: AgentState) -> AgentState:
    """
    Present the plan for user review.

    In notebook mode, this is a no-op — the Notebook cell handles the display.
    The node just persists current state so it can be resumed after user confirmation.
    """
    state = _step(state, "review_plan")
    save_state(state, state.run_dir)
    _log_jsonl(state, "node_done", status="awaiting_user_or_auto_continue")
    return state


# ---------------------------------------------------------------------------
# Node: generate_auxiliary
# ---------------------------------------------------------------------------

def generate_auxiliary(state: AgentState, backend_factory=None) -> AgentState:
    state = _step(state, "generate_auxiliary")
    _log_jsonl(state, "node_start")

    plan = state.scene_plan
    if plan is None:
        return state

    aux_jobs = [
        j for j in plan.jobs.jobs
        if j.type not in (JobType.main_render, JobType.upscale)
        and j.status == JobStatus.pending
    ]

    if not aux_jobs:
        _log_jsonl(state, "node_done", skipped=True)
        return state

    if backend_factory is None:
        from image_agent.generation.selector import get_backend
        backend_factory = lambda name: get_backend(name, fallback_to_mock=False)

    aux_job_ids = []
    for job in aux_jobs:
        if len(aux_job_ids) >= MAX_AUXILIARY_JOBS:
            break
        try:
            from image_agent.generation.base import GenerationRequest
            backend = backend_factory(state.backend or job.backend.value)
            req = GenerationRequest(
                job_id=job.id,
                prompt=job.prompt,
                negative_constraints=job.negative_constraints,
                width=job.width,
                height=job.height,
                steps=job.steps,
                seed=job.seed,
                output_dir=str(Path(state.run_dir) / "auxiliary"),
            )
            result = backend.generate(req)
            if result.success and result.output_paths:
                aux_job_ids.append(job.id)
                # Register output as agent_derived asset
                state = _register_output_assets(state, result.output_paths, AssetProvenance.agent_derived)
        except (BackendError, CudaOOMError, Exception) as exc:
            _log_jsonl(state, "aux_job_failed", job_id=job.id, error=str(exc))

    state = state.model_copy(update={"auxiliary_job_ids": aux_job_ids})
    save_state(state, state.run_dir)
    _log_jsonl(state, "node_done", aux_count=len(aux_job_ids))
    return state


# ---------------------------------------------------------------------------
# Node: inspect_auxiliary
# ---------------------------------------------------------------------------

def inspect_auxiliary(state: AgentState) -> AgentState:
    """Register auxiliary images as references for the main render."""
    state = _step(state, "inspect_auxiliary")
    _log_jsonl(state, "node_done")
    return state


# ---------------------------------------------------------------------------
# Node: generate_main
# ---------------------------------------------------------------------------

def generate_main(state: AgentState, backend_factory=None) -> AgentState:
    state = _step(state, "generate_main")
    _log_jsonl(state, "node_start")

    plan = state.scene_plan
    if plan is None:
        state = state.model_copy(
            update={"status": RunStatusEnum.failed, "error": "No plan for main generation"}
        )
        return state

    main_jobs = [
        j for j in plan.jobs.jobs
        if j.type == JobType.main_render and j.status == JobStatus.pending
    ]
    if not main_jobs:
        # No explicit main render job — synthesize one from the plan
        main_job = GenerationJob(
            type=JobType.main_render,
            purpose="Main render from generation_caption",
            prompt=plan.generation_caption,
            width=state.width,
            height=state.height,
            steps=4,
            seed=state.seed,
            backend=plan.backend_preference,
        )
        main_jobs = [main_job]

    if state.generation_candidate_count >= MAX_GENERATION_CANDIDATES:
        state = state.model_copy(
            update={
                "status": RunStatusEnum.user_required,
                "user_message": f"Reached {MAX_GENERATION_CANDIDATES} generation candidates without acceptance.",
            }
        )
        return state

    if backend_factory is None:
        from image_agent.generation.selector import get_backend
        backend_factory = lambda name: get_backend(name, fallback_to_mock=False)

    job = main_jobs[0]
    try:
        from image_agent.generation.base import GenerationRequest
        # Collect reference images: user_original + agent_derived (aux)
        manifest: AssetManifest = state.asset_manifest  # type: ignore[assignment]
        ref_paths = [
            a.container_path
            for a in manifest.assets
            if a.provenance in (AssetProvenance.user_original, AssetProvenance.agent_derived)
            and a.container_path
        ][:MAX_REFERENCES_PER_JOB]

        backend = backend_factory(state.backend or job.backend.value)
        req = GenerationRequest(
            job_id=job.id,
            prompt=job.prompt or plan.generation_caption,
            negative_constraints=job.negative_constraints,
            width=state.width or job.width,
            height=state.height or job.height,
            steps=job.steps,
            seed=state.seed if state.seed is not None else job.seed,
            reference_image_paths=ref_paths,
            output_dir=str(Path(state.run_dir) / "candidates"),
        )
        result = backend.generate(req)
    except CudaOOMError as exc:
        _log_jsonl(state, "cuda_oom", error=str(exc))
        # Try reduced resolution
        try:
            req.width = max(256, req.width // 2)
            req.height = max(256, req.height // 2)
            result = backend.generate(req)
        except Exception as exc2:
            state = state.model_copy(
                update={"status": RunStatusEnum.failed, "error": f"CUDA OOM: {exc2}"}
            )
            return state
    except BackendError as exc:
        state = state.model_copy(
            update={"status": RunStatusEnum.failed, "error": str(exc)}
        )
        return state

    if not result.success:
        state = state.model_copy(
            update={"status": RunStatusEnum.failed, "error": result.error or "Generation failed"}
        )
        return state

    # Register outputs first, then read the IDs back from the updated manifest
    # so they are consistent (avoids role_hint mismatch between register and path_to_id).
    old_manifest = state.asset_manifest
    old_ids = {a.asset_id for a in old_manifest.assets} if old_manifest else set()
    state = _register_output_assets(state, result.output_paths, AssetProvenance.generated_candidate)
    new_manifest = state.asset_manifest
    new_candidate_ids = [
        a.asset_id
        for a in (new_manifest.assets if new_manifest else [])
        if a.asset_id not in old_ids
        and a.provenance == AssetProvenance.generated_candidate
    ]

    state = state.model_copy(
        update={
            "main_job_id": job.id,
            "candidate_asset_ids": state.candidate_asset_ids + new_candidate_ids,
            "generation_candidate_count": state.generation_candidate_count + 1,
        }
    )
    save_state(state, state.run_dir)
    _log_jsonl(state, "node_done", candidate_count=state.generation_candidate_count)
    return state


# ---------------------------------------------------------------------------
# Node: critic
# ---------------------------------------------------------------------------

def critic_node(state: AgentState, qwen_runner: Optional[QwenPlannerRunner] = None) -> AgentState:
    state = _step(state, "critic")
    _log_jsonl(state, "node_start")

    if not state.candidate_asset_ids:
        state = state.model_copy(
            update={"status": RunStatusEnum.failed, "error": "No candidates to evaluate"}
        )
        return state

    manifest: AssetManifest = state.asset_manifest  # type: ignore[assignment]
    plan = state.scene_plan
    if plan is None:
        state = state.model_copy(update={"status": RunStatusEnum.failed, "error": "No plan"})
        return state

    # Get the latest candidate
    candidate_id = state.candidate_asset_ids[-1]
    candidate = manifest.by_id(candidate_id)
    if candidate is None or not candidate.container_path:
        state = state.model_copy(
            update={"status": RunStatusEnum.failed, "error": f"Candidate {candidate_id} not found"}
        )
        return state

    reference_paths = [
        a.container_path
        for a in manifest.assets
        if a.provenance in (AssetProvenance.user_original, AssetProvenance.agent_derived)
        and a.container_path
    ]

    md_text = load_markdown(Path(state.markdown_path))  # type: ignore[arg-type]

    from image_agent.critic.evaluator import CriticEvaluator
    evaluator = CriticEvaluator(qwen_runner=qwen_runner or QwenPlannerRunner())

    try:
        result = evaluator.evaluate(
            markdown=md_text,
            reference_image_paths=reference_paths,
            candidate_image_path=candidate.container_path,
            candidate_asset_id=candidate_id,
            plan_id=plan.plan_id,
            scene_plan_json=json.dumps(plan.model_dump(mode="json", exclude={"jobs"})),
        )
    except Exception as exc:
        result_dict = {
            "action": "user_required",
            "overall_score": 0.0,
            "confidence": 0.0,
            "rationale": f"Critic failed: {exc}",
        }
        from image_agent.schemas.evaluation import CriticAction, CriticResult
        result = CriticResult(
            candidate_asset_id=candidate_id,
            plan_id=plan.plan_id,
            action=CriticAction.user_required,
            overall_score=0.0,
            confidence=0.0,
            rationale=str(exc),
        )

    _atomic_json_write(
        Path(state.run_dir) / "evaluation.json",
        result.model_dump(mode="json"),
    )
    state = state.model_copy(update={"critic_results": state.critic_results + [result]})
    save_state(state, state.run_dir)
    _log_jsonl(state, "node_done", action=result.action, score=result.overall_score)
    return state


# ---------------------------------------------------------------------------
# Node: decide  (Python-only)
# ---------------------------------------------------------------------------

def decide(state: AgentState) -> str:
    """
    Return the name of the next node based on critic results.

    This is a pure Python routing function — Qwen is NOT involved.
    """
    if state.status in (RunStatusEnum.failed, RunStatusEnum.user_required):
        return "user_required"

    if not state.critic_results:
        return "user_required"

    latest = state.critic_results[-1]
    action = latest.action

    if action.value == "accept":
        return "finish"

    if (
        state.generation_revision_count >= MAX_GENERATION_REVISIONS
        or state.generation_candidate_count >= MAX_GENERATION_CANDIDATES
    ):
        return "user_required"

    if action.value in ("retry_prompt", "retry_reference"):
        return "revise"

    if action.value == "localized_edit":
        return "revise"

    # user_required or unknown
    return "user_required"


# ---------------------------------------------------------------------------
# Node: revise
# ---------------------------------------------------------------------------

def revise(state: AgentState) -> AgentState:
    state = _step(state, "revise")
    _log_jsonl(state, "node_start")

    plan = state.scene_plan
    latest = state.critic_results[-1] if state.critic_results else None
    revision_hint = (
        latest.suggested_revision
        if latest and latest.suggested_revision
        else "Improve fidelity to all must_preserve constraints and reference-image geometry."
    )
    if plan is not None:
        revised_jobs = []
        for job in plan.jobs.jobs:
            if job.type == JobType.main_render:
                base_prompt = job.prompt or plan.generation_caption
                job = job.model_copy(
                    update={
                        "prompt": f"{base_prompt}\nRevision requirements: {revision_hint}",
                        "status": JobStatus.pending,
                    }
                )
            revised_jobs.append(job)
        revised_graph = plan.jobs.model_copy(update={"jobs": revised_jobs})
        plan = plan.model_copy(
            update={
                "generation_caption": (
                    f"{plan.generation_caption}\nRevision requirements: {revision_hint}"
                ),
                "jobs": revised_graph,
            }
        )

    state = state.model_copy(update={
        "generation_revision_count": state.generation_revision_count + 1,
        "scene_plan": plan,
    })
    save_state(state, state.run_dir)
    _log_jsonl(state, "node_done", revision_count=state.generation_revision_count)
    return state


# ---------------------------------------------------------------------------
# Node: finish
# ---------------------------------------------------------------------------

def finish(state: AgentState, upscaler=None) -> AgentState:
    state = _step(state, "finish")
    _log_jsonl(state, "node_start")

    if not state.candidate_asset_ids:
        state = state.model_copy(
            update={"status": RunStatusEnum.failed, "error": "No candidate to finalize"}
        )
        return state

    final_id = state.candidate_asset_ids[-1]
    manifest: AssetManifest = state.asset_manifest  # type: ignore[assignment]
    candidate = manifest.by_id(final_id)

    if candidate and candidate.container_path:
        # Copy base image to final/
        import shutil
        final_dir = Path(state.run_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        base_dest = final_dir / "base.png"
        shutil.copy2(candidate.container_path, str(base_dest))

        # Upscale
        if state.upscale_attempt_count < MAX_UPSCALE_ATTEMPTS:
            if upscaler is None:
                from image_agent.upscale.realesrgan_backend import RealESRGANUpscaler
                upscaler = RealESRGANUpscaler()
            try:
                upscale_result = upscaler.upscale(
                    source_asset_id=final_id,
                    input_path=str(base_dest),
                    output_dir=str(final_dir),
                )
                upscale_paths = list(upscale_result.output_asset_ids)
                old_ids = {a.asset_id for a in state.asset_manifest.assets}
                state = _register_output_assets(
                    state, upscale_paths, AssetProvenance.final_output
                )
                final_ids = [
                    a.asset_id
                    for a in state.asset_manifest.assets
                    if a.asset_id not in old_ids
                    and a.provenance == AssetProvenance.final_output
                ]
                if final_ids:
                    final_id = final_ids[-1]
                    upscale_result = upscale_result.model_copy(
                        update={"output_asset_ids": final_ids}
                    )
                _atomic_json_write(
                    Path(state.run_dir) / "upscale.json",
                    upscale_result.model_dump(mode="json"),
                )
                state = state.model_copy(update={"upscale_result": upscale_result})
            except (UpscaleError, UpscalerOOMError) as exc:
                _log_jsonl(state, "upscale_failed", error=str(exc))
                state = state.model_copy(
                    update={"upscale_attempt_count": state.upscale_attempt_count + 1}
                )

    state = state.model_copy(
        update={
            "final_asset_id": final_id,
            "status": RunStatusEnum.completed,
        }
    )
    save_state(state, state.run_dir)
    _log_jsonl(state, "node_done")
    return state


# ---------------------------------------------------------------------------
# Node: user_required
# ---------------------------------------------------------------------------

def user_required(state: AgentState) -> AgentState:
    state = _step(state, "user_required")
    if state.status != RunStatusEnum.failed:
        state = state.model_copy(update={"status": RunStatusEnum.user_required})
    save_state(state, state.run_dir)
    _log_jsonl(state, "node_done", message=state.user_message, error=state.error)
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atomic_json_write(path: Path, data: dict) -> None:
    from image_agent.orchestration.checkpoint import _atomic_write
    _atomic_write(path, data)


def _register_output_assets(
    state: AgentState,
    output_paths: list[str],
    provenance: AssetProvenance,
) -> AgentState:
    from image_agent.assets.hashing import compute_sha256_file, make_asset_id
    from image_agent.assets.validation import validate_image_file

    manifest: AssetManifest = state.asset_manifest  # type: ignore[assignment]
    for path in output_paths:
        try:
            w, h, mime = validate_image_file(Path(path))
            sha = compute_sha256_file(Path(path))
            asset_id = make_asset_id(sha, provenance.value)
            record = AssetRecord(
                asset_id=asset_id,
                sha256=sha,
                provenance=provenance,
                mime_type=mime,
                width=w,
                height=h,
                container_path=path,
            )
            manifest = manifest.add(record)
        except Exception:
            pass
    return state.model_copy(update={"asset_manifest": manifest})


def _path_to_asset_id(path: str) -> str:
    from image_agent.assets.hashing import compute_sha256_file, make_asset_id
    try:
        sha = compute_sha256_file(Path(path))
        return make_asset_id(sha, "candidate")
    except Exception:
        return hashlib.sha256(path.encode()).hexdigest()[:16]
