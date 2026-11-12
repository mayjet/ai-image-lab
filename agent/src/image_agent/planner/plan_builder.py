"""
Validate and build a ScenePlan from the parsed Qwen JSON response.

Validation pipeline (matches spec §12):
  raw dict
  -> Pydantic parse
  -> AssetManifest reconciliation
  -> job count check
  -> dependency validation
  -> cycle detection
  -> size validation
  -> backend validation
  -> conversion to GenerationJob objects
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from pydantic import ValidationError

from image_agent.errors import JobGraphError, PlanValidationError
from image_agent.schemas.assets import AssetManifest
from image_agent.schemas.planning import (
    BackendName,
    ConstraintSet,
    GenerationJob,
    JobGraph,
    JobStatus,
    JobType,
    ScenePlan,
)

MAX_AUXILIARY_JOBS = 3
MAX_GENERATION_CANDIDATES = 2
MAX_REFERENCES_PER_JOB = 4


def build_scene_plan_from_response(
    raw_dict: dict[str, Any],
    source_markdown: str,
    manifest: AssetManifest,
    planner_model: str = "",
    raw_response: Optional[str] = None,
    json_repaired: bool = False,
) -> ScenePlan:
    """
    Convert the parsed Qwen JSON dict into a validated ScenePlan.

    Raises PlanValidationError or JobGraphError on failure.
    """
    try:
        jobs_raw: list[dict] = raw_dict.get("jobs", [])
        jobs = _build_jobs(jobs_raw, manifest)
        job_graph = JobGraph(jobs=jobs)

        if job_graph.has_cycle():
            raise JobGraphError("Job graph contains a cycle")

        # Count auxiliary vs main jobs
        aux_jobs = [j for j in jobs if j.type != JobType.main_render and j.type != JobType.upscale]
        if len(aux_jobs) > MAX_AUXILIARY_JOBS:
            raise JobGraphError(
                f"Too many auxiliary jobs: {len(aux_jobs)} > {MAX_AUXILIARY_JOBS}"
            )

        constraints = ConstraintSet(
            must_preserve=raw_dict.get("must_preserve", []),
            may_change=raw_dict.get("may_change", []),
            must_design=raw_dict.get("must_design", []),
            inferred=raw_dict.get("inferred", []),
            contradictions=raw_dict.get("contradictions", []),
            uncertainty=raw_dict.get("uncertainty", []),
        )

        backend_pref = _parse_backend(raw_dict.get("backend_preference", "flux2"))
        generation_caption = raw_dict.get("generation_caption", "")
        if not generation_caption:
            raise PlanValidationError("generation_caption is required in planner output")

        # Compute plan hash for duplicate detection
        canonical = json.dumps(
            {
                "caption": generation_caption,
                "jobs": [j.model_dump(exclude={"id", "status", "output_asset_ids"}) for j in jobs],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        plan_hash = hashlib.sha256(canonical.encode()).hexdigest()

        return ScenePlan(
            plan_hash=plan_hash,
            source_markdown=source_markdown,
            asset_manifest_id=manifest.run_id,
            subject=raw_dict.get("subject", ""),
            style=raw_dict.get("style", ""),
            scene=raw_dict.get("scene", ""),
            composition=raw_dict.get("composition", ""),
            constraints=constraints,
            evaluation_criteria=raw_dict.get("evaluation_criteria", []),
            backend_preference=backend_pref,
            generation_caption=generation_caption,
            jobs=job_graph,
            planner_model=planner_model,
            raw_planner_response=raw_response,
            json_repaired=json_repaired,
        )
    except (JobGraphError, PlanValidationError):
        raise
    except ValidationError as exc:
        raise PlanValidationError(f"Pydantic validation failed: {exc}") from exc
    except Exception as exc:
        raise PlanValidationError(f"Plan build failed: {exc}") from exc


def _build_jobs(jobs_raw: list[dict], manifest: AssetManifest) -> list[GenerationJob]:
    jobs: list[GenerationJob] = []
    for raw in jobs_raw:
        # Validate job type
        try:
            jtype = JobType(raw.get("type", "main_render"))
        except ValueError:
            raise PlanValidationError(f"Unknown job type: {raw.get('type')!r}")

        # Reconcile input_asset_ids against manifest
        input_ids_raw: list[str] = raw.get("input_asset_ids", [])
        valid_ids = []
        for aid in input_ids_raw[:MAX_REFERENCES_PER_JOB]:
            if manifest.by_id(aid) is not None:
                valid_ids.append(aid)
            # Unknown asset IDs are silently dropped (Qwen may hallucinate)

        # Validate backend
        backend = _parse_backend(raw.get("backend", "flux2"))

        # Size validation
        width = int(raw.get("width", 512))
        height = int(raw.get("height", 512))
        if width % 8 != 0:
            width = (width // 8) * 8
        if height % 8 != 0:
            height = (height // 8) * 8
        width = max(64, min(2048, width))
        height = max(64, min(2048, height))

        try:
            job = GenerationJob(
                id=raw.get("id", ""),
                type=jtype,
                purpose=raw.get("purpose", ""),
                depends_on=raw.get("depends_on", []),
                input_asset_ids=valid_ids,
                backend=backend,
                prompt=raw.get("prompt", ""),
                negative_constraints=raw.get("negative_constraints", []),
                width=width,
                height=height,
                steps=int(raw.get("steps", 4)),
                seed=raw.get("seed"),
                status=JobStatus.pending,
            )
        except (ValidationError, ValueError) as exc:
            raise PlanValidationError(f"GenerationJob validation failed: {exc}") from exc
        jobs.append(job)

    # Validate that depends_on IDs reference known job IDs
    job_ids = {j.id for j in jobs}
    for job in jobs:
        for dep in job.depends_on:
            if dep not in job_ids:
                raise JobGraphError(
                    f"Job {job.id!r} depends on unknown job {dep!r}"
                )

    return jobs


def _parse_backend(raw: Any) -> BackendName:
    try:
        return BackendName(str(raw))
    except ValueError:
        return BackendName.flux2
