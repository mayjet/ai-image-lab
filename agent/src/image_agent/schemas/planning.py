"""Planning schemas: ScenePlan, GenerationJob, JobGraph."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class JobType(str, Enum):
    composition_draft = "composition_draft"
    pose_draft = "pose_draft"
    costume_sheet = "costume_sheet"
    item_sheet = "item_sheet"
    background_draft = "background_draft"
    main_render = "main_render"
    localized_edit = "localized_edit"
    upscale = "upscale"


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class BackendName(str, Enum):
    flux2 = "flux2"
    flux3 = "flux3"
    mock = "mock"


class ConstraintSet(BaseModel):
    """Extracted conditions from user Markdown."""

    must_preserve: list[str] = Field(default_factory=list)
    may_change: list[str] = Field(default_factory=list)
    must_design: list[str] = Field(default_factory=list)
    inferred: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


class GenerationJob(BaseModel):
    """A single image-generation task in the job graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: JobType
    purpose: str = Field(..., description="Human-readable description of what this job produces")
    depends_on: list[str] = Field(default_factory=list, description="IDs of prerequisite jobs")
    input_asset_ids: list[str] = Field(default_factory=list)
    backend: BackendName = BackendName.flux2
    prompt: str = ""
    negative_constraints: list[str] = Field(default_factory=list)
    width: int = 512
    height: int = 512
    steps: int = 4
    seed: Optional[int] = None
    status: JobStatus = JobStatus.pending
    output_asset_ids: list[str] = Field(default_factory=list)

    @field_validator("width", "height")
    @classmethod
    def _validate_size(cls, v: int) -> int:
        if v < 64 or v > 2048:
            raise ValueError(f"Dimension must be 64–2048, got {v}")
        if v % 8 != 0:
            raise ValueError(f"Dimension must be multiple of 8, got {v}")
        return v

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError(f"steps must be 1–100, got {v}")
        return v


class JobGraph(BaseModel):
    """Directed acyclic graph of generation jobs."""

    jobs: list[GenerationJob] = Field(default_factory=list)

    def by_id(self, job_id: str) -> Optional[GenerationJob]:
        for j in self.jobs:
            if j.id == job_id:
                return j
        return None

    def has_cycle(self) -> bool:
        """Return True if the dependency graph contains a cycle."""
        adj: dict[str, list[str]] = {j.id: list(j.depends_on) for j in self.jobs}
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            stack.add(node)
            for neighbour in adj.get(node, []):
                if neighbour not in visited:
                    if dfs(neighbour):
                        return True
                elif neighbour in stack:
                    return True
            stack.discard(node)
            return False

        for node in adj:
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def topological_order(self) -> list[str]:
        """Return job IDs in a valid execution order (raises on cycle)."""
        if self.has_cycle():
            raise ValueError("Job graph contains a cycle")
        in_degree: dict[str, int] = {j.id: 0 for j in self.jobs}
        for j in self.jobs:
            for dep in j.depends_on:
                in_degree[j.id] = in_degree.get(j.id, 0) + 1
        # Kahn's algorithm
        queue = [jid for jid, deg in in_degree.items() if deg == 0]
        result: list[str] = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for j in self.jobs:
                if node in j.depends_on:
                    in_degree[j.id] -= 1
                    if in_degree[j.id] == 0:
                        queue.append(j.id)
        if len(result) != len(self.jobs):
            raise ValueError("Topological sort incomplete — possible cycle")
        return result


class ScenePlan(BaseModel):
    """Full scene analysis and job graph produced by the Planner."""

    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    plan_hash: str = Field(
        ..., description="SHA-256 of the canonical plan JSON for duplicate detection"
    )

    # --- source material ---
    source_markdown: str = Field(..., description="Original user Markdown, unmodified")
    asset_manifest_id: str = Field(..., description="run_id of the AssetManifest")

    # --- scene analysis ---
    subject: str = ""
    style: str = ""
    scene: str = ""
    composition: str = ""

    # --- constraint classification ---
    constraints: ConstraintSet = Field(default_factory=ConstraintSet)

    # --- evaluation ---
    evaluation_criteria: list[str] = Field(default_factory=list)

    # --- generation ---
    backend_preference: BackendName = BackendName.flux2
    generation_caption: str = Field(
        ..., description="Complete FLUX caption; replaces Prompt Upsampler for FLUX.3"
    )
    jobs: JobGraph = Field(default_factory=JobGraph)

    # --- planner metadata ---
    planner_model: str = ""
    planner_version: str = ""
    raw_planner_response: Optional[str] = Field(
        None, description="Raw Qwen output before JSON extraction; stored for debugging"
    )
    json_repaired: bool = False

    @model_validator(mode="after")
    def _check_job_graph(self) -> "ScenePlan":
        if self.jobs.has_cycle():
            raise ValueError("ScenePlan job graph contains a cycle")
        return self
