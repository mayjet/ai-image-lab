"""Execution and state-tracking schemas."""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from image_agent.schemas.assets import AssetManifest
from image_agent.schemas.evaluation import CriticResult, UpscaleResult
from image_agent.schemas.planning import ScenePlan


class NodeName(str, Enum):
    validate_input = "validate_input"
    analyze_and_plan = "analyze_and_plan"
    validate_plan = "validate_plan"
    review_plan = "review_plan"
    generate_auxiliary = "generate_auxiliary"
    inspect_auxiliary = "inspect_auxiliary"
    generate_main = "generate_main"
    critic = "critic"
    decide = "decide"
    finish = "finish"
    revise = "revise"
    user_required = "user_required"


class RunStatusEnum(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    user_required = "user_required"


class ExecutionMetrics(BaseModel):
    """Runtime resource and timing metrics."""

    node: Optional[str] = None
    job_id: Optional[str] = None
    started_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None
    elapsed_seconds: Optional[float] = None
    vram_peak_mib: Optional[float] = None
    vram_reserved_mib: Optional[float] = None
    ram_peak_mib: Optional[float] = None
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    quantization: Optional[str] = None
    compute_dtype: Optional[str] = None
    child_exit_code: Optional[int] = None
    fallback_reason: Optional[str] = None
    retry_reason: Optional[str] = None


class AgentState(BaseModel):
    """
    Mutable state passed through the LangGraph StateGraph.

    Images are never stored inline — only paths and IDs.
    """

    run_id: str
    run_dir: str

    # Input
    markdown_path: Optional[str] = None
    dataset_root: str = "/workspace/dataset"

    # Manifests and plans
    asset_manifest: Optional[AssetManifest] = None
    scene_plan: Optional[ScenePlan] = None
    scene_plan_hash_history: list[str] = Field(default_factory=list)

    # Generation tracking
    auxiliary_job_ids: list[str] = Field(default_factory=list)
    main_job_id: Optional[str] = None
    candidate_asset_ids: list[str] = Field(default_factory=list)
    critic_results: list[CriticResult] = Field(default_factory=list)
    final_asset_id: Optional[str] = None
    upscale_result: Optional[UpscaleResult] = None

    # Counters
    graph_step_count: int = 0
    auxiliary_job_count: int = 0
    generation_candidate_count: int = 0
    generation_revision_count: int = 0
    plan_repair_count: int = 0
    upscale_attempt_count: int = 0

    # Status
    current_node: Optional[str] = None
    status: RunStatusEnum = RunStatusEnum.pending
    error: Optional[str] = None
    user_message: Optional[str] = None

    # Config snapshot
    backend: str = "flux2"
    seed: Optional[int] = None
    width: int = 512
    height: int = 512


class RunStatus(BaseModel):
    """Persisted run summary (written to state.json)."""

    run_id: str
    run_dir: str
    status: RunStatusEnum
    current_node: Optional[str] = None
    error: Optional[str] = None
    user_message: Optional[str] = None
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    metrics: list[ExecutionMetrics] = Field(default_factory=list)
