"""Evaluation schemas: CriticResult, UpscaleResult."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CriticAction(str, Enum):
    accept = "accept"
    retry_prompt = "retry_prompt"
    retry_reference = "retry_reference"
    localized_edit = "localized_edit"
    user_required = "user_required"


class ItemScore(BaseModel):
    """Score for one evaluation dimension."""

    dimension: str
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""


class CriticResult(BaseModel):
    """Structured evaluation of a generated image."""

    candidate_asset_id: str
    plan_id: str
    action: CriticAction

    # Per-dimension scores
    face_hair_body: Optional[ItemScore] = None
    costume_emblem_items: Optional[ItemScore] = None
    art_style: Optional[ItemScore] = None
    composition_pose: Optional[ItemScore] = None
    background: Optional[ItemScore] = None
    must_preserve_violations: list[str] = Field(default_factory=list)

    # Aggregate
    overall_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = ""
    suggested_revision: Optional[str] = None

    # Raw model output stored for auditability
    raw_critic_response: Optional[str] = None
    critic_model: str = ""


class UpscaleResult(BaseModel):
    """Result of an upscale operation."""

    source_asset_id: str
    output_asset_ids: list[str] = Field(default_factory=list)
    backend: str = "realesrgan"
    model_name: str = ""
    input_width: int = 0
    input_height: int = 0
    output_width: int = 0
    output_height: int = 0
    tile_size: int = 256
    tile_padding: int = 16
    fp16: bool = True
    oom_fallback_used: bool = False
    success: bool = False
    error: Optional[str] = None
    qa_notes: list[str] = Field(default_factory=list)
