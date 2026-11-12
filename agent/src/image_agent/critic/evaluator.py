"""
Critic evaluator using Qwen3-VL via QwenPlannerRunner.

Runs Qwen in a subprocess and parses the structured evaluation JSON.
Decision logic (accept/retry/user_required) is implemented in Python,
not delegated to Qwen.
"""

from __future__ import annotations

from typing import Optional, Sequence

from image_agent.errors import JSONExtractionError, JSONRepairError, PlannerError
from image_agent.planner.json_extractor import extract_json, repair_json_once
from image_agent.planner.qwen_runner import QwenPlannerRunner
from image_agent.runtime.logging import get_logger
from image_agent.schemas.evaluation import CriticAction, CriticResult, ItemScore

logger = get_logger(__name__)

# Thresholds for automatic decision-making
ACCEPT_THRESHOLD = 0.65
LOW_CONFIDENCE_THRESHOLD = 0.5
MUST_PRESERVE_VIOLATION_LIMIT = 0


class CriticEvaluator:
    """
    Wraps QwenPlannerRunner.run_critic() and applies Python-side decision logic.
    """

    def __init__(self, qwen_runner: Optional[QwenPlannerRunner] = None) -> None:
        self._runner = qwen_runner or QwenPlannerRunner()

    def evaluate(
        self,
        markdown: str,
        reference_image_paths: Sequence[str],
        candidate_image_path: str,
        candidate_asset_id: str,
        plan_id: str,
        scene_plan_json: str,
    ) -> CriticResult:
        """
        Run critic evaluation and return a CriticResult with Python-decided action.
        """
        raw_text = self._runner.run_critic(
            markdown=markdown,
            reference_image_paths=reference_image_paths,
            candidate_image_path=candidate_image_path,
            scene_plan_json=scene_plan_json,
        )

        raw_dict = self._parse_critic_output(raw_text)
        result = self._build_result(raw_dict, candidate_asset_id, plan_id, raw_text)
        result = self._apply_decision_logic(result)
        return result

    def _parse_critic_output(self, raw_text: str) -> dict:
        """Parse Qwen critic JSON output with one-shot repair."""
        try:
            return extract_json(raw_text)
        except JSONExtractionError:
            pass
        try:
            return repair_json_once(raw_text)
        except JSONRepairError as exc:
            logger.warning("critic_json_repair_failed", extra={"error": str(exc)})
            # Return minimal safe default
            return {
                "action": "user_required",
                "overall_score": 0.0,
                "confidence": 0.0,
                "rationale": "Critic output could not be parsed",
            }

    def _build_result(
        self, raw: dict, candidate_asset_id: str, plan_id: str, raw_text: str
    ) -> CriticResult:
        def _item(key: str) -> Optional[ItemScore]:
            d = raw.get(key)
            if isinstance(d, dict):
                return ItemScore(
                    dimension=key,
                    score=float(d.get("score", 0.0)),
                    confidence=float(d.get("confidence", 0.0)),
                    notes=str(d.get("notes", "")),
                )
            return None

        try:
            action = CriticAction(raw.get("action", "user_required"))
        except ValueError:
            action = CriticAction.user_required

        return CriticResult(
            candidate_asset_id=candidate_asset_id,
            plan_id=plan_id,
            action=action,
            face_hair_body=_item("face_hair_body"),
            costume_emblem_items=_item("costume_emblem_items"),
            art_style=_item("art_style"),
            composition_pose=_item("composition_pose"),
            background=_item("background"),
            must_preserve_violations=raw.get("must_preserve_violations", []),
            overall_score=float(raw.get("overall_score", 0.0)),
            confidence=float(raw.get("confidence", 0.0)),
            rationale=str(raw.get("rationale", "")),
            suggested_revision=raw.get("suggested_revision"),
            raw_critic_response=raw_text,
        )

    def _apply_decision_logic(self, result: CriticResult) -> CriticResult:
        """
        Python-side decision override.

        Overrides Qwen's action when confidence or violation thresholds are crossed.
        """
        # Low confidence → user_required
        if result.confidence < LOW_CONFIDENCE_THRESHOLD:
            return result.model_copy(update={"action": CriticAction.user_required})

        # Must-preserve violations → user_required
        if len(result.must_preserve_violations) > MUST_PRESERVE_VIOLATION_LIMIT:
            return result.model_copy(update={"action": CriticAction.user_required})

        # Score below threshold despite accept → retry
        if result.action == CriticAction.accept and result.overall_score < ACCEPT_THRESHOLD:
            return result.model_copy(update={"action": CriticAction.retry_prompt})

        return result
