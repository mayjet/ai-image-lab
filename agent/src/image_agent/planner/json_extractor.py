"""
JSON extraction and one-shot repair from Qwen raw text output.

Rules
-----
- Try to find the first ```json ... ``` block; fall back to first {...} span.
- Attempt standard json.loads.
- If it fails, attempt ONE repair using simple heuristics.
- If repair also fails, raise JSONRepairError — do not loop.
"""

from __future__ import annotations

import json
import re
from typing import Any

from image_agent.errors import JSONExtractionError, JSONRepairError

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)```", re.IGNORECASE)
_BRACE_SPAN_RE = re.compile(r"\{[\s\S]+\}", re.DOTALL)


def extract_json(raw_text: str) -> Any:
    """
    Extract and parse the first JSON object from raw model output.

    Raises JSONExtractionError if no JSON-like text is found.
    """
    candidate = _extract_candidate(raw_text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JSONExtractionError(
            f"JSON parse failed: {exc}\nCandidate:\n{candidate[:500]}"
        ) from exc


def repair_json_once(raw_text: str) -> Any:
    """
    One-shot JSON repair attempt.

    Applies simple fixups and retries parse exactly once.
    Raises JSONRepairError if still invalid after repair.
    """
    candidate = _extract_candidate(raw_text)
    repaired = _apply_simple_repairs(candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise JSONRepairError(
            f"JSON repair failed: {exc}\nRepaired candidate:\n{repaired[:500]}"
        ) from exc


def _extract_candidate(raw_text: str) -> str:
    """Find the most likely JSON substring."""
    # Prefer a fenced code block
    m = _CODE_BLOCK_RE.search(raw_text)
    if m:
        return m.group(1).strip()
    # Fall back to first brace span
    m = _BRACE_SPAN_RE.search(raw_text)
    if m:
        return m.group(0).strip()
    raise JSONExtractionError("No JSON object found in model output")


def _apply_simple_repairs(text: str) -> str:
    """Apply heuristic repairs to malformed JSON."""
    # Remove trailing commas before ] or }
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Replace single quotes with double quotes (naïve; may break strings with apostrophes)
    # Only apply when the text clearly uses single-quoted JSON keys
    if re.search(r"'[^']+'\s*:", text) and '"' not in text:
        text = text.replace("'", '"')
    # Remove JavaScript-style comments
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Normalize Python True/False/None to JSON true/false/null
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text
