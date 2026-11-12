"""
Checkpoint persistence for AgentState and RunStatus.

Each write is atomic: write to .tmp then rename to prevent partial reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from image_agent.runtime.logging import get_logger
from image_agent.schemas.execution import AgentState, RunStatus, RunStatusEnum

logger = get_logger(__name__)


def save_state(state: AgentState, run_dir: str) -> None:
    """Atomically write state.json and update run_status.json."""
    _atomic_write(
        Path(run_dir) / "state.json",
        state.model_dump(mode="json"),
    )
    # Lightweight run status for quick querying
    run_status = RunStatus(
        run_id=state.run_id,
        run_dir=run_dir,
        status=state.status,
        current_node=state.current_node,
        error=state.error,
        user_message=state.user_message,
    )
    _atomic_write(
        Path(run_dir) / "run_status.json",
        run_status.model_dump(mode="json"),
    )


def load_state(run_dir: str) -> Optional[AgentState]:
    """Load state.json from a previous run. Returns None if not found."""
    state_path = Path(run_dir) / "state.json"
    if not state_path.exists():
        return None
    try:
        with open(state_path, encoding="utf-8") as f:
            data = json.load(f)
        return AgentState.model_validate(data)
    except Exception as exc:
        logger.warning("state_load_failed", extra={"run_dir": run_dir, "error": str(exc)})
        return None


def compute_input_hash(markdown_path: str, config_snapshot: dict) -> str:
    """
    Hash the combination of input markdown content and config.

    Used to invalidate stale checkpoints when inputs change.
    """
    h = hashlib.sha256()
    try:
        with open(markdown_path, "rb") as f:
            h.update(f.read())
    except OSError:
        h.update(markdown_path.encode())
    h.update(json.dumps(config_snapshot, sort_keys=True).encode())
    return h.hexdigest()


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically using a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
