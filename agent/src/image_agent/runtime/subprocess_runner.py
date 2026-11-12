"""
Safe subprocess execution for model inference.

Design
------
- Arguments always passed as list[str] — never shell=True with concatenated strings.
- Each invocation has a timeout; exceeding it kills the process tree.
- stdout/stderr are captured and returned.
- SIGTERM/SIGINT forwarded to child via signal group.
- Exit code != 0 raises SubprocessError unless allow_failure=True.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from image_agent.errors import SubprocessError
from image_agent.runtime.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 1800  # 30 minutes


def _persist_worker_logs(tag: str, stdout: str, stderr: str) -> tuple[str, str]:
    """Persist complete worker output inside the disposable runtime directory."""
    log_dir = Path(os.environ.get("IMAGE_AGENT_LOG_DIR", "/workspace/agent/.runtime/logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        prefix = log_dir / f"{safe_tag}-{stamp}-{os.getpid()}"
        stdout_path = f"{prefix}.stdout.log"
        stderr_path = f"{prefix}.stderr.log"
        Path(stdout_path).write_text(stdout, encoding="utf-8", errors="replace")
        Path(stderr_path).write_text(stderr, encoding="utf-8", errors="replace")
        return stdout_path, stderr_path
    except OSError:
        return "", ""


@dataclass
class SubprocessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def run_subprocess(
    args: Sequence[str],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    env: Optional[dict[str, str]] = None,
    cwd: Optional[str] = None,
    allow_failure: bool = False,
    log_tag: str = "",
) -> SubprocessResult:
    """
    Run a subprocess with captured output and enforced timeout.

    The child process runs in its own process group so SIGKILL on timeout
    also kills any grandchildren.
    """
    cmd = list(args)  # ensure list[str] — no shell injection
    tag = log_tag or cmd[0].rsplit("/", 1)[-1]
    logger.info("subprocess_start", extra={"tag": tag, "cmd": cmd})

    merged_env = {**os.environ, **(env or {})}

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            cwd=cwd,
            start_new_session=True,  # own process group
            text=True,
        )
    except OSError as exc:
        raise SubprocessError(f"Cannot start subprocess {cmd[0]!r}: {exc}") from exc

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        # Kill entire process group
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()

    result = SubprocessResult(
        returncode=proc.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        timed_out=timed_out,
    )
    stdout_log, stderr_log = _persist_worker_logs(tag, result.stdout, result.stderr)
    log_hint = (
        f"\nFull logs: stdout={stdout_log}, stderr={stderr_log}"
        if stdout_log and stderr_log
        else ""
    )

    logger.info(
        "subprocess_done",
        extra={
            "tag": tag,
            "returncode": result.returncode,
            "timed_out": timed_out,
        },
    )

    if timed_out:
        raise SubprocessError(
            f"Subprocess {tag!r} timed out after {timeout}s.\n"
            f"stderr (last 8000 chars):\n{stderr[-8000:]}{log_hint}"
        )

    if not allow_failure and result.returncode != 0:
        raise SubprocessError(
            f"Subprocess {tag!r} exited with code {result.returncode}.\n"
            f"stdout (last 8000 chars):\n{result.stdout[-8000:]}\n"
            f"stderr (last 16000 chars):\n{result.stderr[-16000:]}{log_hint}"
        )

    return result
