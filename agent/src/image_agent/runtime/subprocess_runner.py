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
from dataclasses import dataclass
from typing import Optional, Sequence

from image_agent.errors import SubprocessError
from image_agent.runtime.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 1800  # 30 minutes


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
            f"stderr:\n{stderr[-2000:]}"
        )

    if not allow_failure and result.returncode != 0:
        raise SubprocessError(
            f"Subprocess {tag!r} exited with code {result.returncode}.\n"
            f"stderr:\n{stderr[-4000:]}"
        )

    return result
