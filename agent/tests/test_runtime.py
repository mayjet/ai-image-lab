"""
Tests for runtime utilities.
GPU NOT required.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from image_agent.runtime.logging import get_logger, redact, write_jsonl
from image_agent.runtime.subprocess_runner import run_subprocess
from image_agent.errors import SubprocessError


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------


def test_redact_dict_token():
    d = {"token": "hf_secret123", "other": "value"}
    r = redact(d)
    assert r["token"] == "[REDACTED]"
    assert r["other"] == "value"


def test_redact_nested():
    d = {"outer": {"hf_token": "secret", "safe": "ok"}}
    r = redact(d)
    assert r["outer"]["hf_token"] == "[REDACTED]"
    assert r["outer"]["safe"] == "ok"


def test_redact_hf_token_in_string():
    s = "token=hf_abcdefghij1234567890"
    r = redact(s)
    assert "hf_" not in r
    assert "[REDACTED]" in r


def test_redact_list():
    items = [{"password": "secret"}, {"safe": "value"}]
    r = redact(items)
    assert r[0]["password"] == "[REDACTED]"
    assert r[1]["safe"] == "value"


def test_redact_safe_values_unchanged():
    d = {"model_id": "Qwen/Qwen3-VL-4B-Instruct", "steps": 4}
    r = redact(d)
    assert r["model_id"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert r["steps"] == 4


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------


def test_write_jsonl(tmp_path: Path):
    log_file = tmp_path / "test.jsonl"
    write_jsonl(log_file, {"event": "test", "value": 42})
    write_jsonl(log_file, {"event": "second"})

    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "test"
    assert json.loads(lines[1])["event"] == "second"


def test_write_jsonl_redacts_secrets(tmp_path: Path):
    log_file = tmp_path / "secrets.jsonl"
    write_jsonl(log_file, {"token": "hf_mysecrettoken12345"})
    content = log_file.read_text()
    assert "hf_mysecrettoken12345" not in content
    assert "[REDACTED]" in content


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def test_subprocess_success(tmp_path: Path):
    result = run_subprocess(
        ["python3", "-c", "print('hello')"],
        log_tag="test",
    )
    assert result.returncode == 0
    assert "hello" in result.stdout


def test_subprocess_failure_raises():
    with pytest.raises(SubprocessError):
        run_subprocess(
            ["python3", "-c", "import sys; sys.exit(1)"],
            log_tag="test_fail",
        )


def test_subprocess_failure_reports_stdout_and_stderr():
    with pytest.raises(SubprocessError) as exc_info:
        run_subprocess(
            [
                "python3",
                "-c",
                "import sys; print('worker-json-error'); "
                "print('loader-detail', file=sys.stderr); sys.exit(1)",
            ],
            log_tag="test_failure_output",
        )
    message = str(exc_info.value)
    assert "worker-json-error" in message
    assert "loader-detail" in message


def test_subprocess_persists_complete_worker_logs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("IMAGE_AGENT_LOG_DIR", str(tmp_path))
    result = run_subprocess(
        [
            "python3",
            "-c",
            "import sys; print('complete-stdout'); print('complete-stderr', file=sys.stderr)",
        ],
        log_tag="qwen_worker",
    )
    assert result.returncode == 0
    stdout_logs = list(tmp_path.glob("qwen_worker-*.stdout.log"))
    stderr_logs = list(tmp_path.glob("qwen_worker-*.stderr.log"))
    assert len(stdout_logs) == 1
    assert len(stderr_logs) == 1
    assert "complete-stdout" in stdout_logs[0].read_text(encoding="utf-8")
    assert "complete-stderr" in stderr_logs[0].read_text(encoding="utf-8")


def test_subprocess_allow_failure():
    result = run_subprocess(
        ["python3", "-c", "import sys; sys.exit(1)"],
        allow_failure=True,
        log_tag="test_allow",
    )
    assert result.returncode == 1


def test_subprocess_timeout():
    with pytest.raises(SubprocessError) as exc_info:
        run_subprocess(
            ["python3", "-c", "import time; time.sleep(60)"],
            timeout=1,
            log_tag="test_timeout",
        )
    assert "timed out" in str(exc_info.value).lower()


def test_subprocess_no_shell_injection():
    """Args list prevents shell injection — semicolons are literal."""
    result = run_subprocess(
        ["python3", "-c", "print('safe; rm -rf /')"],
        log_tag="injection_test",
    )
    assert "safe" in result.stdout
    assert result.returncode == 0


def test_subprocess_env_passthrough(tmp_path: Path):
    result = run_subprocess(
        ["python3", "-c", "import os; print(os.environ.get('TEST_VAR', 'missing'))"],
        env={"TEST_VAR": "injected_value"},
        log_tag="env_test",
    )
    assert "injected_value" in result.stdout


# ---------------------------------------------------------------------------
# GPU info (CPU-only smoke test)
# ---------------------------------------------------------------------------


def test_get_gpu_info_no_crash():
    from image_agent.runtime.gpu_info import get_gpu_info
    info = get_gpu_info()
    # Should not raise even without a GPU
    assert isinstance(info.available, bool)


# ---------------------------------------------------------------------------
# Resource check (CPU-only)
# ---------------------------------------------------------------------------


def test_check_resources_no_crash(tmp_path: Path):
    from image_agent.runtime.resources import check_resources
    report = check_resources(workspace=tmp_path)
    assert isinstance(report.gpu_available, bool)
    assert isinstance(report.warnings, list)
