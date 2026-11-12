"""
JSONL logging and secret redaction.

Each log line is a JSON object with at least:
  {"ts": "...", "level": "...", "logger": "...", "msg": "...", ...extra}

Secrets in the _SECRET_KEYS set are redacted before writing to any log.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Keys whose values should be redacted in logs
_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "token",
        "hf_token",
        "jupyter_token",
        "password",
        "secret",
        "api_key",
        "auth_key",
        "tailscale_authkey",
    }
)

# Regex to scrub tokens that look like HF tokens (hf_... or similar)
_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{10,}", re.IGNORECASE)

_REDACTED = "[REDACTED]"


def redact(value: Any) -> Any:
    """Recursively redact sensitive values from dicts/lists/strings."""
    if isinstance(value, dict):
        return {
            k: _REDACTED if k.lower() in _SECRET_KEYS else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _TOKEN_RE.sub(_REDACTED, value)
    return value


class _JsonlHandler(logging.Handler):
    """Writes JSON lines to a file or stderr."""

    def __init__(self, path: Optional[Path] = None) -> None:
        super().__init__()
        self._path = path
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a", encoding="utf-8")  # noqa: WPS515

    def emit(self, record: logging.LogRecord) -> None:
        entry: dict[str, Any] = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge extra fields
        for k, v in record.__dict__.items():
            if k not in {
                "args", "created", "exc_info", "exc_text", "filename", "funcName",
                "levelname", "levelno", "lineno", "message", "module", "msecs",
                "msg", "name", "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            }:
                entry[k] = redact(v)

        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)

        line = json.dumps(entry, ensure_ascii=False, default=str)
        line = redact(line)  # type: ignore[assignment]
        if isinstance(line, str):
            line_str = line
        else:
            line_str = json.dumps(line, ensure_ascii=False, default=str)

        if self._file:
            self._file.write(line_str + "\n")
            self._file.flush()
        else:
            print(line_str, file=sys.stderr)

    def close(self) -> None:
        if self._file:
            self._file.close()
        super().close()


_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, log_path: Optional[Path] = None) -> logging.Logger:
    """Return a named logger with JSONL output."""
    if name in _loggers:
        return _loggers[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = _JsonlHandler(log_path)
        logger.addHandler(handler)
        logger.propagate = False
    _loggers[name] = logger
    return logger


def write_jsonl(path: Path, entry: dict[str, Any]) -> None:
    """Append a single JSON object to a JSONL file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_entry = redact(entry)
    line = json.dumps(safe_entry, ensure_ascii=False, default=str)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
