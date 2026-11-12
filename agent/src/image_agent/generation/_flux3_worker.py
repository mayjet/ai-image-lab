"""
FLUX.3 ComfyUI INT8 inference worker.

Manages a temporary ComfyUI subprocess, submits a workflow, and collects results.
Runs as a subprocess itself so VRAM is released on exit.

Status: EXPERIMENTAL — RTX 2080 compatibility unverified.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


COMFYUI_STARTUP_TIMEOUT = 120  # seconds
COMFYUI_JOB_TIMEOUT = 600  # 10 minutes
COMFYUI_PORT = 18188  # non-standard port to avoid conflicts


def _start_comfyui(comfyui_dir: str, port: int, extra_args: list[str]) -> subprocess.Popen:
    """Start ComfyUI as a subprocess and return the Popen object."""
    args = [
        sys.executable,
        "-m", "comfy",
        "--listen", "127.0.0.1",
        "--port", str(port),
        "--output-directory", "/workspace/agent/.runtime/cache/comfyui_output",
        "--temp-directory", "/workspace/agent/.runtime/cache/comfyui_temp",
        "--cuda-malloc",
        "--fp16-vae",
    ] + extra_args

    env = dict(os.environ)
    env["COMFYUI_PATH"] = comfyui_dir

    return subprocess.Popen(
        args,
        cwd=comfyui_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _wait_for_comfyui(port: int, timeout: int = COMFYUI_STARTUP_TIMEOUT) -> bool:
    """Poll ComfyUI HTTP until ready or timeout."""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/system_stats", timeout=2
            ) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _submit_workflow(workflow_path: str, port: int) -> str:
    """Submit a workflow JSON and return the prompt_id."""
    import urllib.request

    with open(workflow_path, encoding="utf-8") as f:
        workflow = json.load(f)

    data = json.dumps({"prompt": workflow}).encode("utf-8")
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/prompt",
        data=data,
        timeout=30,
    ) as resp:
        result = json.loads(resp.read())
    return result["prompt_id"]


def _wait_for_job(prompt_id: str, port: int, timeout: int = COMFYUI_JOB_TIMEOUT) -> dict:
    """Poll ComfyUI history until the job completes or timeout."""
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/history/{prompt_id}", timeout=5
            ) as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                return history[prompt_id]
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"ComfyUI job {prompt_id} timed out after {timeout}s")


def run_generation(payload: dict) -> dict:
    workflow_path = payload["workflow_path"]
    output_dir = payload["output_dir"]
    comfyui_dir = payload.get("comfyui_dir", "/workspace/agent/.runtime/comfyui")
    port = int(payload.get("port", COMFYUI_PORT))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    proc = _start_comfyui(comfyui_dir, port, [])

    try:
        if not _wait_for_comfyui(port, COMFYUI_STARTUP_TIMEOUT):
            raise TimeoutError("ComfyUI did not start in time")

        prompt_id = _submit_workflow(workflow_path, port)
        job_result = _wait_for_job(prompt_id, port, COMFYUI_JOB_TIMEOUT)

        # Collect output images
        outputs = []
        for node_id, node_out in job_result.get("outputs", {}).items():
            for img_info in node_out.get("images", []):
                # Images are written to ComfyUI's output dir; copy to our output_dir
                src = Path("/workspace/agent/.runtime/cache/comfyui_output") / img_info["filename"]
                dst = Path(output_dir) / img_info["filename"]
                if src.exists():
                    import shutil
                    shutil.copy2(str(src), str(dst))
                    outputs.append(str(dst))

        elapsed = time.monotonic() - t0
        return {
            "status": "ok",
            "output_paths": outputs,
            "elapsed_seconds": elapsed,
        }
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "error": "No payload path"}))
        sys.exit(1)

    payload_path = sys.argv[1]
    try:
        with open(payload_path, encoding="utf-8") as f:
            payload = json.load(f)
        result = run_generation(payload)
        print(json.dumps(result))
    except MemoryError as exc:
        print(json.dumps({"status": "oom", "error": str(exc)}))
        sys.exit(2)
    except TimeoutError as exc:
        print(json.dumps({"status": "timeout", "error": str(exc)}))
        sys.exit(3)
    except Exception as exc:
        import traceback
        print(json.dumps({"status": "error", "error": str(exc), "traceback": traceback.format_exc()}))
        sys.exit(1)


if __name__ == "__main__":
    main()
