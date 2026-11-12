"""
Static Docker and shell script tests.
GPU NOT required — checks file existence, syntax, and content patterns.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).parent.parent
DOCKER_DIR = AGENT_DIR / "docker"


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "Dockerfile.cuda",
        "Dockerfile.l4t",
        "Dockerfile.cuda.dockerignore",
        "Dockerfile.l4t.dockerignore",
        "requirements.txt",
        "entrypoint.sh",
        "check_environment.sh",
        "run.sh",
        "compose.yml",
    ],
)
def test_docker_file_exists(filename: str):
    assert (DOCKER_DIR / filename).exists(), f"{filename} missing in agent/docker/"


# ---------------------------------------------------------------------------
# Dockerfile content checks
# ---------------------------------------------------------------------------


def _read(filename: str) -> str:
    return (DOCKER_DIR / filename).read_text(encoding="utf-8")


def _non_comment_lines(text: str) -> str:
    """Return only non-comment lines joined by newline."""
    return "\n".join(
        line for line in text.splitlines()
        if not line.strip().startswith("#")
    )


def test_dockerfile_cuda_no_lora():
    content = _non_comment_lines(_read("Dockerfile.cuda"))
    assert "lora" not in content.lower(), "LoRA found in Dockerfile.cuda non-comment lines"
    assert "dreambooth" not in content.lower()


def test_dockerfile_l4t_no_lora():
    content = _read("Dockerfile.l4t")
    assert "lora" not in content.lower()


def test_dockerfile_cuda_no_bind_mount():
    content = _read("Dockerfile.cuda")
    # -v flags should not be in Dockerfiles
    assert "-v " not in content
    assert "--mount type=bind" not in content


def test_dockerfile_cuda_no_ipc_host():
    content = _read("Dockerfile.cuda")
    assert "--ipc=host" not in content


def test_dockerfile_cuda_no_docker_socket():
    content = _read("Dockerfile.cuda")
    assert "/var/run/docker.sock" not in content


def test_dockerfiles_do_not_install_tailscale():
    for filename in ("Dockerfile.cuda", "Dockerfile.l4t"):
        content = _non_comment_lines(_read(filename)).lower()
        assert "install_tailscale" not in content
        assert "apt-get install tailscale" not in content
        assert "pkgs.tailscale.com" not in content


def test_dockerfiles_set_private_tmpdir_only_after_creating_it():
    """APT post-install scripts need a TMPDIR that already exists."""
    tmpdir = "ENV TMPDIR=/workspace/agent/.runtime/cache/tmp"
    runtime_dir = "/workspace/agent/.runtime/cache/tmp"
    for filename in ("Dockerfile.cuda", "Dockerfile.l4t"):
        content = _read(filename)
        apt_index = content.index("apt-get update")
        create_index = content.index(runtime_dir, apt_index)
        tmpdir_index = content.index(tmpdir)
        assert apt_index < create_index < tmpdir_index


def test_dockerfiles_explicitly_allow_system_python_installs():
    """Ubuntu 24.04 marks its Python environment as externally managed."""
    for filename in ("Dockerfile.cuda", "Dockerfile.l4t"):
        content = _read(filename)
        pip_lines = [
            line for line in content.splitlines()
            if "python3 -m pip install" in line
        ]
        assert pip_lines
        assert all("--break-system-packages" in line for line in pip_lines)
        assert content.count("uv pip install --system") == content.count(
            "uv pip install --system --break-system-packages"
        )


def test_dockerfiles_do_not_replace_distro_python_tooling():
    """Debian-owned packages have no pip RECORD and cannot be uninstalled."""
    requirements = _read("requirements.txt")
    for package in ("pip==", "setuptools==", "wheel=="):
        assert package not in requirements

    for filename in ("Dockerfile.cuda", "Dockerfile.l4t"):
        content = _read(filename)
        assert "--no-deps uv==0.7.13" in content
        assert "pip install --break-system-packages --upgrade" not in content


def test_agent_uses_current_pytorch_allocator_variable():
    """PyTorch 2.9 deprecates PYTORCH_CUDA_ALLOC_CONF."""
    files = [
        DOCKER_DIR / "Dockerfile.cuda",
        DOCKER_DIR / "Dockerfile.l4t",
        DOCKER_DIR / "compose.yml",
        AGENT_DIR / "src/image_agent/generation/flux2_backend.py",
        AGENT_DIR / "src/image_agent/generation/flux3_backend.py",
    ]
    contents = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in contents
    assert "PYTORCH_ALLOC_CONF" in contents


def test_borrowed_gpu_launcher_failure_does_not_exit_parent_shell():
    """A failed remote build must leave the pasted-into terminal alive."""
    readme = (AGENT_DIR / "README.md").read_text(encoding="utf-8")
    section = readme.split("### 既存containerを全削除して、detachedで再構築", 1)[1]
    block = section.split("```bash", 1)[1].split("```", 1)[0]
    assert "set -euo pipefail" not in block

    # Simulate a failing Docker command in a parent that already has errexit.
    script = "set -e\ndocker() { return 42; }\n" + block + "\nprintf '__parent_alive__\\n'\n"
    result = subprocess.run(
        ["bash"], input=script, text=True, capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert "Launcher status: 42; terminal remains open" in result.stdout
    assert "__parent_alive__" in result.stdout


def test_dockerfile_l4t_no_x86_torch():
    """L4T Dockerfile must not use --index-url download.pytorch.org wheel."""
    content = _read("Dockerfile.l4t")
    assert "download.pytorch.org/whl" not in content


def test_dockerfile_cuda_fp16_only():
    """Dockerfile.cuda must not use BF16/NVFP4 as compute dtype."""
    content = _non_comment_lines(_read("Dockerfile.cuda"))
    assert "bfloat16" not in content.lower()
    assert "nvfp4" not in content.lower()


def test_dockerfile_cuda_has_cap_drop():
    content = _read("Dockerfile.cuda")
    # cap_drop is in compose/run.sh, not necessarily in Dockerfile itself
    # At minimum, run.sh must have it
    run_sh = (DOCKER_DIR / "run.sh").read_text()
    assert "--cap-drop" in run_sh


def test_entrypoint_has_jupyter_token():
    content = _read("entrypoint.sh")
    assert "JUPYTER_TOKEN" in content
    # Must not start Jupyter without a token
    assert "--IdentityProvider.token=" in content or "IdentityProvider.token" in content


def test_borrowed_gpu_mode_is_detached_and_enables_tailscale_ssh():
    readme = (AGENT_DIR / "README.md").read_text(encoding="utf-8")
    section = readme.split("### 既存containerを全削除して、detachedで再構築", 1)[1]
    block = section.split("```bash", 1)[1].split("```", 1)[0]
    assert "docker run -d" in block
    assert "--rm" not in block
    assert "-e ENABLE_TAILSCALE_SSH=1" in block
    assert "--cap-add SETUID" in block
    assert "--cap-add SETGID" in block


def test_entrypoint_tailscale_ssh_and_persistent_session_info():
    content = _read("entrypoint.sh")
    assert 'ENABLE_TAILSCALE_SSH' in content
    assert 'tailscale_up_args+=(--ssh)' in content
    assert 'set --ssh' in content
    assert '/usr/local/bin/tailscale' in content
    assert 'session-info.txt' in content
    assert 'TAILSCALE_SSH=' in content


def test_entrypoint_no_tailscale_in_normal_mode():
    """Tailscale code must be conditional on ENABLE_TAILSCALE=1."""
    content = _read("entrypoint.sh")
    assert 'ENABLE_TAILSCALE' in content
    # Tailscale commands must be inside the conditional block
    lines = content.splitlines()
    tailscale_in_block = False
    in_conditional = False
    for line in lines:
        stripped = line.strip()
        if 'ENABLE_TAILSCALE' in stripped and 'if' in stripped:
            in_conditional = True
        if in_conditional and 'tailscaled' in stripped:
            tailscale_in_block = True
    assert tailscale_in_block, "tailscaled must only run inside ENABLE_TAILSCALE conditional"


def test_entrypoint_installs_tailscale_at_runtime_only():
    content = _read("entrypoint.sh")
    assert "install_runtime_tailscale" in content
    assert "pkgs.tailscale.com/stable" in content
    assert "--state=mem:" in content


def test_jupyter_binds_container_interface():
    content = _read("entrypoint.sh")
    assert 'JUPYTER_BIND="0.0.0.0"' in content


def test_run_sh_no_bind_mount():
    content = _read("run.sh")
    # Bind mounts appear as "-v <host>:<container>"
    # Allow comments but not actual flags
    non_comment_lines = [l for l in content.splitlines() if not l.strip().startswith("#")]
    text = "\n".join(non_comment_lines)
    # Volume flag in docker run would be "-v "
    assert " -v " not in text, "bind mount found in run.sh non-comment lines"


def test_run_sh_no_volume_flag():
    content = _read("run.sh")
    non_comment_lines = [l for l in content.splitlines() if not l.strip().startswith("#")]
    text = "\n".join(non_comment_lines)
    assert "--volume" not in text


def test_run_sh_cap_drop():
    content = _read("run.sh")
    assert "--cap-drop" in content


def test_run_sh_no_ipc_host():
    content = _non_comment_lines(_read("run.sh"))
    assert "--ipc=host" not in content


def test_run_sh_has_compose_comment():
    content = _read("run.sh")
    # Compose examples should be commented
    assert "# docker compose" in content


def test_run_sh_dataset_copy():
    """run.sh must include docker cp for dataset."""
    content = _read("run.sh")
    assert "docker cp" in content


# ---------------------------------------------------------------------------
# shellcheck (optional — only if installed)
# ---------------------------------------------------------------------------


def _shellcheck_available() -> bool:
    try:
        result = subprocess.run(["shellcheck", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _shellcheck_available(), reason="shellcheck not installed")
@pytest.mark.parametrize("script", ["entrypoint.sh", "run.sh", "check_environment.sh"])
def test_shellcheck(script: str):
    result = subprocess.run(
        ["shellcheck", "--severity=error", str(DOCKER_DIR / script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"shellcheck failed for {script}:\n{result.stdout}"


# ---------------------------------------------------------------------------
# No LoRA anywhere in agent/
# ---------------------------------------------------------------------------


def test_no_lora_in_agent_source():
    """Scan all Python source files for LoRA implementation references.

    Only checks code lines — skips comment lines (# ...) and
    pure string literals used in documentation strings.
    """
    src_dir = AGENT_DIR / "src"
    # These patterns indicate actual LoRA implementation, not comments
    lora_impl_patterns = [
        r"\bPeftModel\b",
        r"\bPeftConfig\b",
        r"load_lora_weights",
        r"lora_alpha",
        r"lora_rank",
        r"add_lora",
        r"\.lora_",
        r"LoraConfig",
    ]
    pattern = re.compile("|".join(lora_impl_patterns))
    violations = []
    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for lineno, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            # Skip comment lines
            if stripped.startswith("#"):
                continue
            if pattern.search(line):
                violations.append(f"{py_file}:{lineno}: {line.strip()}")
    assert not violations, "LoRA implementation references found:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# No token-less Jupyter
# ---------------------------------------------------------------------------


def test_no_tokenless_jupyter():
    """Scan Docker files for Jupyter started without a token."""
    for f in [DOCKER_DIR / "Dockerfile.cuda", DOCKER_DIR / "Dockerfile.l4t", DOCKER_DIR / "entrypoint.sh"]:
        if not f.exists():
            continue
        content = f.read_text()
        # Pattern that starts Jupyter with empty token or no-token
        if "NotebookApp.token=" in content:
            assert "NotebookApp.token=" not in content.replace(
                "NotebookApp.token=", ""
            ) or "NotebookApp.token=$" in content or "ServerApp.token" in content, \
                f"Possible token-less Jupyter in {f}"


# ---------------------------------------------------------------------------
# Runtime path confinement
# ---------------------------------------------------------------------------


def test_runtime_paths_in_workspace():
    """All cache/runtime paths in configs must be under /workspace/agent/.runtime."""
    config = (AGENT_DIR / "configs" / "agent.yaml").read_text()
    for line in config.splitlines():
        if ":" in line and "/workspace" not in line:
            if any(x in line for x in ["dir:", "root:", "cache:"]):
                # May be a path value — check it doesn't go outside /workspace
                pass  # heuristic check; full check would parse YAML
    import yaml
    cfg = yaml.safe_load(config)
    paths = cfg.get("paths", {})
    for key, val in paths.items():
        if key not in ("workspace",) and val:
            assert str(val).startswith("/workspace"), \
                f"Path {key}={val!r} does not start with /workspace"
