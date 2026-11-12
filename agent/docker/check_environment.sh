#!/usr/bin/env bash
# Diagnostic script for image-agent container environment.
# Exits 0 if environment is healthy, 1 if critical issues found.

set -euo pipefail

PASS=0
WARN=1
FAIL=2
overall=0

check() {
    local level=$1; shift
    local msg="$*"
    case $level in
        PASS) echo "  [PASS] $msg" ;;
        WARN) echo "  [WARN] $msg"; [[ $overall -lt $WARN ]] && overall=$WARN ;;
        FAIL) echo "  [FAIL] $msg"; overall=$FAIL ;;
    esac
}

echo "=== image-agent environment check ==="
echo ""

# --- Python ---
echo "Python:"
python_ver=$(python3 --version 2>&1) && check PASS "$python_ver" || check FAIL "python3 not found"

# --- Core imports ---
echo ""
echo "Python imports:"
python3 - <<'PY'
import sys
results = []
for mod in ["torch", "diffusers", "transformers", "pydantic", "langgraph",
            "PIL", "numpy", "yaml", "rich"]:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        results.append(("PASS", f"{mod}=={ver}"))
    except ImportError as e:
        results.append(("FAIL", f"{mod}: {e}"))
for status, msg in results:
    print(f"  [{status}] {msg}")
PY

# --- CUDA / GPU ---
echo ""
echo "GPU:"
python3 - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        mem_mib = props.total_memory // (1024 * 1024)
        print(f"  [PASS] {props.name}, VRAM {mem_mib} MiB")
        print(f"  [PASS] torch CUDA version: {torch.version.cuda}")
        if mem_mib < 7500:
            print(f"  [WARN] VRAM {mem_mib} MiB < 7500 MiB recommended")
    else:
        print("  [WARN] CUDA not available — running CPU-only / mock mode")
except Exception as e:
    print(f"  [WARN] GPU check failed: {e}")
PY

# --- Runtime directories ---
echo ""
echo "Runtime paths:"
for dir in \
    /workspace/dataset \
    /workspace/agent/runs \
    /workspace/agent/.runtime/models \
    /workspace/agent/.runtime/cache/huggingface \
    /workspace/agent/.runtime/logs; do
    if [[ -d "$dir" ]]; then
        check PASS "$dir exists"
    else
        check WARN "$dir missing (will be created at runtime)"
    fi
done

# --- Notebook ---
echo ""
echo "Notebook:"
if [[ -f /workspace/agent/image_agent.ipynb ]]; then
    check PASS "/workspace/agent/image_agent.ipynb present"
else
    check FAIL "/workspace/agent/image_agent.ipynb not found"
fi

# --- Tailscale (optional) ---
echo ""
echo "Tailscale:"
if command -v tailscale &>/dev/null; then
    check PASS "tailscale installed (borrowed GPU mode available)"
else
    check PASS "tailscale not installed (normal mode — OK)"
fi

# --- Summary ---
echo ""
echo "==========================="
case $overall in
    0) echo "Result: ALL CHECKS PASSED" ;;
    1) echo "Result: PASSED WITH WARNINGS" ;;
    2) echo "Result: FAILED"; exit 1 ;;
esac
