#!/usr/bin/env bash
set -euo pipefail

cd /workspace

echo "[system] architecture=$(uname -m)"
echo "[system] disk=$(df -h /workspace | awk 'NR==2 {print $4 " available"}')"

python3 - <<'PY'
import sys
import numpy
import torch

print(f"[train] python={sys.version.split()[0]}")
print(f"[train] numpy={numpy.__version__}")
print(f"[train] torch={torch.__version__}")
print(f"[train] torch_cuda={torch.version.cuda}")
print(f"[train] cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("学習環境からCUDAを利用できません")
PY

/opt/face-detector/bin/python /workspace/anime_face_collect.py --check --device auto

test -f /workspace/ai-image-lab-work/sd-scripts/train_network.py
echo "[train] sd-scripts=ready"

if curl -fsS http://127.0.0.1:8888/api/status >/dev/null 2>&1; then
    echo "[service] jupyter=ready"
else
    echo "[service] jupyter=not-running"
fi

if curl -fsS http://127.0.0.1:6006/ >/dev/null 2>&1; then
    echo "[service] tensorboard=ready"
else
    echo "[service] tensorboard=not-running"
fi
