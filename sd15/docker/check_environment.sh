#!/usr/bin/env bash
set -euo pipefail

root="${AI_IMAGE_ROOT:-/ai-image-lab}"
python3 - <<'PY'
import os
import platform
import torch

print("backend: sd15")
print("platform:", os.environ.get("AI_IMAGE_PLATFORM", "cuda"))
print("machine:", platform.machine())
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

test -d "$root/datasets"
test -f "$root/sd15/vendor/sd-scripts/train_network.py"
if [[ "${AI_IMAGE_PLATFORM:-cuda}" == "l4t" ]]; then
    /opt/face-detector/bin/python "$root/sd15/anime_face_collect.py" --check --device auto
fi
