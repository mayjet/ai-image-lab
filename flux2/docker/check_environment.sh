#!/usr/bin/env bash
set -euo pipefail

root="${AI_IMAGE_ROOT:-/ai-image-lab}"
python3 - <<'PY'
import os
import platform
import torch
from diffusers import Flux2KleinPipeline  # noqa: F401

print("backend: flux2")
print("platform:", os.environ.get("AI_IMAGE_PLATFORM", "cuda"))
print("capability:", os.environ.get("AI_IMAGE_CAPABILITY", "unknown"))
print("machine:", platform.machine())
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

test -d "$root/datasets"
test -f "$root/flux2/vendor/diffusers/examples/dreambooth/train_dreambooth_lora_flux2_klein.py"
