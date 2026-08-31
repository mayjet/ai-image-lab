#!/usr/bin/env bash
set -euo pipefail

backend="${AI_IMAGE_BACKEND:-sd15}"
platform="${AI_IMAGE_PLATFORM:-cuda}"

python3 - <<'PY'
import os
import platform
import torch

print("backend:", os.environ.get("AI_IMAGE_BACKEND", "sd15"))
print("platform:", os.environ.get("AI_IMAGE_PLATFORM", "cuda"))
print("capability:", os.environ.get("AI_IMAGE_CAPABILITY", "unknown"))
print("machine:", platform.machine())
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory GiB:", round(props.total_memory / 1024**3, 2))
PY

python3 -c "import jupyterlab, notebook, ipykernel, tensorboard; print('Jupyter/TensorBoard: ready')"

if [[ "$backend" == "sd15" ]]; then
    python3 -c "import diffusers, transformers; from diffusers import StableDiffusionPipeline; print('SD1.5:', diffusers.__version__, transformers.__version__)"
    test -f /workspace/vendor/sd-scripts/train_network.py
    if [[ "$platform" == "l4t" ]]; then
        /opt/face-detector/bin/python /workspace/shared/anime_face_collect.py --check --device auto
    fi
else
    python3 -c "import diffusers, transformers; from diffusers import Flux2KleinPipeline; print('FLUX.2:', diffusers.__version__, transformers.__version__)"
    test -f /workspace/vendor/diffusers/examples/dreambooth/train_dreambooth_lora_flux2_klein.py
    if [[ "$platform" == "l4t" ]]; then
        echo "FLUX.2 Jetson check: environment/import only (training is intentionally unsupported)."
    else
        python3 -c "import bitsandbytes, peft; from diffusers.quantizers import PipelineQuantizationConfig; print('FLUX.2 4bit/offload extras: ready')"
    fi
fi
