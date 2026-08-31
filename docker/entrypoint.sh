#!/usr/bin/env bash
set -euo pipefail

cd /workspace

backend="${AI_IMAGE_BACKEND:-sd15}"
platform="${AI_IMAGE_PLATFORM:-cuda}"
enable_tensorboard="${ENABLE_TENSORBOARD:-1}"
tensorboard_port="${TENSORBOARD_PORT:-6006}"
tensorboard_logdir="${TENSORBOARD_LOGDIR:-/workspace/outputs/${backend}/training/logs}"

mkdir -p \
    /workspace/cache/huggingface \
    "/workspace/cache/${backend}" \
    "/workspace/models/${backend}" \
    "/workspace/outputs/${backend}" \
    /workspace/vendor \
    "$tensorboard_logdir"

if [[ "$backend" == "sd15" ]]; then
    if [[ ! -f /workspace/vendor/sd-scripts/train_network.py ]]; then
        git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git \
            /workspace/vendor/sd-scripts
    fi
    if [[ ! -d /workspace/vendor/IP-Adapter ]]; then
        git clone --depth 1 https://github.com/tencent-ailab/IP-Adapter.git \
            /workspace/vendor/IP-Adapter
    fi
elif [[ "$backend" == "flux2" ]]; then
    # The maintained LoRA entry points live in Diffusers, not in BFL's
    # inference-only repository.  Keep the examples beside the mounted project.
    if [[ ! -f /workspace/vendor/diffusers/examples/dreambooth/train_dreambooth_lora_flux2_klein.py ]]; then
        git clone --depth 1 --branch "${DIFFUSERS_REF:-v0.40.0}" \
            https://github.com/huggingface/diffusers.git \
            /workspace/vendor/diffusers
    fi
else
    echo "Unsupported AI_IMAGE_BACKEND: $backend" >&2
    exit 2
fi

python3 -m ipykernel install --sys-prefix \
    --name "ai-image-lab-${backend}-${platform}" \
    --display-name "Python (${backend}-${platform})" >/dev/null

tensorboard_pid=""
if [[ "$enable_tensorboard" == "1" ]]; then
    python3 -m tensorboard.main \
        --logdir "$tensorboard_logdir" \
        --host 0.0.0.0 \
        --port "$tensorboard_port" \
        --reload_interval 2 &
    tensorboard_pid=$!
fi

"$@" &
application_pid=$!

shutdown() {
    kill "$application_pid" 2>/dev/null || true
    if [[ -n "$tensorboard_pid" ]]; then
        kill "$tensorboard_pid" 2>/dev/null || true
    fi
    wait "$application_pid" 2>/dev/null || true
    if [[ -n "$tensorboard_pid" ]]; then
        wait "$tensorboard_pid" 2>/dev/null || true
    fi
}
trap shutdown INT TERM EXIT

set +e
wait "$application_pid"
status=$?
set -e
exit "$status"
