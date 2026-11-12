#!/usr/bin/env bash
set -euo pipefail

root="${AI_IMAGE_ROOT:-/ai-image-lab}"
backend_root="$root/sd15"
tensorboard_logdir="${TENSORBOARD_LOGDIR:-$backend_root/outputs/training/logs}"

cd "$backend_root"
mkdir -p \
    "$backend_root/cache/huggingface" \
    "$backend_root/models/base" \
    "$backend_root/outputs" \
    "$backend_root/vendor" \
    "${AI_IMAGE_SD15_MODEL_ROOT:-$root/loras/sd15/stablediffusionapi--counterfeit-v30}" \
    "$tensorboard_logdir"

if [[ ! -f "$backend_root/vendor/sd-scripts/train_network.py" ]]; then
    git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git \
        "$backend_root/vendor/sd-scripts"
fi
if [[ ! -d "$backend_root/vendor/IP-Adapter" ]]; then
    git clone --depth 1 https://github.com/tencent-ailab/IP-Adapter.git \
        "$backend_root/vendor/IP-Adapter"
fi

python3 -m ipykernel install --sys-prefix \
    --name "ai-image-lab-sd15-${AI_IMAGE_PLATFORM:-cuda}" \
    --display-name "Python (sd15-${AI_IMAGE_PLATFORM:-cuda})" >/dev/null

tensorboard_pid=""
if [[ "${ENABLE_TENSORBOARD:-1}" == "1" ]]; then
    python3 -m tensorboard.main \
        --logdir "$tensorboard_logdir" \
        --host 0.0.0.0 \
        --port "${TENSORBOARD_PORT:-6006}" \
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
