#!/usr/bin/env bash
set -euo pipefail

root="${AI_IMAGE_ROOT:-/ai-image-lab}"
backend_root="$root/flux2"
tensorboard_logdir="${TENSORBOARD_LOGDIR:-$backend_root/outputs/training/logs}"

cd "$backend_root"
mkdir -p \
    "$backend_root/cache/huggingface" \
    "$backend_root/models/base" \
    "$backend_root/outputs" \
    "$backend_root/vendor" \
    "${AI_IMAGE_FLUX2_MODEL_ROOT:-$root/loras/flux2/black-forest-labs--flux.2-klein-base-4b}" \
    "$tensorboard_logdir"

if [[ ! -f "$backend_root/vendor/diffusers/examples/dreambooth/train_dreambooth_lora_flux2_klein.py" ]]; then
    git clone --depth 1 --branch "${DIFFUSERS_REF:-v0.40.0}" \
        https://github.com/huggingface/diffusers.git \
        "$backend_root/vendor/diffusers"
fi

python3 -m ipykernel install --sys-prefix \
    --name "ai-image-lab-flux2-${AI_IMAGE_PLATFORM:-cuda}" \
    --display-name "Python (flux2-${AI_IMAGE_PLATFORM:-cuda})" >/dev/null

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
