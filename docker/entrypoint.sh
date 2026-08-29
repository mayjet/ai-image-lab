#!/usr/bin/env bash
set -euo pipefail

cd /workspace

mkdir -p ai-image-lab-work
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-/workspace/ai-image-lab-work/output/folder_lora_train/logs}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
mkdir -p "$TENSORBOARD_LOGDIR"

if [ ! -d "ai-image-lab-work/sd-scripts" ]; then
    git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git ai-image-lab-work/sd-scripts
fi

if [ ! -d "ai-image-lab-work/IP-Adapter" ]; then
    git clone --depth 1 https://github.com/tencent-ailab/IP-Adapter.git ai-image-lab-work/IP-Adapter
fi

python3 -m ipykernel install --sys-prefix --name ai-image-lab --display-name "Python (ai-image-lab)"

python3 -m tensorboard.main \
    --logdir "$TENSORBOARD_LOGDIR" \
    --host 0.0.0.0 \
    --port "$TENSORBOARD_PORT" \
    --reload_interval 1 &
tensorboard_pid=$!

"$@" &
application_pid=$!

shutdown() {
    kill "$application_pid" "$tensorboard_pid" 2>/dev/null || true
    wait "$application_pid" "$tensorboard_pid" 2>/dev/null || true
}
trap shutdown INT TERM EXIT

wait -n "$application_pid" "$tensorboard_pid"
status=$?
exit "$status"
