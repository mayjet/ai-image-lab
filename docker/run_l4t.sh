#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-ai-image-lab:l4t}"
CONTAINER_NAME="${CONTAINER_NAME:-ai-image-lab-l4t}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
HF_TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_TOKEN_ARGS=(-e "HF_TOKEN=$HF_TOKEN")
fi

# docker compose -f "$PROJECT_ROOT/docker/docker-compose.l4t.yml" up --build -d

docker build \
    -f "$PROJECT_ROOT/docker/Dockerfile.l4t" \
    -t "$IMAGE_NAME" \
    "$PROJECT_ROOT/docker"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    --ipc=host \
    --shm-size=8g \
    -p "$JUPYTER_PORT:8888" \
    -p "$TENSORBOARD_PORT:6006" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e TENSORBOARD_LOGDIR=/workspace/ai-image-lab-work/output/folder_lora_train/logs \
    -e TENSORBOARD_PORT=6006 \
    "${HF_TOKEN_ARGS[@]}" \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE_NAME"

echo "Jupyter:     http://localhost:${JUPYTER_PORT}/lab"
echo "TensorBoard: http://localhost:${TENSORBOARD_PORT}/"
