#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-ai-image-lab:cuda}"
CONTAINER_NAME="${CONTAINER_NAME:-ai-image-lab-cuda}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"

# docker compose -f "$PROJECT_ROOT/docker/docker-compose.yml" up --build -d

docker build \
    -f "$PROJECT_ROOT/docker/Dockerfile" \
    -t "$IMAGE_NAME" \
    "$PROJECT_ROOT/docker"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "$JUPYTER_PORT:8888" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE_NAME"
