#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-ai-image-lab:l4t}"
CONTAINER_NAME="${CONTAINER_NAME:-ai-image-lab-l4t}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3}"

# docker compose -f "$PROJECT_ROOT/docker/docker-compose.l4t.yml" up --build -d

docker build \
    -f "$PROJECT_ROOT/docker/Dockerfile.l4t" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t "$IMAGE_NAME" \
    "$PROJECT_ROOT/docker"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    --ipc=host \
    --shm-size=8g \
    -p "$JUPYTER_PORT:8888" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE_NAME"
