#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
platform="${1:-cuda}"

case "$platform" in
    cuda)
        dockerfile="$project_root/sd15/docker/Dockerfile.cuda"
        image_name="ai-image-lab:sd15-cuda"
        container_name="ai-image-lab-sd15-cuda"
        gpu_args=(--gpus all)
        shm_size="16g"
        ;;
    l4t)
        dockerfile="$project_root/sd15/docker/Dockerfile.l4t"
        image_name="ai-image-lab:sd15-l4t"
        container_name="ai-image-lab-sd15-l4t"
        gpu_args=(--runtime nvidia)
        shm_size="8g"
        ;;
    *)
        echo "Usage: bash ./sd15/docker/run.sh [cuda|l4t]" >&2
        exit 2
        ;;
esac

build_args=()
if [[ "$platform" == "l4t" ]]; then
    build_args=(--build-arg "L4T_PYTORCH_IMAGE=${L4T_PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:24.12-py3-igpu}")
fi

hf_token_args=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    hf_token_args=(-e "HF_TOKEN=$HF_TOKEN")
fi

docker build "${build_args[@]}" -f "$dockerfile" -t "$image_name" "$project_root"

exec docker run --rm -it \
    --name "$container_name" \
    "${gpu_args[@]}" \
    --ipc=host \
    --shm-size="$shm_size" \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "${JUPYTER_PORT:-8888}:8888" \
    -p "${TENSORBOARD_PORT:-6006}:6006" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e "ENABLE_TENSORBOARD=${ENABLE_TENSORBOARD:-1}" \
    -e TENSORBOARD_PORT=6006 \
    "${hf_token_args[@]}" \
    -v "$project_root:/ai-image-lab" \
    -w /ai-image-lab/sd15 \
    "$image_name"
