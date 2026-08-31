#!/usr/bin/env bash
set -euo pipefail

# x86 NVIDIA CUDA runner. The container stays in the foreground and is removed
# when it exits. Use docker/run_l4t.sh on Jetson.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="${1:-sd15}"

case "$backend" in
    sd15)
        dockerfile="$project_root/docker/Dockerfile.sd15.cuda"
        image_name="ai-image-lab:sd15-cuda"
        container_name="ai-image-lab-sd15-cuda"
        ;;
    flux2)
        dockerfile="$project_root/docker/Dockerfile.flux2.cuda"
        image_name="ai-image-lab:flux2-cuda"
        container_name="ai-image-lab-flux2-cuda"
        ;;
    *)
        echo "Usage: bash ./docker/run.sh [sd15|flux2]" >&2
        exit 2
        ;;
esac

if [[ "$(uname -m)" == "aarch64" ]] || [[ -f /etc/nv_tegra_release ]]; then
    echo "Jetson/L4Tでは bash ./docker/run_l4t.sh $backend を使用してください。" >&2
    exit 2
fi

jupyter_port="${JUPYTER_PORT:-8888}"
tensorboard_port="${TENSORBOARD_PORT:-6006}"
hf_token_args=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    hf_token_args=(-e "HF_TOKEN=$HF_TOKEN")
fi

# Compose alternative (documentation only; this script does not execute it):
# docker compose -f "$project_root/docker/docker-compose.yml" build "${backend}-cuda"
# docker compose -f "$project_root/docker/docker-compose.yml" run --rm --service-ports "${backend}-cuda"

docker build \
    -f "$dockerfile" \
    -t "$image_name" \
    "$project_root"

echo "Jupyter:     http://localhost:${jupyter_port}/lab"
echo "TensorBoard: http://localhost:${tensorboard_port}/"
echo "Stop:        Ctrl+C"

exec docker run --rm -it \
    --name "$container_name" \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "$jupyter_port:8888" \
    -p "$tensorboard_port:6006" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e "ENABLE_TENSORBOARD=${ENABLE_TENSORBOARD:-1}" \
    -e "TENSORBOARD_PORT=6006" \
    "${hf_token_args[@]}" \
    -v "$project_root:/workspace" \
    -w /workspace \
    "$image_name"
