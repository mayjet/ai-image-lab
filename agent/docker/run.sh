#!/usr/bin/env bash
# image-agent run script
#
# Usage:
#   bash ./agent/docker/run.sh cuda   — build and run CUDA image
#   bash ./agent/docker/run.sh l4t    — build and run L4T/Jetson image
#
# After the container starts:
#   Open the URL printed to the terminal to access Jupyter Lab.
#   Then open: agent/image_agent.ipynb
#
# To copy dataset into the running container (alternative to Jupyter upload):
#   docker cp ./dataset/ image-agent-cuda:/workspace/dataset/
#
# Compose examples (auxiliary path, not primary):
#   # docker compose -f agent/docker/compose.yml --profile cuda up --build
#   # docker compose -f agent/docker/compose.yml --profile l4t up --build

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
platform="${1:-cuda}"
shift || true

case "$platform" in
    cuda)
        dockerfile="$project_root/agent/docker/Dockerfile.cuda"
        dockerignore="$project_root/agent/docker/Dockerfile.cuda.dockerignore"
        image_name="image-agent:cuda"
        container_name="image-agent-cuda"
        gpu_args=(--gpus device=0)
        ;;
    l4t)
        dockerfile="$project_root/agent/docker/Dockerfile.l4t"
        dockerignore="$project_root/agent/docker/Dockerfile.l4t.dockerignore"
        image_name="image-agent:l4t"
        container_name="image-agent-l4t"
        gpu_args=(--runtime nvidia)
        echo "NOTE: L4T target — FLUX.2/3 inference on Jetson is not guaranteed." >&2
        ;;
    *)
        echo "Usage: bash ./agent/docker/run.sh [cuda|l4t]" >&2
        exit 2
        ;;
esac

JUPYTER_PORT="${JUPYTER_PORT:-8888}"
HF_TOKEN_ARG=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_TOKEN_ARG=(-e "HF_TOKEN=${HF_TOKEN}")
fi

# L4T base image override
L4T_BUILD_ARG=()
if [[ "$platform" == "l4t" && -n "${L4T_PYTORCH_IMAGE:-}" ]]; then
    L4T_BUILD_ARG=(--build-arg "L4T_PYTORCH_IMAGE=${L4T_PYTORCH_IMAGE}")
fi

echo "[run.sh] Building $image_name from $dockerfile ..."
DOCKER_BUILDKIT=1 docker buildx build \
    --file "$dockerfile" \
    "${L4T_BUILD_ARG[@]}" \
    --load \
    --tag "$image_name" \
    "$project_root"

# Stop any existing container with the same name
docker stop "$container_name" 2>/dev/null || true
docker rm "$container_name" 2>/dev/null || true

echo "[run.sh] Starting container $container_name ..."
echo "[run.sh] Jupyter will be available at http://127.0.0.1:${JUPYTER_PORT}"
echo ""

# Copy dataset into the running container after startup
# (run in background so we can wait for container creation)
_copy_dataset() {
    sleep 5
    if [[ -d "$project_root/dataset" ]]; then
        echo "[run.sh] Copying dataset/ into container..."
        docker exec "$container_name" mkdir -p /workspace/dataset 2>/dev/null || true
        docker cp "$project_root/dataset/." "$container_name:/workspace/dataset/" && \
            echo "[run.sh] dataset/ copied." || \
            echo "[run.sh] WARNING: dataset copy failed (container may not be ready yet)."
    fi
}
_copy_dataset &
copy_pid=$!

# Run the container
# --rm           : remove on exit
# --cap-drop ALL : minimal capabilities
# No bind mounts. No Docker volumes. No shared IPC. No Docker socket.
exec docker run \
    --rm \
    --name "$container_name" \
    "${gpu_args[@]}" \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --tmpfs /tmp:size=512m \
    -p "127.0.0.1:${JUPYTER_PORT}:8888" \
    -e "JUPYTER_PORT=8888" \
    "${HF_TOKEN_ARG[@]}" \
    "$image_name"

wait "$copy_pid" 2>/dev/null || true
