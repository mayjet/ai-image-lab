#!/usr/bin/env bash
# image-agent container entrypoint
#
# Normal mode (run.sh cuda / run.sh l4t):
#   - Tailscale is NOT started. Jupyter on localhost.
#
# Borrowed GPU mode (ENABLE_TAILSCALE=1):
#   - temporary official static binaries are downloaded inside the container
#   - tailscaled starts in userspace networking mode (no TUN, no NET_ADMIN)
#   - Interactive Tailscale login URL printed to terminal
#   - Tailscale Serve exposes Jupyter over HTTPS on the tailnet
#   - binaries and in-memory state disappear with the runtime container

set -euo pipefail

RUNTIME_DIR="${RUNTIME_DIR:-/workspace/agent/.runtime}"
LOG_DIR="$RUNTIME_DIR/logs"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
ENABLE_TAILSCALE="${ENABLE_TAILSCALE:-0}"
ENABLE_TAILSCALE_SSH="${ENABLE_TAILSCALE_SSH:-0}"
TAILSCALE_TIMEOUT="${TAILSCALE_TIMEOUT:-600}"
TAILSCALE_VERSION="${TAILSCALE_VERSION:-1.90.6}"
TAILSCALE_HOSTNAME="${TAILSCALE_HOSTNAME:-image-agent}"
SESSION_INFO="$RUNTIME_DIR/session-info.txt"

mkdir -p \
    "$RUNTIME_DIR/models" \
    "$RUNTIME_DIR/cache/huggingface" \
    "$RUNTIME_DIR/cache/transformers" \
    "$RUNTIME_DIR/cache/torch" \
    "$RUNTIME_DIR/cache/xdg" \
    "$RUNTIME_DIR/cache/tmp" \
    "$RUNTIME_DIR/offload" \
    "$LOG_DIR" \
    /workspace/dataset \
    /workspace/agent/runs

# ---------------------------------------------------------------------------
# Generate a random Jupyter token (always required — no token-less launch)
# ---------------------------------------------------------------------------
JUPYTER_TOKEN="${JUPYTER_TOKEN:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}"
export JUPYTER_TOKEN

# ---------------------------------------------------------------------------
# Tailscale — borrowed GPU mode only
# ---------------------------------------------------------------------------
tailscaled_pid=""
tailscale_enabled=0
tailscale_bin=""
tailscaled_bin=""

install_runtime_tailscale() {
    local machine_arch archive_arch install_dir archive url
    machine_arch="$(uname -m)"
    case "$machine_arch" in
        x86_64|amd64) archive_arch="amd64" ;;
        aarch64|arm64) archive_arch="arm64" ;;
        *)
            echo "[entrypoint] ERROR: Unsupported Tailscale architecture: $machine_arch" >&2
            return 1
            ;;
    esac

    install_dir="$RUNTIME_DIR/tailscale"
    archive="/tmp/tailscale-${TAILSCALE_VERSION}-${archive_arch}.tgz"
    url="https://pkgs.tailscale.com/stable/tailscale_${TAILSCALE_VERSION}_${archive_arch}.tgz"
    mkdir -p "$install_dir"
    echo "[entrypoint] Downloading temporary Tailscale ${TAILSCALE_VERSION} (${archive_arch})..."
    curl --fail --silent --show-error --location "$url" --output "$archive"
    if [[ -n "${TAILSCALE_SHA256:-}" ]]; then
        echo "${TAILSCALE_SHA256}  ${archive}" | sha256sum --check --status
    fi
    tar -xzf "$archive" -C "$install_dir" --strip-components=1
    rm -f "$archive"
    tailscale_bin="$install_dir/tailscale"
    tailscaled_bin="$install_dir/tailscaled"
    export PATH="$install_dir:$PATH"
    [[ -x "$tailscale_bin" && -x "$tailscaled_bin" ]]
}

if [[ "$ENABLE_TAILSCALE" == "1" ]]; then
    if ! install_runtime_tailscale; then
        echo "[entrypoint] ERROR: Temporary Tailscale installation failed." >&2
        exit 1
    fi

    echo "[entrypoint] Starting tailscaled (userspace networking, in-memory state)..."
    "$tailscaled_bin" \
        --tun=userspace-networking \
        --state=mem: \
        --socket=/tmp/tailscaled.sock \
        --port=41641 \
        >> "$LOG_DIR/tailscale.log" 2>&1 &
    tailscaled_pid=$!

    sleep 2

    echo "[entrypoint] Running tailscale up — open the URL in your browser to authenticate:"
    echo "----------------------------------------------------------------------"
    # --timeout ensures we don't wait forever
    tailscale_up_args=(
        --socket=/tmp/tailscaled.sock up
        --timeout="${TAILSCALE_TIMEOUT}s"
        --hostname="$TAILSCALE_HOSTNAME"
        --accept-routes
    )
    if [[ "$ENABLE_TAILSCALE_SSH" == "1" ]]; then
        # Tailscale SSH is provided by tailscaled itself. No sshd, host keys,
        # host port 22, or bind mount is required.
        tailscale_up_args+=(--ssh)
    fi
    if ! "$tailscale_bin" "${tailscale_up_args[@]}" 2>&1 | tee /dev/stderr; then
        echo "[entrypoint] ERROR: Tailscale authentication failed or timed out after ${TAILSCALE_TIMEOUT}s."
        kill "$tailscaled_pid" 2>/dev/null || true
        exit 1
    fi
    echo "----------------------------------------------------------------------"

    # Configure Tailscale Serve to proxy Jupyter
    echo "[entrypoint] Setting up Tailscale Serve for Jupyter on port $JUPYTER_PORT..."
    if ! "$tailscale_bin" --socket=/tmp/tailscaled.sock serve --bg "$JUPYTER_PORT"; then
        echo "[entrypoint] ERROR: Tailscale Serve setup failed." >&2
        kill "$tailscaled_pid" 2>/dev/null || true
        exit 1
    fi

    TAILSCALE_URL="$("$tailscale_bin" --socket=/tmp/tailscaled.sock status --json \
        | python3 -c 'import json,sys; s=json.load(sys.stdin); \
          print(s.get("Self",{}).get("DNSName","unknown").rstrip("."))' \
        2>/dev/null || echo 'unknown')"

    echo ""
    echo "========================================================================"
    echo "  Tailscale ACTIVE"
    echo "  Jupyter URL (Tailscale): https://${TAILSCALE_URL}/lab/tree/agent/image_agent.ipynb?token=${JUPYTER_TOKEN}"
    echo "  Jupyter token: $JUPYTER_TOKEN"
    if [[ "$ENABLE_TAILSCALE_SSH" == "1" ]]; then
        echo "  Tailscale SSH: ssh root@${TAILSCALE_URL}"
    fi
    echo "  Open: /workspace/agent/image_agent.ipynb"
    echo "========================================================================"
    tailscale_enabled=1
fi

# ---------------------------------------------------------------------------
# Install Jupyter kernel for image-agent
# ---------------------------------------------------------------------------
python3 -m ipykernel install --sys-prefix \
    --name "image-agent" \
    --display-name "Python (image-agent)" >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# Start Jupyter Lab
# ---------------------------------------------------------------------------
CMD="${1:-jupyter}"

if [[ "$CMD" == "jupyter" ]]; then
    if [[ "$tailscale_enabled" == "0" ]]; then
        JUPYTER_BIND="0.0.0.0"
        echo ""
        echo "========================================================================"
        echo "  Jupyter Lab starting on localhost"
        echo "  URL: http://127.0.0.1:${JUPYTER_PORT}/lab/tree/agent/image_agent.ipynb?token=${JUPYTER_TOKEN}"
        echo "  Token: $JUPYTER_TOKEN"
        echo "========================================================================"
    else
        JUPYTER_BIND="0.0.0.0"
    fi

    # Persist connection details inside the disposable container. This lets a
    # detached launcher retrieve the URL/token without attaching to PID 1.
    {
        printf 'JUPYTER_TOKEN=%s\n' "$JUPYTER_TOKEN"
        printf 'JUPYTER_LOCAL_URL=%s\n' "http://127.0.0.1:${JUPYTER_PORT}/lab/tree/agent/image_agent.ipynb?token=${JUPYTER_TOKEN}"
        if [[ "$tailscale_enabled" == "1" ]]; then
            printf 'TAILSCALE_DNS=%s\n' "$TAILSCALE_URL"
            printf 'JUPYTER_TAILSCALE_URL=%s\n' "https://${TAILSCALE_URL}/lab/tree/agent/image_agent.ipynb?token=${JUPYTER_TOKEN}"
            if [[ "$ENABLE_TAILSCALE_SSH" == "1" ]]; then
                printf 'TAILSCALE_SSH=%s\n' "ssh root@${TAILSCALE_URL}"
            fi
        fi
    } > "$SESSION_INFO"
    chmod 600 "$SESSION_INFO"

    jupyter lab \
        --ip="$JUPYTER_BIND" \
        --port="$JUPYTER_PORT" \
        --no-browser \
        --allow-root \
        --IdentityProvider.token="$JUPYTER_TOKEN" \
        --ServerApp.password="" \
        --ServerApp.root_dir=/workspace \
        --ServerApp.default_url="/lab/tree/agent/image_agent.ipynb" \
        &
    jupyter_pid=$!
else
    exec "$@"
fi

# ---------------------------------------------------------------------------
# Shutdown handler
# ---------------------------------------------------------------------------
shutdown() {
    echo "[entrypoint] Shutting down..."
    kill "$jupyter_pid" 2>/dev/null || true
    if [[ -n "$tailscaled_pid" ]]; then
        # Attempt graceful Tailscale logout
        "$tailscale_bin" --socket=/tmp/tailscaled.sock logout 2>/dev/null || true
        kill "$tailscaled_pid" 2>/dev/null || true
        wait "$tailscaled_pid" 2>/dev/null || true
    fi
    wait "$jupyter_pid" 2>/dev/null || true
}
trap shutdown INT TERM EXIT

set +e
wait "$jupyter_pid"
exit_status=$?
set -e
exit "$exit_status"
