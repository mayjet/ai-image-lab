#!/usr/bin/env bash
set -euo pipefail

# Local Python environment for notebooks and scripts. Apple Silicon/MPS is
# supported, but the setup is intentionally usable on other local machines.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

cd "$PROJECT_ROOT"

if ! command -v git >/dev/null 2>&1; then
    echo "git が必要です。先にインストールしてください。" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "uvの導入にcurlが必要です。先にcurlをインストールしてください。" >&2
        exit 1
    fi
    echo "uvをインストールします..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uvをPATHから見つけられません。シェルを再起動して再実行してください。" >&2
    exit 1
fi

if [[ "${RECREATE_VENV:-0}" == "1" && -d "$VENV_DIR" ]]; then
    echo "RECREATE_VENV=1: 既存環境を作り直します: $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Python $PYTHON_VERSION の仮想環境を作成します: $VENV_DIR"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
else
    echo "既存の仮想環境を更新します: $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

echo "Notebook・生成・学習依存をインストールします..."
uv pip install --python "$VENV_PYTHON" \
    "setuptools==80.9.0" \
    wheel \
    ipykernel \
    ipywidgets \
    "numpy==1.26.4" \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    tqdm \
    toml \
    pyyaml \
    requests \
    pillow \
    opencv-python \
    torch \
    torchvision \
    torchaudio \
    "transformers==4.44.0" \
    "diffusers[torch]==0.32.1" \
    "accelerate==1.6.0" \
    "huggingface-hub==0.34.3" \
    "safetensors==0.4.5" \
    datasets \
    peft \
    tokenizers \
    sentencepiece \
    "protobuf>=3.20.3,<6" \
    imagesize \
    ftfy \
    einops \
    voluptuous \
    "tensorboard==2.18.0" \
    rich \
    lion-pytorch \
    schedulefree \
    pytorch-optimizer \
    prodigyopt \
    prodigy-plus-schedule-free \
    open-clip-torch \
    controlnet_aux

mkdir -p "$PROJECT_ROOT/cache" "$PROJECT_ROOT/models/sd15" "$PROJECT_ROOT/outputs/sd15" "$PROJECT_ROOT/vendor"
if [[ ! -f "$PROJECT_ROOT/vendor/sd-scripts/train_network.py" ]]; then
    echo "kohya-ss/sd-scriptsを取得します..."
    git clone https://github.com/kohya-ss/sd-scripts.git \
        "$PROJECT_ROOT/vendor/sd-scripts"
fi

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
KERNEL_BASE="$(printf '%s' "$(basename "$PROJECT_ROOT")" | tr -cs '[:alnum:]_.-' '-')"
KERNEL_NAME="${KERNEL_BASE}_venv"
echo "ipykernelを登録します: Python ($KERNEL_NAME)"
"$VENV_PYTHON" -m ipykernel install --user \
    --name "$KERNEL_NAME" \
    --display-name "Python ($KERNEL_NAME)"

echo "環境を検査します..."
"$VENV_PYTHON" - <<'PY'
import platform
import cv2
import diffusers
import torch
import transformers
from controlnet_aux import LineartDetector  # noqa: F401

print(f"OS: {platform.system()} {platform.machine()}")
print(f"Python: {platform.python_version()}")
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS built: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
print(f"Diffusers: {diffusers.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"OpenCV: {cv2.__version__}")
print("sd15/generate_character.ipynb dependencies: OK")
PY

echo
echo "環境構築が完了しました。"
echo "Python interpreter: $VENV_PYTHON"
echo "VS Code / Jupyter kernel: Python ($KERNEL_NAME)"
echo "再構築する場合: RECREATE_VENV=1 bash ./uvvenv.sh"
