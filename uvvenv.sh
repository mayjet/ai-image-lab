#!/bin/zsh
set -e

# M2 Mac local environment for ai-image-lab notebooks.
# Creates ./.venv in the current directory.

PYTHON_VERSION="3.11"
VENV_DIR="./.venv"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv がインストールされていません。pipx 経由でインストールします..."
    if ! command -v pipx >/dev/null 2>&1; then
        echo "pipx がありません。先に pipx をインストールしてください。"
        echo "参考: https://pypa.github.io/pipx/installation/"
        exit 1
    fi
    pipx install uv
fi

if [ -d "$VENV_DIR" ]; then
    echo "既存の仮想環境を削除します: $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

echo "仮想環境を作成します: $VENV_DIR"
uv venv --python "python$PYTHON_VERSION" "$VENV_DIR"

source "$VENV_DIR/bin/activate"

echo "基本ツールをインストールします..."
uv pip install --upgrade pip setuptools wheel

echo "Jupyter 実行環境をインストールします..."
uv pip install \
    jupyter \
    jupyterlab \
    notebook \
    ipykernel \
    ipywidgets

echo "Notebook 補助ライブラリをインストールします..."
uv pip install \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    tqdm \
    toml \
    pyyaml

echo "PyTorch for Apple Silicon / MPS をインストールします..."
uv pip install \
    torch \
    torchvision \
    torchaudio

echo "生成・学習関連ライブラリをインストールします..."
uv pip install \
    "transformers==4.54.1" \
    "diffusers[torch]==0.32.1" \
    "accelerate==1.6.0" \
    "huggingface-hub==0.34.3" \
    "safetensors==0.4.5" \
    "fsspec==2026.6.0" \
    datasets \
    peft \
    tokenizers \
    sentencepiece \
    protobuf

echo "sd-scripts の学習依存ライブラリをインストールします..."
uv pip install \
    imagesize \
    ftfy \
    einops \
    voluptuous \
    tensorboard \
    rich \
    lion-pytorch \
    schedulefree \
    pytorch-optimizer \
    prodigyopt \
    prodigy-plus-schedule-free \
    open-clip-torch

echo "画像処理・アップスケール関連ライブラリをインストールします..."
uv pip install \
    pillow \
    opencv-python \
    onnxruntime \
    controlnet_aux \
    basicsr \
    facexlib \
    gfpgan

echo "Real-ESRGAN をインストールします..."
uv pip install "git+https://github.com/xinntao/Real-ESRGAN.git"

# bitsandbytes と xformers は CUDA 前提のため M2 Mac では入れません。
# Notebook 側で AdamW8bit を使う場合は、M2 Mac では AdamW などへ変更してください。

if [ ! -d "./ai-image-lab-work/sd-scripts" ]; then
    echo "kohya-ss/sd-scripts を取得します..."
    mkdir -p ./ai-image-lab-work
    git clone https://github.com/kohya-ss/sd-scripts.git ./ai-image-lab-work/sd-scripts
fi

if [ ! -d "./ai-image-lab-work/IP-Adapter" ]; then
    echo "IP-Adapter を取得します..."
    mkdir -p ./ai-image-lab-work
    git clone https://github.com/tencent-ailab/IP-Adapter.git ./ai-image-lab-work/IP-Adapter
fi

KERNEL_NAME="${PWD##*/}_venv"
echo "Jupyter kernel を登録します: $KERNEL_NAME"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)"

echo "環境構築が完了しました。"
echo "仮想環境を有効化: source $VENV_DIR/bin/activate"
echo "Jupyter 起動: jupyter lab"
echo "Notebook kernel: Python ($KERNEL_NAME)"
