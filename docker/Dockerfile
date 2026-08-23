FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/ai-image-lab-work/.cache/huggingface
ENV XDG_CACHE_HOME=/workspace/ai-image-lab-work/.cache

WORKDIR /tmp/build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel uv

RUN uv pip install --system \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1

RUN uv pip install --system \
    xformers==0.0.28.post1 \
    bitsandbytes==0.43.3

COPY requirements.txt /tmp/build/requirements.txt
RUN uv pip install --system -r /tmp/build/requirements.txt

COPY entrypoint.sh /usr/local/bin/ai-image-lab-entrypoint
RUN chmod +x /usr/local/bin/ai-image-lab-entrypoint

WORKDIR /workspace
EXPOSE 8888

ENTRYPOINT ["/usr/local/bin/ai-image-lab-entrypoint"]
CMD ["python3", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
