#!/usr/bin/env bash
set -e

cd /workspace

mkdir -p ai-image-lab-work

if [ ! -d "ai-image-lab-work/sd-scripts" ]; then
    git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git ai-image-lab-work/sd-scripts
fi

if [ ! -d "ai-image-lab-work/IP-Adapter" ]; then
    git clone --depth 1 https://github.com/tencent-ailab/IP-Adapter.git ai-image-lab-work/IP-Adapter
fi

python3 -m ipykernel install --sys-prefix --name ai-image-lab --display-name "Python (ai-image-lab)"

exec "$@"
