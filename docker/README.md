# ai-image-lab Docker

プロジェクトルートから実行します。

```bash
bash ./docker/run.sh
```

Jetson / L4T は以下です。

```bash
bash ./docker/run_l4t.sh
```

Jupyter Lab は既定で `http://localhost:8888` です。
コンテナ内の作業ディレクトリは `/workspace` で、ホストのプロジェクトルートをそのままマウントします。

## データセット

Notebookの既定値は `DATASET_ROOT = Path("./dataset")` です。
実データセット名を変えたくない場合はNotebook側で `DATASET_ROOT` を変更してください。

例:

```python
DATASET_ROOT = Path("./<dataset>")
```

またはホスト側で `dataset` というsymlinkを作っても構いません。

## x86 CUDA

`Dockerfile` は通常のx86 CUDAマシン向けです。
CUDA対応PyTorch、xformers、bitsandbytesはDockerfile側で入れます。

```bash
IMAGE_NAME=ai-image-lab:cuda CONTAINER_NAME=ai-image-lab-cuda JUPYTER_PORT=8888 bash ./docker/run.sh
```

## Jetson / L4T

`Dockerfile.l4t` はJetson向けです。
PyTorchはL4Tベースイメージに含まれるものを使います。
bitsandbytesとxformersは入れません。
`BASE_IMAGE=nvcr.io/nvidia/pytorch:24.09-py3-igpu` の想定はJetPack 6系です。
Pythonはベースイメージ側の `python3` を使います。r36/Ubuntu 22.04系なら通常Python 3.10系です。
Jupyter Lab、Notebook、ipykernelは `requirements.txt` から入れ、ビルド時と起動時に `Python (ai-image-lab)` kernelを登録します。

JetPackに合わない場合は `Dockerfile.l4t` の `ARG BASE_IMAGE=...` を変更してください。

## Compose

Compose設定はおまけです。
`run.sh` / `run_l4t.sh` の先頭付近にコメントとして残しています。
