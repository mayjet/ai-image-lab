# ai-image-lab Docker

プロジェクトルートから実行します。

```bash
bash ./docker/run.sh
```

Jetson / L4T は以下です。

```bash
bash ./docker/run_l4t.sh
```

Jupyter Lab は既定で `http://localhost:8888`、TensorBoardは
`http://localhost:6006` です。どちらもコンテナ起動時に自動起動します。
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
PyTorchはL4Tベースイメージに含まれるものを使います（`requirements.txt` 側ではtorchを入れません）。
bitsandbytesとxformersは入れません。
ベースイメージは `FROM nvcr.io/nvidia/pytorch:24.12-py3-igpu` を直書きしており、JetPack 6.x / L4T r36系(CUDA 12.6)の想定です。
LoRA学習はベースイメージ側のPython 3.12を使います。
顔検出はJetson CUDA 12.6用ONNX Runtime GPUとの互換性を保つため、uvが構築する
`/opt/face-detector` のPython 3.10へ隔離します。
`python3 anime_face_collect.py` はスクリプト自身が専用環境へ切り替えるため、利用者が
専用Pythonのパスを指定する必要はありません。
Jupyter Lab、Notebook、ipykernelは `requirements.txt` から入れ、ビルド時と起動時に `Python (ai-image-lab)` kernelを登録します。

`requirements.txt` はx86とL4Tで共通の1ファイルです。
ソースビルドが必要でaarch64で失敗しやすく、かつNotebookからは未使用の復元系（`basicsr` / `facexlib` / `gfpgan` / `Real-ESRGAN`）は含めていません。

JetPackに合わない場合は `Dockerfile.l4t` の `FROM` 行を直接書き換えてください。

### 顔データ準備

GPU優先、利用不可時はCPUへ切り替えて全キャラクターを処理します。

```bash
docker exec -it -w /workspace ai-image-lab-l4t \
  python3 anime_face_collect.py
```

環境だけを検査する場合:

```bash
docker exec -it -w /workspace ai-image-lab-l4t \
  python3 anime_face_collect.py --check
```

### LoRA学習

```bash
docker exec -it -w /workspace ai-image-lab-l4t \
  python3 train_lora.py --train
```

### 環境診断

```bash
docker exec -it ai-image-lab-l4t check-l4t-environment
```

診断では学習用PyTorch/CUDA、顔検出ONNX Runtime、sd-scripts、Jupyter、TensorBoardを確認します。

### Hugging Face token

任意の`HF_TOKEN`をホスト環境から渡せます。Dockerfileやリポジトリにはtokenを書きません。

```bash
HF_TOKEN="..." bash ./docker/run_l4t.sh
```

### TensorBoardログ

既定の監視先は以下です。

```text
/workspace/ai-image-lab-work/output/folder_lora_train/logs
```

## Compose

Compose設定はおまけです。
`run.sh` / `run_l4t.sh` の先頭付近にコメントとして残しています。
