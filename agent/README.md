# image-agent

LoRAを使用しないローカル画像生成エージェント。

Qwen3-VL でMarkdownと参考画像を分析し、FLUX.2（またはFLUX.3）で画像を生成、Qwen Criticで評価、Real-ESRGANでアップスケールします。

---

## ⚠️ 重要な制約

- **LoRAなし**: 顔・髪・衣装の完全な一貫性は保証されません。
- **外部API不使用**: 推論はすべてローカルで実行されます。
- **VRAM制約**: RTX 2080 (8 GB) を対象とした設定です。

FLUX.2 Klein 4Bの無量子化時の公式目安は約13GB VRAMです。本実装はRTX 2080向けに
transformerとtext encoderをbitsandbytes NF4 4bit化し、GPU割当を6500MiBまで、CPU側を
24GiBまでに制限します。これは速度を犠牲にする実験的な8GB経路であり、実機確認が完了する
までは動作保証ではありません。BF16、FP8、NVFP4はTuringでは使用しません。

---

## 実機確認状況

| 機能 | 状態 |
|------|------|
| Pythonパッケージ（mockテスト） | ✅ 確認済み |
| RTX 2080 Qwen 4bit | ❌ 実機未確認 |
| RTX 2080 FLUX.2 512x512 | ❌ 実機未確認 |
| RTX 2080 Real-ESRGAN | ❌ 実機未確認 |
| RTX 2080 フルパイプライン | ❌ 実機未確認 |
| FLUX.3 INT8 ConvRot | ❌ EXPERIMENTAL / 実機未確認 |
| Jetson L4T ビルド | ❌ 実機未確認 |

**実機で確認した結果はこのREADMEを更新してください。**

---

## 通常ローカル環境での起動

### 前提条件

- Docker（BuildKit有効）
- NVIDIA Driver 450+ (CUDAモード)
- `docker buildx` プラグイン

### CUDAモード（RTX 2080など）

```bash
bash ./agent/docker/run.sh cuda
```

起動後、ターミナルに表示されるURL（例: `http://127.0.0.1:8888/lab/tree/agent/image_agent.ipynb?token=...`）をブラウザで開いてください。

### L4T / Jetsonモード

```bash
bash ./agent/docker/run.sh l4t
```

L4Tベースイメージを変更する場合:

```bash
L4T_PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3-igpu bash ./agent/docker/run.sh l4t
```

### datasetの投入方法

**方法1: docker cp（推奨）**

```bash
# コンテナ起動後（約5秒待機してから自動的に実行されます）
docker cp ./dataset/. image-agent-cuda:/workspace/dataset/
```

`run.sh` はコンテナ起動後に `dataset/` を自動コピーします。

**方法2: Jupyterのファイルブラウザ**

1. Jupyter Labを開く
2. 左のファイルツリーで `dataset/` フォルダに移動
3. ファイルをドラッグ＆ドロップ

**方法3: Notebookのセル2**

`image_agent.ipynb` のセル2に FileUpload ウィジェットがあります。

### Jupyterへの接続方法

```
http://127.0.0.1:8888/lab/tree/agent/image_agent.ipynb?token=<表示されたtoken>
```

### containerとimageの削除

```bash
docker stop image-agent-cuda
docker rm image-agent-cuda
docker rmi image-agent:cuda
```

---

## Compose（補助経路）

```bash
# CUDAモード
docker compose -f agent/docker/compose.yml --profile cuda up --build

# L4Tモード
docker compose -f agent/docker/compose.yml --profile l4t up --build
```

---

## 借用GPUマシン（clone禁止）

> **この章はホストにリポジトリをcloneできない環境向けです。**
>
> Docker remote Git build contextを使用します。
> Tailscaleが起動されますが、Tailscale control plane側には一時ノード情報や接続記録が残る可能性があります。

### 前提条件

- Docker（BuildKit + buildx）
- インターネット接続（GitHub + Tailscaleへの接続が必要）
- Tailscaleアカウント（ブラウザでの認証が必要）

### 一括コマンド（1回コピペで実行）

以下のブロックをターミナルにそのまま貼り付けて実行してください。

`COMMIT` は、remote buildで取得するリポジトリの版を固定する40文字のGit commit SHAです。
`main`のような移動するブランチ名ではなくSHAを指定すると、後日も同じソースからbuildでき、
意図しない更新の取り込みを防げます。下記は本エージェント実装を収録したcommitに固定して
います。新しい実装を使う場合だけ、GitHubのcommit画面に表示される新しい完全SHAへ更新して
ください。

**公開リポジトリの場合:**

```bash
set -euo pipefail

# 公開リポジトリと検証済みソース版
REPO_URL="https://github.com/mayjet/ai-image-lab.git"
COMMIT="6497cff5f94266f77ec0284f0151c592cb31e786"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"

# セッション固有の名前
SESSION_ID="ia-$(date +%Y%m%d%H%M%S)-$(head -c4 /dev/urandom | xxd -p)"
IMAGE="${SESSION_ID}:local"
BUILDER="${SESSION_ID}-builder"

echo "=== Session: $SESSION_ID ==="

# 一時builderを作成
docker buildx create --name "$BUILDER" --use

cleanup() {
    echo "=== Cleanup ==="
    docker stop "$SESSION_ID" 2>/dev/null || true
    docker rm "$SESSION_ID" 2>/dev/null || true
    docker rmi "$IMAGE" 2>/dev/null || true
    docker buildx rm "$BUILDER" 2>/dev/null || true
    echo "=== Done ==="
}
trap cleanup EXIT INT TERM

# Remote Git contextからビルド
echo "=== Building from $REPO_URL @ $COMMIT ==="
docker buildx build \
  --builder "$BUILDER" \
  --pull \
  --no-cache \
  --load \
  --file agent/docker/Dockerfile.cuda \
  --tag "$IMAGE" \
  "${REPO_URL}#${COMMIT}"

echo "=== Starting container (Tailscale mode) ==="
echo "=== Open the Tailscale login URL shown below in your browser. ==="

docker run --rm -it \
  --name "$SESSION_ID" \
  --gpus device=0 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --tmpfs /tmp:size=512m \
  -p "127.0.0.1:${JUPYTER_PORT}:8888" \
  -e ENABLE_TAILSCALE=1 \
  -e TAILSCALE_TIMEOUT=120 \
  -e TAILSCALE_VERSION=1.90.6 \
  "$IMAGE"

# trapによりcleanupが自動実行されます
```

Docker imageにはTailscaleを含めません。`ENABLE_TAILSCALE=1`のときだけ、公式static
binaryをコンテナ内部の一時runtimeディレクトリへダウンロードして起動します。
Auth keyは使わず、`tailscale up`が表示する対話ログインURLを使用します。バイナリ、
state、ログはコンテナ削除と同時に消えます。固定検証する場合は`TAILSCALE_SHA256`も
環境変数で渡してください。

> **privateリポジトリの場合**: `REPO_URL` を `git+ssh://git@github.com/<owner>/<repo>` 形式にし、SSH agent forwardingを使用してください（`-e` フラグやBuild argにシークレットを入れないでください）。

### datasetの投入（借用GPUモード）

Tailscale認証後、JupyterのURLが表示されます。

1. ブラウザでJupyterを開く（Tailscale URL + token）
2. `dataset/` フォルダに移動
3. Markdownファイルと参考画像をアップロード
4. `image_agent.ipynb` を開いて実行

### 後片付け

コンテナを終了すると、trap により自動的に以下が削除されます:
- runtime container
- session image
- buildx builder

---

## ライセンス注意

| モデル | ライセンス |
|--------|-----------|
| Qwen3-VL-4B-Instruct | Apache-2.0 |
| FLUX.2-klein-4B | Apache-2.0 |
| FLUX.3 Image Klein | **FLUX Kommunity License** (Apache-2.0 ではありません) |
| Real-ESRGAN | BSD-3-Clause |

FLUX.3を使用する場合は https://bfl.ai でライセンスを確認してください。

---

## ディレクトリ構成

```text
agent/
├── image_agent.ipynb     # Jupyterインターフェース
├── README.md
├── IMPLEMENTATION_SPEC.md
├── prompt_template.md    # 入力Markdownのテンプレート
├── pyproject.toml
├── configs/              # agent.yaml, generation.yaml, models.yaml
├── src/image_agent/      # Pythonパッケージ
│   ├── schemas/          # Pydanticスキーマ
│   ├── assets/           # Markdownと画像の読み込み
│   ├── planner/          # Qwenプランナー
│   ├── generation/       # FLUX.2/3生成バックエンド
│   ├── critic/           # Qwen評価
│   ├── upscale/          # Real-ESRGAN
│   ├── orchestration/    # LangGraph StateGraph
│   └── runtime/          # GPU診断、ログ、subprocess
├── workflows/            # FLUX.2/3パラメータ参照
├── docker/               # Docker関連ファイル
└── tests/                # テスト
```

実行時のみ作成（Git管理外）:
```text
agent/runs/      # 生成run出力
agent/.runtime/  # モデルキャッシュ
```

入力dataset:
```text
./dataset/       # リポジトリルート（コンテナ内: /workspace/dataset/）
```

---

## テスト実行

```bash
cd agent
pip install -e ".[dev]"
pytest tests/ -v
```

GPU不要テストのみ:
```bash
pytest tests/ -v -k "not gpu"
```

---

## 技術情報（実装時確認）

詳細は `IMPLEMENTATION_SPEC.md` と `configs/models.yaml` を参照してください。

| 項目 | 値 |
|------|-----|
| Qwen model ID | `Qwen/Qwen3-VL-4B-Instruct` |
| Qwen revision | main（実機確認後にpinすること） |
| FLUX.2 model ID | `black-forest-labs/FLUX.2-klein-4B` |
| FLUX.2 revision | main（実機確認後にpinすること） |
| torch wheel | 2.9.1 / CUDA 12.8 |
| diffusers | 0.42.1（CUDA） |
| transformers | 5.1.0（CUDA） |
| bitsandbytes | 0.48以上 |
| langgraph | 0.4.8 |
| ComfyUI | 未確認（FLUX.3使用時に確認要） |
