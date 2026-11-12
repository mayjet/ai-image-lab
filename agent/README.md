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

### 既存containerを全削除して、detachedで再構築

次のブロックは借用GPUホスト上の**全containerを強制削除**します。他用途のcontainerが
存在しないことを `docker ps -a` で再確認してから、一度に貼り付けてください。remote Git
contextの`main`最新版をbuildし、専用build cacheを削除した後、runtime containerを
detached (`-d`) で起動します。ターミナルを閉じてもcontainerとJupyterは停止しません。

```bash
start_image_agent_borrowed() (
    # Subshell内で失敗を処理し、呼出元の対話terminalを終了させない。
    REPO_URL="https://github.com/mayjet/ai-image-lab.git"
    REPO_REF="main"
    IMAGE="image-agent-borrowed:latest"
    CONTAINER="image-agent-borrowed"
    BUILDER="image-agent-borrowed-builder"
    JUPYTER_PORT="${JUPYTER_PORT:-8888}"
    TS_HOSTNAME="image-agent-$(date +%Y%m%d-%H%M%S)"

    echo "=== WARNING: removing every Docker container on this host ==="
    docker ps -a
    docker ps -aq | xargs -r docker rm -f || return $?

    # このlauncher専用の以前のimage/builderだけを削除する。
    docker buildx rm "$BUILDER" 2>/dev/null || true
    docker image rm "$IMAGE" 2>/dev/null || true
    docker buildx create --name "$BUILDER" --driver docker-container --use || return $?

    echo "=== Building $REPO_URL @ $REPO_REF ==="
    docker buildx build \
      --builder "$BUILDER" \
      --pull \
      --no-cache \
      --load \
      --file agent/docker/Dockerfile.cuda \
      --tag "$IMAGE" \
      "${REPO_URL}#${REPO_REF}"
    BUILD_STATUS=$?
    if [[ "$BUILD_STATUS" -ne 0 ]]; then
        docker buildx rm "$BUILDER" 2>/dev/null || true
        return "$BUILD_STATUS"
    fi
    docker buildx rm "$BUILDER" 2>/dev/null || true

    echo "=== Starting persistent detached container ==="
    docker run -d \
      --name "$CONTAINER" \
      --gpus device=0 \
      --restart=no \
      --cap-drop ALL \
      --cap-add SETUID \
      --cap-add SETGID \
      --security-opt no-new-privileges \
      --tmpfs /tmp:size=1g \
      --shm-size=1g \
      -p "127.0.0.1:${JUPYTER_PORT}:8888" \
      -e ENABLE_TAILSCALE=1 \
      -e ENABLE_TAILSCALE_SSH=1 \
      -e TAILSCALE_HOSTNAME="$TS_HOSTNAME" \
      -e TAILSCALE_TIMEOUT=600 \
      -e TAILSCALE_VERSION=1.90.6 \
      "$IMAGE" || return $?

    echo "=== Container started: $CONTAINER ==="
    echo "Run: docker logs -f $CONTAINER"
)

if start_image_agent_borrowed; then
    START_STATUS=0
else
    START_STATUS=$?
fi
unset -f start_image_agent_borrowed
echo "=== Launcher status: ${START_STATUS}; terminal remains open ==="
```

Docker image自体にはTailscaleを含めません。`ENABLE_TAILSCALE=1`のruntime containerだけが
公式static binaryを一時ダウンロードします。Auth key、host側Tailscale、bind mount、Docker
volumeは使用しません。`CAP_SETUID`と`CAP_SETGID`はTailscale SSHが接続ユーザーのprocessを
開始するためだけに戻しています。`NET_ADMIN`およびTUN deviceは不要です。

### Tailscale認証と起動確認

```bash
docker logs -f image-agent-borrowed
```

表示されたTailscaleログインURLをブラウザで開きます。認証後にJupyterが起動します。
ログ追跡は `Ctrl-C` で終了して構いません。これは `docker attach` ではないため、containerや
Jupyterは停止しません。接続情報は後から何度でも取得できます。

```bash
docker exec image-agent-borrowed cat /workspace/agent/.runtime/session-info.txt
docker exec image-agent-borrowed check-image-agent-environment
docker exec image-agent-borrowed nvidia-smi
docker ps --filter name=image-agent-borrowed
```

`session-info.txt` の `JUPYTER_TAILSCALE_URL` をブラウザで開くか、表示された
`JUPYTER_TOKEN`を使います。tokenをログやチャットへ貼り付けないでください。

### Tailscale SSH / VS Code / Cursor

接続元PCも同じtailnetへ参加させ、次を実行します。

```bash
ssh root@<session-info.txtのTAILSCALE_DNS>
```

Tailscaleの標準policyなら、自分が所有するnodeへの`root`接続はcheck modeで許可されます。
policyを変更済みの場合は、port 22へのnetwork accessと、宛先nodeへ`root`で接続できるSSH
ruleの両方が必要です。公開port 22、password、SSH key、container内`sshd`は不要です。

VS Code/CursorのRemote SSH設定例:

```sshconfig
Host image-agent-borrowed
    HostName <session-info.txtのTAILSCALE_DNS>
    User root
```

Remote SSHで接続後は `/workspace` を開きます。Tailscale SSHはSFTPを提供するため、Remote
SSHのファイル転送も使用できます。編集・download・model cache・VS Code serverはすべて
container filesystem内に置かれ、container削除時に消えます。

### 障害確認

```bash
# containerが終了した理由（OOMKilled=trueならhost kernelによるkill）
docker inspect image-agent-borrowed \
  --format 'status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} error={{.State.Error}}'

# entrypoint / Tailscale / Jupyterログ
docker logs --tail 300 image-agent-borrowed

# Qwen/FLUX workerの完全なstderr（進捗で例外が欠落しない）
docker exec image-agent-borrowed bash -lc \
  'for f in $(ls -1t /workspace/agent/.runtime/logs/*.stderr.log 2>/dev/null | head -3); do echo "===== $f"; cat "$f"; done'

# Tailscale接続状態
docker exec image-agent-borrowed \
  /workspace/agent/.runtime/tailscale/tailscale \
  --socket=/tmp/tailscaled.sock status

# SSH広告状態（RunSSH=trueでcontainer側は有効）
docker exec image-agent-borrowed tailscale set --ssh
docker exec image-agent-borrowed tailscale debug prefs | grep -E '"RunSSH"|WantRunning'
```

SSHだけ拒否される場合はcontainerの再buildではなくtailnet policyを確認します。Jupyter URLが
未生成なら、Tailscale認証待ちまたは600秒timeout後にcontainerが終了していないか
`docker logs`と`docker inspect`を確認します。Qwen/FLUXの初回実行はmodel downloadを含むため、
それぞれ30分/60分のtimeoutを設定しています。

`docker exec -it ... bash`で表示される`groups: cannot find name for group ID ...`は、host/NVIDIA
runtimeから渡された補助group IDに対応する名前がcontainerの`/etc/group`に無いという警告です。
Tailscale SSHの有効・無効とは関係ありません。

### datasetの投入（借用GPUモード）

Tailscale認証後、JupyterのURLが表示されます。

1. ブラウザでJupyterを開く（Tailscale URL + token）
2. `dataset/` フォルダに移動
3. Markdownファイルと参考画像をアップロード
4. `image_agent.ipynb` を開いて実行

### 停止・後片付け

```bash
docker rm -f image-agent-borrowed
docker image rm image-agent-borrowed:latest
docker buildx rm image-agent-borrowed-builder 2>/dev/null || true
```

container、model、dataset、生成物、Jupyter token、Tailscale binary/stateが削除されます。
必要な生成物は削除前にSFTP/Jupyterで回収してください。Tailscale control plane上の一時node
recordはhost filesystemではないため、このコマンドでは消えません。必要ならMachines画面から
削除してください。

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
