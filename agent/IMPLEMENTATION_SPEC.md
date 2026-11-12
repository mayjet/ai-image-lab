# ローカル画像生成エージェント実装仕様

この文書は、`agent/` 配下に実装するLoRAなし画像生成エージェントの唯一の基準仕様である。
実装者は既存コード、README、過去のLoRA実装よりも本書を優先すること。本書に明記されて
いない機能を推測で「実装済み」としないこと。

## 1. 目的

自由形式のMarkdownと複数の参考画像を入力として、完全ローカルのQwen VLMが生成計画を
作成し、必要なら補助画像を生成した後、FLUX.2またはFLUX.3で本画像を生成する。生成後は
Qwen VLMが元指示と参考画像に照らして評価し、有限回だけ修正を試みる。受理された画像は
Real-ESRGANを基本とするアップスケーラーでFull HD相当以上へ拡大する。

Jupyter Notebookをユーザーインターフェースにするが、モデルロード、推論、状態遷移、
ファイル管理などの実処理は再利用可能なPythonパッケージへ分離する。

## 2. 絶対要件

- LoRAの学習、読み込み、選択、適用を実装しない。
- ファインチューニングを実装しない。
- 推論時に外部生成API、外部VLM API、外部モデレーションAPIを使用しない。
- モデルと依存パッケージのダウンロード以外の推論処理はローカルで完結させる。
- 対象x86環境はNVIDIA GeForce RTX 2080、VRAM 8GB、CPU RAM 32GBである。
- RTX 2080ではFP16を使用し、BF16、FP8、NVFP4を前提にしない。
- QwenとFLUXを同時にGPUへ常駐させない。
- モデル推論はNotebook kernelではなく子プロセスで行い、プロセス終了をVRAM解放境界にする。
- `torch.cuda.empty_cache()`だけをVRAM解放手段にしない。
- Notebook内へ大量の処理ロジックを書かない。
- Qwenへ任意のshell、Python、ファイル操作を実行させない。
- 自由なReActループや無制限のtool callingを使用しない。
- ループ回数、分岐、終了条件、利用可能な処理はPython側で制御する。
- ユーザーが与えたMarkdownを書き換えない。
- ユーザー提供資料とAIが推測・生成した資料を明確に区別する。
- FLUX.1向けControlNetやAdapterをFLUX.2/3へ無条件に流用しない。
- 実機未検証の機能を動作確認済みと記載しない。
- privateな画像、Markdown、出力、モデルキャッシュをGitまたはDockerイメージへ含めない。

## 3. 対象外

- LoRA、DreamBooth、Textual Inversionなどの追加学習。
- 複数GPUや分散推論。
- 外部公開、共有、投稿、クラウド保存。
- 常設Webサービス。
- モデルが自由にコードを書くCode Agent。
- 生成結果が特定キャラクターと完全一致することの保証。

LoRAなしでは、顔、髪、衣装、小物、画風の厳密な一貫性に限界がある。この制約はREADMEと
Notebookに明記すること。

### 3.1 書き込み範囲

実装者が新規作成・編集してよい実装ファイルは `agent/` 配下だけである。既存の
`flux2/`、`sd15/`、`shared/`、ルート`README.md`、その他の既存機能を変更しない。

ルート`.gitignore`にはdeny-by-default設定から`agent/`を追跡可能にする次の例外だけが
必要であり、すでに追加済みである。

```gitignore
!/agent/
!/agent/**
```

ルート`.dockerignore`は変更しない。`agent/docker/Dockerfile.cuda.dockerignore`と
`agent/docker/Dockerfile.l4t.dockerignore`を作り、Dockerfile別のbuild context制御を行う。
Docker imageへは`agent/`の実装ファイルだけをコピーし、ルート`dataset/`、既存backend、
Git metadata、生成物、cacheを含めない。

実行時に読み取る入力はリポジトリ直下の `./dataset` である。入力Markdownと参考画像は
読み取り対象であり、元ファイルを書き換えない。通常ローカル実行でホストの`./dataset`を
利用する具体的方法は、コンテナへの明示的コピーまたはJupyter uploadとし、bind mountを
主要経路にしない。

## 4. ディレクトリ構成

最終的に次の構成にする。

```text
agent/
├── .gitignore
├── image_agent.ipynb
├── README.md
├── IMPLEMENTATION_SPEC.md
├── prompt_template.md
├── pyproject.toml
├── prompts/
├── configs/
│   ├── agent.yaml
│   ├── generation.yaml
│   └── models.yaml
├── src/image_agent/
│   ├── schemas/
│   ├── assets/
│   ├── planner/
│   ├── generation/
│   ├── critic/
│   ├── upscale/
│   ├── orchestration/
│   └── runtime/
├── workflows/
│   ├── flux2/
│   └── flux3/
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cuda.dockerignore
│   ├── Dockerfile.l4t
│   ├── Dockerfile.l4t.dockerignore
│   ├── requirements.txt
│   ├── entrypoint.sh
│   ├── check_environment.sh
│   ├── run.sh
│   └── compose.yml
└── tests/
```

実行時にのみ次を作成し、GitとDocker build contextから除外する。

```text
agent/runs/
agent/.runtime/
```

入力は既存のルートディレクトリを使用する。

```text
dataset/
```

`agent/`直下にNotebookを置くこと。初期実装でNotebookをサブフォルダへ分割しない。

## 5. コンテナ内パス

Dockerイメージ内ではagent実装を `/workspace/agent` に配置し、空の入力ディレクトリ
`/workspace/dataset`を作成する。Jupyterのroot directoryと基本`WORKDIR`は `/workspace` とし、
起動直後に `agent/image_agent.ipynb`を開けるURLまたは案内を表示する。

```text
/workspace/agent/image_agent.ipynb
/workspace/dataset/
/workspace/agent/runs/
/workspace/agent/.runtime/models/
/workspace/agent/.runtime/cache/
/workspace/agent/.runtime/offload/
/workspace/agent/.runtime/logs/
```

Hugging Face、Transformers、Diffusers、Torch、ComfyUI、アップスケーラーのキャッシュは
すべて `/workspace/agent/.runtime/` 以下へ向ける。ホームディレクトリやホストへ意図せず
保存しない。

## 6. Docker要件

### 6.1 共通

- runtime containerは1個だけにする。Jupyter、Qwen子プロセス、FLUX子プロセスは同じ
  runtime container内で動作させる。借用GPUモードだけTailscaleも同じcontainer内で動かす。
- bind mountとDocker volumeを使用しない。
- 入力はJupyterのファイル画面またはNotebook UIから `./dataset` へアップロードする。
- 出力はJupyterから明示的にダウンロードする。ダウンロードしなければコンテナ削除時に消える。
- モデルはコンテナ起動後に必要なものだけダウンロードする。
- モデルをDockerfileへ焼き込まない。
- モデルは毎回再ダウンロードしてよい。
- `--rm`、`--log-driver=none`、`--cap-drop=ALL`、
  `--security-opt=no-new-privileges`を基本とする。
- `/tmp`は可能ならtmpfsにするが、大規模モデルの保存先には使わない。
- Jupyterはランダムtoken必須とし、tokenなし起動を禁止する。
- Docker socketをコンテナへ渡さない。
- `--ipc=host`を使用しない。
- runtime container終了後に、そのセッション専用imageと専用buildx builderを削除する。
- 無関係なDocker資産を消す `docker system prune -a` を使用しない。
- 通常のCUDA/L4TモードではTailscaleをインストールも起動もせず、Jupyterを
  `127.0.0.1`へport publishする。

### 6.2 CUDA

`Dockerfile.cuda`はx86_64 NVIDIA CUDA環境を対象とする。ホストはDriver 595.84、
`nvidia-smi`表示CUDA 13.2だが、コンテナ内PyTorchはRTX 2080で実績のある安定版CUDA
wheelを選び、バージョンを固定する。ホスト表示と同じCUDA toolkitを無条件に選ばない。

起動は次を主経路とする。

```bash
bash ./agent/docker/run.sh cuda
```

GPUは原則 `--gpus device=0` で1台だけ渡す。

### 6.3 L4T / Jetson

`Dockerfile.l4t`はJetPack/L4T対応イメージを使用し、x86用PyTorch wheelを入れない。
Jetsonの機種、JetPack、共有RAMが未指定なので、初期の保証範囲はbuild、Jupyter、CUDA認識、
主要importまでとする。FLUX.2/3の実生成をx86と同等に保証しない。

起動は次とする。

```bash
bash ./agent/docker/run.sh l4t
```

L4Tベースイメージをbuild argumentで差し替えられるようにする。

### 6.4 shellとCompose

主要経路は `run.sh` 内の明示的な `docker buildx build` と `docker run` にする。
`docker compose`を主要経路にしない。`run.sh`には補助経路としてComposeコマンド例を
コメントアウトして記載する。

`compose.yml`には `cuda` と `l4t` のprofileを用意するが、オマケ扱いとする。

### 6.5 clone禁止の借用GPUマシン

借用GPUマシンではホストにリポジトリをcloneせず、Dockerのremote Git build contextから
構築する。Docker imageにはTailscaleを含めず、このモードだけruntime container内へ
公式static binaryを一時展開してcontainerをtailnetへ参加させる。
READMEに通常ローカル実行とは分離した専用章を設け、固定commitを指定した完全な
build/run/login/cleanup手順を記載する。

例の概念形：

```bash
docker buildx build \
  --pull \
  --no-cache \
  --load \
  --file agent/docker/Dockerfile.cuda \
  --tag image-agent-session:local \
  'https://github.com/<owner>/<repo>.git?ref=<commit>'
```

READMEの借用GPU手順は、ホストに `run.sh`やrepository working treeがあることを前提にしては
ならない。buildとrunをまとめた「一度に貼り付けられるshell command block」を提供する。
これは複数のDocker CLI呼び出しを安全に連結してよいが、ユーザー操作は一度の貼り付けで
済むようにする。公開リポジトリを前提とし、private repositoryの場合の認証方法は秘密を
image、build arg、ログへ残さない方法だけを別途記載する。

## 7. 借用GPUモード専用Tailscale要件

Tailscaleはclone禁止の借用GPUマシンでだけ使用する。通常の `run.sh cuda`、`run.sh l4t`では
インストール、起動、参加を行わない。通常モードのentrypointがTailscale認証を要求しては
ならない。

借用GPUモードではruntime container自身をユーザーのtailnetへ参加させる。

- Auth keyを使用しない。
- Dockerfileおよび通常imageへTailscaleをインストールしない。
- Docker run時に `ENABLE_TAILSCALE=1`を明示した場合だけ公式static binaryをcontainer内へ
  ダウンロードし、runtimeディレクトリから起動する。
- `tailscaled --tun=userspace-networking --state=mem:` を基本とする。
- `tailscale up`または同等の対話ログインでURLを端末へ表示する。
- ユーザーがURLを開いて承認するまで待機する。
- Tailscale Serveでコンテナ内のJupyterへHTTPSで接続できるようにする。
- ホストのTailscaleを使用しない。
- `/dev/net/tun`や`NET_ADMIN`を要求しないuserspace構成を優先する。
- 終了時に可能ならlogoutし、stateはファイルへ永続化しない。
- Tailscaleの外部control planeには一時node情報が残り得ることをREADMEへ記載する。

借用GPUモードでTailscale認証が完了していなくても、entrypointが無限に無言で待たないよう、
状態とURLを端末へ明瞭に表示する。

## 8. ハードウェア・メモリ方針

対象：

```text
GPU: NVIDIA GeForce RTX 2080
VRAM: 8192 MiB
CPU RAM: 32 GB
GPU世代: Turing
```

既定値：

```yaml
runtime:
  compute_dtype: float16
  allow_bfloat16: false
  allow_fp8: false
  allow_nvfp4: false
  max_vram_gib: 7.5
  min_free_ram_gib: 20
  batch_size: 1
```

- モデルロード前にGPU、空きVRAM、空きRAM、空きディスクを検査する。
- VRAMピーク、reserved VRAM、CPU RAMピーク、ロード時間、推論時間を記録する。
- Qwen、FLUX、Upscalerを別プロセスとして逐次実行する。
- OOMはNotebook kernelを巻き込まず、構造化エラーとしてrunへ保存する。
- OOM時は解像度、参照画像数、tile sizeを安全側へ下げるが、無限再試行しない。
- 32GB RAMを超える構成を安全に動くとみなさない。

## 9. モデル

### 9.1 Planner / Critic

第一候補：

```text
Qwen/Qwen3-VL-4B-Instruct
```

用途：

- 多言語Markdown理解。
- 参考画像分析。
- 顔、髪、体格、衣装、小物、画風、構図の分析。
- 固定条件、変更可能条件、新規設計条件、推測条件、矛盾の分類。
- 補助生成を含むJob Graphの作成。
- FLUX向け完全キャプションの作成。
- 生成結果の評価。
- 修正内容の提案。

RTX 2080では4bit重量とFP16 computeを基本とする。Thinkingモデルを使用せず、隠れた思考の
出力を要求しない。必要なのは短い根拠要約と検証可能な構造化JSONのみである。

### 9.2 FLUX.2

第一MVP：

```text
black-forest-labs/FLUX.2-klein-4B
```

学習用Baseではなくdistilled生成モデルを使用する。Diffusersの公式対応を使用し、4bit量子化、
sequential CPU offload、VAE tiling/slicingを基本とする。

初期値：

```yaml
flux2:
  width: 512
  height: 512
  steps: 4
  guidance_scale: 1.0
  max_reference_images: 1
  reference_max_side: 512
  max_sequence_length: 256
  quantization: bnb4
  offload: sequential
```

次に768x432、432x768、参照画像2枚を段階的に試す。未検証のメモリ推定を成功扱いしない。

### 9.3 FLUX.3

第二バックエンド：

```text
FLUX 3 Image Klein distilled
```

RTX 2080では公式ComfyUI向けINT8 ConvRotファイルを優先する。

```text
flux3-image-klein-int8-convrot.safetensors
qwen3vl_4b_int8_convrot.safetensors
flux2-vae.safetensors
```

FLUX.3の公式9B Prompt Upsamplerは8GB/32GB環境で重いため既定ではロードしない。Plannerの
QwenにBFL公開のsystem prompt相当を与え、完全なシーンキャプションを事前生成させる。
ComfyUI workflow側ではPrompt Upsamplerを無効にし、完成済みキャプションを渡す。

ComfyUIは常設しない。FLUX.3ジョブごとにlocalhostで子プロセス起動し、APIへworkflowを投入し、
成果物回収後にプロセスを終了する。

FLUX.3のINT8 8GB例はRTX 3070級であり、RTX 2080での成功は実機試験まで未検証とする。
失敗時はFLUX.2へ明示的にフォールバックできるようにする。

FLUX.3 weightsはFLUX Kommunity Licenseであり、FLUX.2 4BのApache 2.0と異なる。READMEへ
ライセンス差を記載する。

### 9.4 バージョン固定

モデルrevision、Python依存、ComfyUI commit、workflowを固定し、run metadataへ記録する。
実装時点の公式配布元と公式APIを確認する。不明なモデル形式やcommunity conversionを、
確認なしで主要経路へ入れない。

## 10. 入力仕様

ユーザーはルート `dataset/` にMarkdownと画像を置く。コンテナ内では
`/workspace/dataset/`として扱う。Markdownは固定JSON/YAML形式を要求せず、
日本語、英語、中国語および混在を許可する。

例：

```markdown
# 雪山のシーン

このキャラクターを雪山で描く。
顔と髪型、襟、胸のエンブレムは変更しない。
袖と裾を冬服として新しく設計する。

参考画像:
- 顔: ./characters/alice_face.png
- 全身: ./characters/alice_fullbody.png
- 衣装: ./outfits/uniform.png
- 画風: ./styles/style01.png

The character should look back toward the camera.
使用低角度视角，保持原来的动漫画风。
```

パス解決規則：

- 相対パスはMarkdownファイルのディレクトリ基準。
- `resolve()`後に `/workspace/dataset` 配下であることを確認する。
- symlink、`..`、絶対パスによるdataset外脱出を拒否する。
- 許可画像拡張子を明示する。
- 画像をPillowで実際にdecodeして検証する。
- 画像へSHA-256と安定した`asset_id`を付ける。
- Qwenが返したパスを信用せず、Qwenには原則`asset_id`だけを返させる。

## 11. スキーマ

少なくとも次のPydanticモデルを実装する。

- `AssetRecord`
- `AssetManifest`
- `ConstraintSet`
- `ScenePlan`
- `GenerationJob`
- `JobGraph`
- `CriticResult`
- `UpscaleResult`
- `ExecutionMetrics`
- `AgentState`
- `RunStatus`

Assetの由来を次で区別する。

```text
user_original
agent_inferred
agent_derived
generated_candidate
```

`agent_derived`は`user_original`と同格の正式資料として扱わない。AIが補完した設定は元設定と
区別してNotebookへ表示する。

`GenerationJob`は少なくとも次を持つ。

```text
id
type
purpose
depends_on
input_asset_ids
backend
prompt
negative_constraints
width
height
steps
seed
status
output_asset_ids
```

許可job type：

```text
composition_draft
pose_draft
costume_sheet
item_sheet
background_draft
main_render
localized_edit
upscale
```

## 12. Agent制御

LangGraphの低レベルな`StateGraph`とPydanticを使用する。LangChainの汎用Agent、自由なReAct、
smolagents CodeAgentは使用しない。各graph nodeは通常のPython関数として実装し、将来
LangGraphを外せるようドメイン処理を密結合させない。

状態遷移：

```text
validate_input
  -> analyze_and_plan
  -> validate_plan
  -> review_plan
  -> generate_auxiliary
  -> inspect_auxiliary
  -> generate_main
  -> critic
  -> decide
       -> finish
       -> revise
       -> user_required
```

`decide`はQwenではなくPythonが行う。

ハードリミット：

```text
MAX_GRAPH_STEPS = 20
MAX_AUXILIARY_JOBS = 3
MAX_GENERATION_CANDIDATES = 2
MAX_GENERATION_REVISIONS = 2
MAX_PLAN_REPAIRS = 1
MAX_UPSCALE_ATTEMPTS = 2
MAX_REFERENCES_PER_JOB = 4
```

同じplan hashを再生成した場合は停止する。低confidence、矛盾、正式資料不足、最大回数到達時は
`user_required`または失敗として停止し、無限に再生成しない。

Qwen出力は次の順で扱う。

```text
raw text
-> JSON抽出
-> Pydantic検証
-> AssetManifest照合
-> job数・依存関係・サイズ・backend検証
-> 実行可能jobへ変換
```

JSON修復は最大1回。修復失敗時はユーザーへエラーを示す。

## 13. 補助生成

Plannerは本生成前に必要な場合だけ補助jobを選ぶ。毎回すべてを実行しない。

用途例：

- 衣装の前後・部品構造を整理する`costume_sheet`。
- ファッションアイテム単体を整理する`item_sheet`。
- 低解像度の`composition_draft`。
- ポーズや視点を確認する`pose_draft`。
- 背景要素の`background_draft`。

補助画像はQwenで確認し、明白な破綻がある場合のみ有限回修正する。最終生成では補助画像だけに
依存せず、正式なユーザー参考画像も常に参照可能にする。

## 14. Critic

Criticには次を渡す。

- 元Markdown全文。
- `user_original`参考画像。
- ScenePlan。
- 使用した補助画像。
- 生成候補。

評価項目：

- 顔、目、髪、体格。
- 衣装、エンブレム、小物。
- 画風、線、塗り、色、陰影。
- 構図、ポーズ、カメラ位置。
- 背景、時間帯、天候。
- `must_preserve`違反。
- 変更を許可されていない箇所の変化。
- AI補完部分の不確実性。

出力action：

```text
accept
retry_prompt
retry_reference
localized_edit
user_required
```

項目別score、confidence、短い根拠、修正可能範囲を構造化して返す。Critic自身の判定を過信せず、
低confidenceや重要条件不一致はユーザー確認へ送る。

## 15. アップスケール

生成画像がCriticに受理された後だけアップスケールする。FLUXプロセス終了とVRAM解放確認後に
別子プロセスで実行する。

第一候補：

```text
RealESRGAN_x4plus_anime_6B
RealESRGAN_x4plus
```

任意バックエンド：

```text
waifu2x-ncnn-vulkan
```

Real-ESRGANを既定にする。waifu2x-ncnn-vulkanはVulkan依存が増えるためオプションにする。

既定：

```yaml
upscale:
  enabled: true
  backend: realesrgan
  anime_model: RealESRGAN_x4plus_anime_6B
  general_model: RealESRGAN_x4plus
  tile: 256
  tile_padding: 16
  fp16: true
  face_enhance: false
```

GFPGAN等のface enhancementはキャラクター同一性を変える可能性があるため既定で無効にする。

目標：

```text
768x432  -> 1920x1080
432x768  -> 1080x1920
512x512  -> 2048x2048
```

倍率適用後に厳密な目標解像度へリサイズまたは最小限のクロップを行い、その処理をmetadataへ
記録する。中間JPEGを使用せずPNGを基本とする。

OOM時はtileを256から128へ1回だけ下げる。元画像、アップスケール画像、最終サイズ画像を
すべて保存する。アップスケール後は、顔、エンブレム、文字、線、タイル境界の破綻を別QAとして
確認する。アップスケール失敗だけでFLUX生成を最初からやり直さない。

## 16. run出力

```text
agent/runs/run_<timestamp>_<id>/
├── original_prompt.md
├── asset_manifest.json
├── scene_plan.json
├── job_graph.json
├── generation_config.json
├── controls/
├── auxiliary/
├── candidates/
├── final/
│   ├── base.png
│   ├── upscaled.png
│   └── fullhd.png
├── evaluation.json
├── upscale.json
├── execution_metrics.json
├── execution_log.jsonl
└── state.json
```

存在しない工程のダミーファイルは作らない。各JSONはatomic writeする。画像そのものを
LangGraph stateへ埋め込まず、パス、hash、metadataだけを保持する。

記録項目：

- model IDとrevision。
- quantizationとdtype。
- dependency versionとComfyUI commit。
- 参考画像hash。
- 入力promptと最終caption。
- Seed、steps、guidance、解像度。
- workflow hash。
- 各工程の時間、VRAM、RAM。
- fallbackとretryの理由。

## 17. Notebook

`agent/image_agent.ipynb`は起動直後に理解できるUIにする。

セル構成：

1. 使用方法と制約。
2. GPU、RAM、ディスク、モデル状態の確認。
3. `dataset`へのアップロードまたは既存ファイル選択。
4. Markdownと参考画像のプレビュー。
5. backend、seed、解像度、修正回数、upscale設定。
6. 計画のみ実行。
7. ScenePlanとJob Graphの表示。
8. ユーザー承認または修正指示。
9. 生成開始。
10. 補助画像と進捗の表示。
11. 最終候補とCritic評価の表示。
12. アップスケールと最終画像表示。
13. run一式のZIPダウンロード。

Notebookの公開API概念：

```python
from image_agent.notebook import ImageAgentSession

session = ImageAgentSession(
    prompt="./dataset/request.md",
    backend="flux2",
)
plan = session.plan()
session.show_plan()
result = session.run()
session.show_result()
```

Notebook import時にモデルをダウンロード・ロードしない。生成を明示的に実行するまで重い処理を
開始しない。Notebook再実行による二重jobを防ぐ。

## 18. エラー・再開

- 各node開始前後にstateとstatusを保存する。
- CUDA OOM、モデル取得失敗、不正JSON、画像decode失敗、ComfyUI timeout、子プロセス異常終了を
  区別する。
- 子プロセスにはtimeoutを設け、stdout/stderrをrunログへ保存する。
- 秘密tokenをログへ出さない。
- 失敗したnodeから再開できるが、完了済みの重いjobを不必要に再実行しない。
- 入力hashやconfigが変わった場合は古いcheckpointを無条件に再利用しない。
- SIGTERM/SIGINTで子プロセスを終了し、借用GPUモードではTailscaleも終了する。

## 19. テスト

### 19.1 GPU不要

- Markdown解析と日英中混在。
- 相対パス解決とdataset外脱出拒否。
- symlink拒否。
- AssetManifestとhash。
- 全Pydantic schema。
- 不正JSONと1回だけの修復。
- Job Graph依存関係、循環、上限制御。
- plan hash重複停止。
- Agent分岐と最大ステップ。
- run保存、atomic write、再開。
- mock Qwen、mock FLUX、mock Upscalerによるend-to-end。
- CLI引数と設定優先順位。
- 秘密値のログredaction。

### 19.2 Docker静的・CPU

- CUDA/L4T Dockerfile build構文。
- entrypointのsignal処理。
- Jupyter tokenなし起動を拒否。
- 通常モードでTailscaleを起動・要求しない。
- 借用GPUモードでだけTailscaleを有効化できる。
- containerにbind/volumeを要求しない。
- runtime pathが `/workspace/agent/.runtime` に限定される。
- `check_environment.sh`の明瞭な診断。

### 19.3 x86 GPU実機

- RTX 2080認識。
- Qwen 4bit Planner。
- Planner終了後のVRAM解放。
- FLUX.2 512x512。
- FLUX.2参照画像1枚。
- Critic再ロード。
- Real-ESRGAN tile upscale。
- Full pipelineのVRAMピーク7.5GiB以下を目標。
- OOMからNotebookを失わず復帰。
- FLUX.3 INT8は実際に成功した場合だけ合格記録。

### 19.4 Jetson実機

- 機種、JetPack、L4T、共有RAMを記録。
- build、CUDA認識、import、Jupyter。
- 実生成を行っていない場合は環境確認のみと明記。

## 20. 実装順序

1. 現在のrepositoryとdirty worktreeを調査し、既存変更を保護する。
2. 公式情報から現在のmodel ID、license、Diffusers、ComfyUI、量子化互換性を再確認する。
3. Python package、schema、path安全性、mock backendを実装する。
4. CUDA/L4T Docker、通常モードentrypoint、Jupyterを実装する。
5. Qwen Plannerを実装し、mockと実機を分けて検証する。
6. FLUX.2 backendを実装し、512x512から検証する。
7. LangGraph統合とNotebookを実装する。
8. Qwen Criticと有限修正を実装する。
9. Real-ESRGANとpost-upscale QAを実装する。
10. FLUX.3 ComfyUI INT8 backendを実装する。
11. clone禁止の借用GPUモード、任意Tailscale、remote build手順を実装・文書化する。
12. テスト結果、未検証事項、制約をREADMEへ反映する。

各段階でテストを行い、前段が成立しない状態で後段を完成扱いしない。

## 21. MVP完了条件

最低限、次がすべて満たされたときMVP完了とする。

- `agent/image_agent.ipynb`から操作できる。
- 自由形式Markdownと複数参考画像を入力できる。
- Qwenが検証可能なScenePlanを作る。
- 必要に応じて補助画像を1枚生成できる。
- FLUX.2で本画像を生成できる。
- Qwen Criticが元資料と比較して評価できる。
- 修正ループが最大2回で必ず終了する。
- Real-ESRGANでFull HD相当へ拡大できる。
- Qwen、FLUX、Upscalerが別プロセスで逐次実行される。
- LoRA関連機能がない。
- bind mountとDocker volumeがない。
- 通常モードがTailscaleなしで起動し、localhostからJupyterへ接続できる。
- 借用GPUモードではTailscaleの対話URLからコンテナへ接続できる。
- container、session image、専用builderを終了時に削除できる。
- GPUなし環境でmock end-to-endテストが通る。
- 実機未検証事項が明記されている。

## 22. FLUX.3完了条件

FLUX.3 backendのコードが存在するだけでは対応済みとしない。

- 公式INT8 model filesとworkflowを使用する。
- Prompt Upsampler代替captionをQwenで作る。
- ComfyUI子プロセスを確実に終了する。
- 512px、参照なしから検証する。
- RTX 2080で成功しなければ未対応またはexperimentalと表示する。
- 失敗時にFLUX.2へフォールバックできる。
- licenseをユーザーへ表示する。

## 23. 実装品質

- 型注釈を付ける。
- 外部process実行は引数listを使い、shell文字列結合を避ける。
- Pydanticと明示的な例外型を用いる。
- 依存バージョンを固定する。
- 巨大な単一モジュールを作らない。
- public APIと内部APIを分離する。
- ログはJSONLを基本とし、Notebookには要約を表示する。
- すべての実装変更を`agent/`配下に限定し、既存の`sd15/`、`flux2/`、`shared/`、
  ルートREADMEその他の既存ファイルを変更しない。ルート`.gitignore`の既存agent例外は維持する。
- 既存実装は参考にしてよいが、LoRA前提、bind mount前提、固定Markdown schema前提のコードを
  そのまま新agentへ持ち込まない。
- テストしていない成功をREADMEに書かない。

## 24. 実装者が判断してよい範囲

- Pythonモジュールの細かな分割。
- CLIライブラリ、ログライブラリ、Notebook表示部品。
- 同等の安全性を維持したcheckpoint形式。
- RTX 2080で実測して決めるtile sizeやoffload細部。
- 公式API変更に伴うクラス名やworkflow形式の更新。

次は変更してはならない。

- LoRAなし。
- 単一runtime container。
- bind/volumeなし。
- Qwen/FLUX/Upscalerの逐次子プロセス実行。
- Python側の有限状態制御。
- CUDA/L4Tの2系統。
- shell主経路、Compose補助経路。
- clone禁止の借用GPUマシン専用remote Git build・Tailscale手順。
- 通常モードではTailscaleを使用しない。
- `agent/`直下のNotebook。
- Full HD相当へのアップスケール。
