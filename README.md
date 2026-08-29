# ai-image-lab

ローカルのキャラクターデータセットからStable Diffusion 1.5用LoRAを準備・学習・検証し、
OllamaのVLMで参照画像の解析と生成候補の選定を行うための環境です。

データセット、学習済み重み、生成結果、個人用プロンプトはGitへ含めません。

## 目次

- [全体構成](#overview)
- [必要環境](#requirements)
- [データセット](#dataset)
- [ローカルPython環境](#local-python)
- [Docker環境](#docker)
  - [x86 CUDA](#docker-cuda)
  - [Jetson / L4T](#docker-jetson)
- [顔データの準備](#face-collection)
- [LoRA学習](#lora-training)
  - [Pythonスクリプト](#training-script)
  - [Notebook](#training-notebook)
- [学習済みLoRAの生成検証](#lora-validation)
- [Ollama VLMとTailscale](#ollama-tailscale)
- [プロンプト](#prompts)
- [キャラクター画像の生成](#generation)
- [出力先](#outputs)
- [トラブルシューティング](#troubleshooting)

<a id="overview"></a>
## 全体構成

```text
入力画像
  ↓
anime_face_collect.py  ─ 顔学習画像を抽出（Jetson環境）
  ↓
train_lora.py / local_lora_train.ipynb
  ↓
validate_lora.py       ─ LoRAなし・項目・強度を比較
  ↓
local_character_single.ipynb
  ├─ Ollama VLMで参照画像と指示を解析
  ├─ SD1.5 + LoRA + img2img
  ├─ optional ControlNet / IP-Adapter
  └─ VLMで候補を評価して最終画像を保存
```

主なファイル:

| ファイル | 用途 |
|---|---|
| `anime_face_collect.py` | 元画像から顔学習データを準備 |
| `train_lora.py` | フォルダ単位のLoRA学習 |
| `local_lora_train.ipynb` | データ確認、必要時の顔収集・学習、生成検証 |
| `validate_lora.py` | 学習済みLoRAの一括生成比較 |
| `local_character_single.ipynb` | VLMを使った1人キャラクター生成 |
| `prompt.md` | 公開用プロンプトテンプレート |
| `uvvenv.sh` | ローカルvenvとipykernelの構築 |

<a id="requirements"></a>
## 必要環境

用途に応じていずれかを使います。

- NVIDIA GPU搭載Linux: x86 CUDA Docker
- Jetson: JetPack 6.x / L4T r36系を想定したDocker
- Apple Siliconなどのローカル環境: uvで作る`.venv`（生成向け）
- VLM: Ollamaで利用できるVision Language Model
- リモートVLM接続: Tailscale

学習スクリプトはCUDAを必要とします。Apple Silicon/MPS環境は主に
`local_character_single.ipynb`による生成に使用します。

<a id="dataset"></a>
## データセット

既定のルートは`./dataset`です。実データを直接置くか、symlinkを作ります。

```bash
ln -s /path/to/private-dataset ./dataset
```

想定構成:

```text
dataset/
└── character-a/
    ├── character.toml
    ├── lora/
    ├── portrait/
    ├── anime/
    ├── game/
    ├── illust/
    ├── face/
    └── folder_loras/
```

画像フォルダの項目は次の6種類です。

```text
lora portrait anime game illust face
```

`character.toml`例:

```toml
schema_version = 1
name = "Character A"
folder = "character-a"
unit = ""
identity_tags = ["high school girl"]

[folders]
lora = []
portrait = ["full body", "character reference"]
anime = ["anime screenshot", "anime style"]
game = ["game artwork"]
illust = ["illustration"]
face = ["face focus", "close-up"]
```

`dataset/`、`.safetensors`、生成画像は`.gitignore`で除外されます。

<a id="local-python"></a>
## ローカルPython環境

Apple Silicon/MPSを含むローカル実行では、プロジェクトルートで次だけを実行します。

```bash
bash ./uvvenv.sh
```

このコマンドは以下を行います。

1. 必要ならuvを導入
2. `.venv`をPython 3.11で作成または更新
3. Notebook・生成・学習用依存を導入
4. `sd-scripts`を`ai-image-lab-work/`へ取得
5. ipykernelをユーザー環境へ登録
6. PyTorch、MPS/CUDA、Diffusers等を診断

VS Codeでは次を選択します。

```text
Python interpreter: ./.venv/bin/python
Notebook kernel: Python (ai-image-lab_venv)
```

Jupyter Labの起動は必須ではありません。環境を完全に作り直す場合:

```bash
RECREATE_VENV=1 bash ./uvvenv.sh
```

<a id="docker"></a>
## Docker環境

コンテナはプロジェクトルートを`/workspace`へマウントします。詳細は
[`docker/README.md`](docker/README.md)も参照してください。

<a id="docker-cuda"></a>
### x86 CUDA

```bash
bash ./docker/run.sh
```

既定コンテナ名:

```text
ai-image-lab-cuda
```

Jupyterは`http://localhost:8888`で起動します。

<a id="docker-jetson"></a>
### Jetson / L4T

```bash
bash ./docker/run_l4t.sh
```

既定コンテナ名とポート:

```text
container:   ai-image-lab-l4t
Jupyter:     http://localhost:8888
TensorBoard: http://localhost:6006
```

環境診断:

```bash
docker exec -it ai-image-lab-l4t check-l4t-environment
```

Hugging Face tokenが必要な場合:

```bash
HF_TOKEN="..." bash ./docker/run_l4t.sh
```

<a id="face-collection"></a>
## 顔データの準備

Jetsonでは顔検出依存を学習環境と分離したPython 3.10環境へ入れています。
スクリプト自身が環境を切り替えます。

環境確認:

```bash
docker exec -it -w /workspace ai-image-lab-l4t \
  python3 anime_face_collect.py --check
```

全キャラクター:

```bash
docker exec -it -w /workspace ai-image-lab-l4t \
  python3 anime_face_collect.py
```

キャラクター指定:

```bash
docker exec -it -w /workspace ai-image-lab-l4t \
  python3 anime_face_collect.py --characters character-a
```

既存判定を無視して再構築する場合は`--rebuild`を追加します。

<a id="lora-training"></a>
## LoRA学習

ベースモデルは既定で`stablediffusionapi/counterfeit-v30`、解像度512、
rank/alpha 32、FP16、CLIP skip 2です。

<a id="training-script"></a>
### Pythonスクリプト

データ準備:

```bash
docker exec -it -w /workspace ai-image-lab-l4t \
  python3 train_lora.py --prepare
```

検証のみ:

```bash
docker exec -it -w /workspace ai-image-lab-l4t \
  python3 train_lora.py --validate
```

全キャラクター・全項目を学習:

```bash
docker exec -it -w /workspace ai-image-lab-l4t \
  python3 train_lora.py --train
```

対象指定:

```bash
docker exec -it -w /workspace ai-image-lab-l4t \
  python3 train_lora.py --train \
  --characters character-a \
  --folders lora anime face
```

同じデータ・設定の完成済み重みはfingerprintによりスキップされます。
再学習は`--force`を追加します。

<a id="training-notebook"></a>
### Notebook

`local_lora_train.ipynb`は以下を一続きで確認できます。

1. 対象キャラクター・項目の指定
2. 入力画像とcaptionのプレビュー
3. 必要な場合だけ顔収集
4. 学習済み重みが揃っていれば学習をスキップ
5. 不足分を`train_lora.py`で学習
6. `validate_lora.py`による生成比較

冒頭の`DATASET_ROOT`、`TARGET_CHARACTERS`、`TARGET_FOLDERS`を設定して、
登録済みkernelまたはDocker内kernelで上から実行します。

<a id="lora-validation"></a>
## 学習済みLoRAの生成検証

対象確認だけ行う場合:

```bash
docker exec -it -w /workspace ai-image-lab-l4t \
  python3 validate_lora.py --dry-run
```

全キャラクター・全項目:

```bash
docker exec -it -w /workspace ai-image-lab-l4t \
  python3 validate_lora.py
```

対象と強度を指定:

```bash
docker exec -it -w /workspace ai-image-lab-l4t \
  python3 validate_lora.py \
  --characters character-a \
  --folders anime face \
  --scales 0.8 0.9 1.0 \
  --seeds 42 43 44
```

途中チェックポイントも比較する場合は`--checkpoints`を追加します。
結果は画像、比較グリッド、JSON、CSVで保存されます。

<a id="ollama-tailscale"></a>
## Ollama VLMとTailscale

Ollamaサーバー側でTailscaleへ参加し、Ollamaをtailnetから到達可能にします。

```bash
sudo systemctl enable --now tailscaled
sudo tailscale up
```

Ollamaを全インターフェースで待ち受けさせ、VLMを準備します。

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

別のターミナル:

```bash
ollama pull qwen2.5vl:7b
```

生成側マシンも同じtailnetへ参加し、確認します。

```bash
tailscale ping ollama.tail1cadae.ts.net
curl http://ollama.tail1cadae.ts.net:11434/api/tags
```

Notebook既定値:

```python
OLLAMA_BASE_URL = "http://ollama.tail1cadae.ts.net:11434"
OLLAMA_MODEL = "qwen2.5vl:7b"
```

Docker内でMagicDNS名を解決できない場合は、OllamaサーバーのTailscale IPv4
（`100.x.y.z`）を`OLLAMA_BASE_URL`へ直接指定してください。

<a id="prompts"></a>
## プロンプト

`prompt.md`は公開用テンプレートです。個人用ファイルは`prompts/`へコピーします。

```bash
cp prompt.md prompts/my-prompt.md
```

Notebook設定:

```python
PROMPT_MD_PATH = Path("./prompts/my-prompt.md")
```

`prompts/.gitkeep`だけがGit管理され、その他の`prompts/`内ファイルは無視されます。

プロンプトでは次を指定できます。

- キャラクターフォルダ名
- identity / pose / costume / style参照
- positive / negative / rules
- 使用するLoRAファイルと基本強度
- 必要なら生成探索条件

<a id="generation"></a>
## キャラクター画像の生成

`local_character_single.ipynb`をVS Code等で開き、使用環境のkernelを選択します。

- ローカル: `Python (ai-image-lab_venv)`
- Docker: `Python (ai-image-lab)`

冒頭で主に次を確認します。

```python
DATASET_ROOT = Path("./dataset")
PROMPT_MD_PATH = Path("./prompts/my-prompt.md")
QUALITY_PRESET = "standard"
EXECUTION_MODE = "auto"
```

処理順:

1. プロンプトと参照画像を読み込む
2. Ollama VLMが参照画像を分析してSD用タグと生成方針を作る
3. 指定LoRAをDiffusersへロードする
4. img2imgと必要に応じてControlNet/IP-Adapterで候補を生成する
5. VLMが候補を比較して最良画像を選ぶ
6. 必要な場合だけinpaint修復を行う
7. `final.png`とmetadataを保存する

32GB RAMのApple Silicon環境では`QUALITY_PRESET = "standard"`から開始できます。
Jetsonでメモリ不足になる場合は、ControlNetとIP-Adapterを一度無効にし、
64の倍数の小さい解像度から段階的に有効化してください。

<a id="outputs"></a>
## 出力先

```text
ai-image-lab-work/output/
├── anime_face_collect/
├── folder_lora_train/
├── lora_validation/
└── local_character_single/
```

生成Notebookの各runには概ね次が保存されます。

```text
final.png
final_metadata.json
candidates/
candidate_plan.json
verify_report.json
specs/
control/
logs/
```

<a id="troubleshooting"></a>
## トラブルシューティング

### プロンプトが見つからない

```python
print(Path.cwd())
print(PROMPT_MD_PATH.resolve())
print(PROMPT_MD_PATH.is_file())
```

通常は`./prompts/...`です。`.prompts/...`ではありません。

### ControlNetのtensorサイズが一致しない

生成の幅と高さを64の倍数にします。例:

```text
384x512
512x512
640x832
768x1024
```

サイズ変更後はinit/control画像の作成セルから再実行します。

### JetsonのNvMap / CUDAメモリエラー

Jupyter kernelを再起動し、次の順に負荷を下げます。

1. IP-Adapterとinpaintを無効化
2. ControlNetを無効化
3. 解像度を下げる
4. 候補数を減らす
5. 必要なら`low_vram`を使う

### Ollamaのホスト名をDocker内で解決できない

ホスト側のTailscale接続を確認し、解決できなければTailscale IPv4を直接指定します。

### 個人データがGitへ入っていないか確認する

```bash
git status --ignored --short dataset prompts ai-image-lab-work
```

これらはignore表示になるのが正常です。`prompts/.gitkeep`だけは公開対象です。
