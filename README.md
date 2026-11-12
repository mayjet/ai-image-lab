# ai-image-lab

ローカルGPUで画像生成・検証を行うためのリポジトリです。SD1.5/FLUX.2のLoRAワークフローに加え、
Qwen3-VLとFLUX.2を使うLoRAなしの画像生成エージェントを収録しています。

```bash
git clone https://github.com/mayjet/ai-image-lab.git
cd ai-image-lab
```

## 構成

```text
datasets/                 非公開の共有データセット
loras/                    非公開の学習済みLoRA
sd15/
  docker/                 SD1.5専用CUDA/L4T環境
  *.ipynb                 SD1.5 Notebook（起動直後に開ける直下配置）
  prompts/                SD1.5個人用プロンプト
  models/base/            再取得可能なベースモデル
  cache/ outputs/ vendor/ SD1.5専用ランタイム領域
flux2/
  docker/                 FLUX.2専用CUDA/L4T環境
  *.ipynb                 FLUX.2 Notebook（起動直後に開ける直下配置）
  prompts/                FLUX.2個人用プロンプト
  models/base/            再取得可能なベースモデル
  cache/ outputs/ vendor/ FLUX.2専用ランタイム領域
agent/
  image_agent.ipynb       Qwen3-VL + FLUX.2画像生成エージェント
  docker/                 使い捨てCUDA/L4T環境
  src/ tests/ configs/    実装、テスト、モデル・生成設定
shared/                   真にモデル非依存なパス・互換処理
```

`datasets/`の既存構造は変更しません。`datasets/`、`loras/`、各バックエンドの
`models/`、`cache/`、`outputs/`、`vendor/`、`prompts/`はGit管理外です。

## Docker

Dockerはリポジトリ全体を`/ai-image-lab`へマウントします。SD1.5コンテナの
作業ディレクトリは`/ai-image-lab/sd15`、FLUX.2は`/ai-image-lab/flux2`です。

x86 NVIDIA CUDA:

```bash
bash ./sd15/docker/run.sh cuda
bash ./flux2/docker/run.sh cuda
```

Jetson / L4T:

```bash
bash ./sd15/docker/run.sh l4t
bash ./flux2/docker/run.sh l4t
```

FLUX.2 L4Tは環境確認用で、LoRA学習はサポート対象外です。

- Jupyter Lab: `http://localhost:8888/lab`
- TensorBoard: `http://localhost:6006/`

各Dockerfile専用の`.dockerignore`により、データセット、LoRA、モデル、出力、
キャッシュはDocker build contextへ入りません。

## データセット

共有データセットは次の構造を維持します。

```text
datasets/<work>/<character>/
├── lora/
├── portrait/
├── anime/
├── game/
├── face/
├── illust/
└── character.toml
```

作品ディレクトリはコマンドの`--dataset-root`またはNotebook冒頭で明示します。

## SD1.5

ローカルvenv:

```bash
bash ./sd15/setup_local.sh
```

顔画像準備、学習、検証:

```bash
python3 sd15/anime_face_collect.py --dataset-root ./datasets/<work>
python3 sd15/train_lora.py --dataset-root ./datasets/<work> --train
python3 sd15/validate_lora.py --dataset-root ./datasets/<work>
```

Notebook:

```text
sd15/train_lora.ipynb
sd15/generate_character.ipynb
```

SD1.5 LoRAは`loras/sd15/<base-model>/<work>/<character>/`へ保存されます。

詳細は[`sd15/README.md`](sd15/README.md)を参照してください。

## FLUX.2

```bash
bash ./flux2/docker/run.sh cuda
python3 flux2/generate.py --prompt "A portrait" --dry-run
```

Notebook:

```text
flux2/train_lora.ipynb
flux2/generate.ipynb
```

FLUX.2 LoRAは`loras/flux2/black-forest-labs--flux.2-klein-base-4b/<run-name>/`へ保存されます。

詳細は[`flux2/README.md`](flux2/README.md)を参照してください。

## LoRAなし画像生成エージェント

RTX 2080（VRAM 8GB）向けの実験的な省メモリ構成です。入力Markdownと参考画像を
`./dataset/`へ置き、次を実行します。

```bash
bash ./agent/docker/run.sh cuda
```

起動後は、ターミナルに表示されたURLから`agent/image_agent.ipynb`を開きます。
通常起動ではTailscaleを使いません。clone禁止の借用GPUマシンで、コンテナだけを
一時的にTailscaleへ参加させる手順は[`agent/README.md`](agent/README.md)を参照してください。
借用GPU用のremote buildは固定commitではなく`main`を参照するため、実行時点の最新版が
使用されます。

Qwen3-VL、FLUX.2、Real-ESRGANを含むGPU実機経路は未検証です。既知の制約と
検証状況も`agent/README.md`に記載しています。

## プロンプトと出力

公開テンプレートと個人用ファイルをバックエンドごとに分離します。

```text
sd15/prompt_template.md
sd15/prompts/
flux2/prompt_template.md
flux2/prompts/
```

出力とTensorBoardログもバックエンド内に限定されます。

```text
sd15/outputs/{training,validation,generation}/
flux2/outputs/{training,validation,generation}/
```

## Git管理

公開前に次を確認します。

```bash
git status --short
git status --ignored --short datasets loras \
  sd15/cache sd15/models sd15/outputs sd15/vendor sd15/prompts \
  flux2/cache flux2/models flux2/outputs flux2/vendor flux2/prompts
```

作品名、データセット、LoRA、生成画像、個人用プロンプトは公開Gitへ追加しません。
