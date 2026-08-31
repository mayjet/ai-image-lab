# ai-image-lab

キャラクター画像データセットからLoRAを学習し、参照画像と指示を使って生成・検証する環境です。
SD1.5ワークフローを維持しながら、FLUX.2 kleinの複数参照編集へ移行できる構成です。

## 目次

- [構成](#構成)
- [Docker](#docker)
- [ローカルvenv](#ローカルvenv)
- [データセット](#データセット)
- [SD15](#sd15)
- [FLUX2](#flux2)
- [プロンプト](#プロンプト)
- [OllamaとTailscale](#ollamaとtailscale)
- [出力とモデル](#出力とモデル)
- [Git管理](#git管理)

## 構成

```text
sd15/       SD1.5の学習・検証・生成
flux2/      FLUX.2の学習・検証・生成（移行中）
shared/     共通のパス・前処理・実行環境互換処理
datasets/   作品単位の非公開データセット
models/     学習済み重み
outputs/    学習ログ・検証・生成結果
cache/      Hugging Face・モデル固有キャッシュ
vendor/     sd-scripts、IP-Adapter、FLUX.2公式コード
prompts/    個人用プロンプト
docker/     CUDA/L4T Docker環境
```

`datasets/`、`models/`、`outputs/`、`cache/`、`vendor/`、個人用`prompts/`はGit管理外です。

`shared/`にはこのプロジェクトが管理するモデル非依存コードを置きます。`vendor/`は外部リポジトリと外部実装専用で、自作コードは置きません。Dockerイメージへ組み込む顔検出ランナーとJetson互換処理も、役割が分かるよう`shared/face_detection/`と`shared/compat/`に配置しています。

## Docker

プロジェクトルートから、マシン別のスクリプトとバックエンドを指定して起動します。コンテナは前景で動作し、`Ctrl+C`で停止すると自動削除されます。

```bash
bash ./docker/run.sh sd15
bash ./docker/run.sh flux2
```

Jetson/L4Tでは`run_l4t.sh`を使います。

```bash
bash ./docker/run_l4t.sh sd15
bash ./docker/run_l4t.sh flux2
```

`run.sh`と`run_l4t.sh`はComposeを実行せず、`docker build`と前景の`docker run --rm -it`を直接実行します。Compose例はスクリプト内のコメントと`docker-compose.yml`にのみ残しています。

Dockerのビルドコンテキストはリポジトリルートですが、`.dockerignore`により`docker/`と`shared/`以外は送信されません。非公開のデータセット、モデル、プロンプト、出力はイメージ構築へ含まれません。

- Jupyter Lab: `http://localhost:8888/lab`
- TensorBoard: `http://localhost:6006/`

Dockerは次の4環境を分離します。

```text
Dockerfile.sd15.cuda
Dockerfile.sd15.l4t
Dockerfile.flux2.cuda
Dockerfile.flux2.l4t
```

FLUX.2 L4Tは環境構築・疎通確認専用です。Jetson上での推論やLoRA学習は成功条件に含めません。

詳しくは [docker/README.md](docker/README.md) を参照してください。

## ローカルvenv

Apple Siliconを含むローカル環境では、次のコマンドでSD1.5用venvとipykernelを構築します。

```bash
bash ./uvvenv.sh
```

Jupyter Labは自動起動しません。VS Codeなどから登録済みkernelを選択します。

## データセット

標準配置は次です。

```text
datasets/<works>/<character>/
├── portrait/
├── anime/
├── game/
├── face/
├── illust/
└── character.toml
```

`<works>`は作品を表すディレクトリ名です。作品ごとに画像の特性とユーザーが定義した分類があるため、コードは既存の分類フォルダを移動・改名・細分化しません。画風や見た目を追加分類する場合も、ユーザーが作品配下へ新しいフォルダを追加したうえで明示的に指定します。

公開例と引数未指定時の既定値には`./datasets/works`を使用します。実データセットはCLIの`--dataset-root`またはNotebook冒頭で作品ディレクトリを指定します。

SD1.5の`*_sd.npz`は再生成可能なlatentキャッシュなので、データセットには保存しません。

## SD15

本体は`sd15/`、共通前処理は`shared/`にあります。ルート直下にはPythonスクリプトとNotebookを置きません。

### 顔画像収集

```bash
python3 shared/anime_face_collect.py \
  --dataset-root ./datasets/works
```

### LoRA学習

```bash
python3 sd15/train_lora.py \
  --dataset-root ./datasets/works \
  --train
```

キャラクター・項目を限定できます。

```bash
python3 sd15/train_lora.py \
  --dataset-root ./datasets/works \
  --characters character-a \
  --folders anime face \
  --train
```

### LoRA検証

```bash
python3 sd15/validate_lora.py \
  --dataset-root ./datasets/works
```

Notebookは次の本体を直接開きます。

```text
sd15/train_lora.ipynb
sd15/generate_character.ipynb
```

### Jupyter Labから実行

Docker起動後に`http://localhost:8888/lab`を開き、ファイル一覧から`sd15/`へ入ってNotebookを開きます。コンテナではリポジトリが`/workspace`へマウントされ、Notebook冒頭が`.git`を基準に`/workspace`へ移動するため、`datasets/`、`models/`、`outputs/`の相対パスは維持されます。

### VS Codeから実行

VS Codeでリポジトリルートを開き、`sd15/`内のNotebookを直接開きます。ローカルvenvの場合は`bash ./uvvenv.sh`で登録したkernelを選択します。Dockerを使う場合はVS CodeのNotebookから既存Jupyter Serverとして`http://localhost:8888`へ接続し、`Python (sd15-cuda)`または`Python (sd15-l4t)`を選択します。どちらの場合もNotebookがプロジェクトルートを自動検出するため、開いたファイルの階層に依存しません。

## FLUX2

JetsonではDocker/CUDA/PyTorch/Diffusers/Jupyterの環境だけを先に用意します。

```bash
bash ./docker/run_l4t.sh flux2
```

詳しい手順は [flux2/README.md](flux2/README.md) にあります。後日のLoRA学習は
x86 NVIDIA GPU上で `bash ./docker/run.sh flux2` を使い、対象モデルを
`FLUX.2-klein-base-4B` とします。

実装済みです。

- FLUX.2 klein 4Bによるテキスト生成
- FLUX.2 klein Base 4BのキャラクターLoRA学習
- 1枚の参照画像による編集
- TensorBoardによるloss・学習率の記録
- CLIとNotebookの低メモリ実行

複数参照編集、参照画像ごとの役割指定、LoRAあり・なしの一括比較は未実装です。

既存のSD1.5 LoRAはFLUX.2では使用できません。FLUX用LoRAは別途学習します。

## プロンプト

`prompt.md`は公開テンプレートです。個人用プロンプトは`prompts/`へ置きます。

```bash
cp prompt.md prompts/my-prompt.md
```

個人用ファイルはGit管理されません。

SD1.5ではnegative promptやLoRA強度を使用します。FLUX.2では複数画像の役割を明示し、negative promptではなく望む状態を肯定形で記述します。

FLUX.2のCLIとNotebookも同じMarkdownを読みます。

```bash
python3 flux2/generate.py --prompt-md prompts/prompt1.md --no-lora --dry-run
```

FLUX.2用の新規テンプレートは `flux2/prompt_template.md` です。

## OllamaとTailscale

SD1.5生成Notebookでは、Tailscale上のOllama VLMを使用できます。

```text
http://ollama.tail1cadae.ts.net:11434
```

VLMはプロンプト整理、参照画像確認、候補評価に使用します。生成モデルへ渡す参照画像をVLMのテキストだけに置き換えません。

## 出力とモデル

SD1.5モデルは次に保存されます。

```text
models/sd15/<works>/<character>/
├── anime.safetensors
├── anime.train.json
├── face.safetensors
├── face.train.json
└── checkpoints/
```

出力はバックエンド別です。

```text
outputs/sd15/training/
outputs/sd15/validation/
outputs/sd15/generation/
outputs/flux2/training/
outputs/flux2/validation/
outputs/flux2/generation/
```

TensorBoardログは各`training/logs/`へ保存されます。

## Git管理

公開前に次を確認します。

```bash
git status --short
git status --ignored --short datasets models outputs cache vendor prompts
```

コード、Notebook、Docker設定、公開テンプレートだけが追跡対象です。データセット、学習重み、生成画像、キャッシュ、個人用プロンプトは追跡しません。
