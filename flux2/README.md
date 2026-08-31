# FLUX.2 LoRA: Jetsonで先に環境だけ作る

対象は `black-forest-labs/FLUX.2-klein-base-4B` を使うLoRA学習です。
Jetsonでは学習せず、Docker、NVIDIA runtime、CUDA版PyTorch、Diffusersの
import、Jupyterまでを準備します。学習は後でVRAM 24 GB以上のx86 NVIDIA GPUへ
同じリポジトリを移して行う想定です。

## 何ができるか

| 環境 | 用途 | 保証範囲 |
|---|---|---|
| Jetson / JetPack 6 | 先行セットアップ | build、CUDA認識、import、Jupyter |
| x86 + NVIDIA GPU | LoRA学習・生成 | Diffusers公式trainer、8GB向け4bit/offload生成 |

Jetsonでモデルをダウンロードしたり、推論・学習を成功条件に含めたりはしません。
FLUX.2 [dev] は32Bで、ここで対象にする [klein] 4Bとは別物です。

## 1. Jetsonホストの前提

- JetPack 6 / L4T r36系、aarch64
- DockerとNVIDIA Container Runtimeが設定済み
- リポジトリとDockerイメージ用の十分な空き容量
- NGC (`nvcr.io`) とGitHubへ接続できること

Jetson側で、必要に応じて次を確認してください。

```bash
uname -m
head -n 1 /etc/nv_tegra_release
docker info
free -h
```

`uname -m` は `aarch64`、L4Tはr36系を想定しています。異なる場合はそのまま
ビルドせず、JetPackの世代とNVIDIA PyTorchコンテナを合わせてください。

## 2. Jetson用イメージを構築・起動

```bash
bash ./docker/run_l4t.sh flux2
```

既定のベースは `nvcr.io/nvidia/pytorch:24.12-py3-igpu` です。JetPackに合わせて
別のNVIDIA iGPU PyTorchイメージを使う場合だけ、明示的に差し替えます。

```bash
L4T_PYTORCH_IMAGE='nvcr.io/nvidia/pytorch:<compatible-tag>-igpu' \
  bash ./docker/run_l4t.sh flux2
```

起動後、別ターミナルで診断します。

```bash
docker exec -it ai-image-lab-flux2-l4t check-ai-image-environment
```

次が表示されればJetson側の準備は完了です。

- `machine: aarch64`
- `CUDA available: True`
- Jetson GPU名と共有メモリ容量
- `Flux2KleinPipeline` のimport成功
- `capability: environment-only`

Jupyter Labは `http://<JetsonのIP>:8888/lab` です。LAN外へ8888番を公開しないで
ください。この構成はトークンなしJupyterなので、信頼できるLANまたはSSH
ポートフォワード内だけで使います。

## 3. 後日、学習機で起動

x86 NVIDIA GPU機では次を使います。

```bash
HF_TOKEN='hf_...' bash ./docker/run.sh flux2
```

モデルカード上でアクセス条件への同意が必要な場合は、同じHugging Face
アカウントで先に同意します。トークンはコンテナへ環境変数として渡され、
DockerイメージやGitには保存されません。

公式学習スクリプトはコンテナ起動時に次へ用意されます。

```text
/workspace/vendor/diffusers/examples/dreambooth/
  train_dreambooth_lora_flux2_klein.py
```

学習モデルには蒸留済み `FLUX.2-klein-4B` ではなく、LoRA向けの
`FLUX.2-klein-base-4B` を使います。学習コマンドと低メモリ既定値は後述します。

## 失敗時

- `no matching manifest`: ベースイメージタグがJetPack/aarch64に非対応
- `CUDA available: False`: NVIDIA runtimeまたはJetPack/Docker設定の問題
- build中のOOM: メモリ、swap、ストレージ空き容量を確認
- `gated repo`: Hugging Faceでアクセス同意後、正しい `HF_TOKEN` を渡す

Jetson用Dockerfileには `bitsandbytes`、FP8学習、モデル取得を入れていません。
これらの欠如は不具合ではなく、環境構築だけを再現可能にするための境界です。

## LoRA学習スクリプト

学習対象はBase 4B、生成対象は蒸留4Bです。

```text
学習: black-forest-labs/FLUX.2-klein-base-4B
生成: black-forest-labs/FLUX.2-klein-4B
```

学習はx86 NVIDIA GPUの `flux2-cuda` コンテナで行います。まずコマンドだけ確認します。

```bash
python3 flux2/train_lora.py \
  --character character-name \
  --instance-prompt 'ZK_CHAR, one anime girl' \
  --run-name zk-char-flux2 \
  --dry-run
```

問題がなければ `--dry-run` を `--train` に変更します。

```bash
python3 flux2/train_lora.py \
  --character character-name \
  --instance-prompt 'ZK_CHAR, one anime girl' \
  --run-name zk-char-flux2 \
  --train
```

既定値はRAM/VRAMを抑えるため、512px、batch 1、gradient checkpointing、latent
cache、worker 0、8-bit Adam、最終生成なしです。FP8はcompute capability 8.9以上で
自動的に有効になります。`--offload` はVRAMを減らしますがCPU RAMを増やすため、
既定では無効です。

`datasets/works/<character>/lora` 内の画像だけを一時symlinkフォルダ経由で公式
Diffusers trainerへ渡します。現状のDreamBooth経路では各画像の `.txt` captionは
使わず、全画像へ `--instance-prompt` を使用します。triggerと対象を表現したpromptを
指定してください。

Notebookは `flux2/train_lora.ipynb` です。誤操作防止のため、学習セルは
`RUN_TRAIN = False` から始まります。

## LoRA生成スクリプト

生成指示は既存の `prompts/prompt1.md` 形式から読み込めます。現在の
`prompt1.md` はSD1.5 LoRAを指しているため、参照画像と指示だけ使う場合は
`--no-lora` を付けます。

```bash
python3 flux2/generate.py \
  --prompt-md prompts/prompt1.md \
  --no-lora \
  --dry-run
```

確認後、`--dry-run` を外すと生成します。

```bash
python3 flux2/generate.py \
  --prompt-md prompts/prompt1.md \
  --no-lora
```

FLUX.2 LoRAが完成したら、Markdownの `## lora` を次のように変更します。

```text
## lora
use_roles: base
base: ./models/flux2/zk-char-flux2/pytorch_lora_weights.safetensors
base_weight: 1.0
```

この場合は `--no-lora` を外します。複数roleを指定した場合も、それぞれの重みで
adapterを読み込みます。SD1.5 LoRAをFLUX.2へ渡そうとすると、読み込み前に停止します。

Markdownの各sectionは次のように処理されます。

| section | FLUX.2での用途 |
|---|---|
| `settings` | width、height、steps、guidance、seed、memory mode、出力先 |
| `references` | 最大4枚の参照画像。重複パスは1枚へ統合 |
| `lora` | role別LoRAパスとweight |
| `generation_preferences` | 画風などの優先事項 |
| `positive` | 主な生成指示 |
| `negative` | `Avoid all of the following` として本文へ統合 |
| `rules` | 維持条件と参照画像の役割 |

参照画像は `source_image`、`identity`、`pose`、`costume`、`style` の順で読みます。
同じファイルを複数roleで指定した場合は1枚にまとめます。ディレクトリ指定では名前順に
画像を選び、合計4枚までに制限します。最終プロンプトには「参照画像1はidentity用」
のような対応関係を自動挿入します。8GB VRAM向けの既定は参照1枚、長辺512pxです。
`settings.max_reference_images` と `settings.reference_max_side` で最大4枚まで変更できますが、
参照画像を増やすほどattentionの作業領域も増えます。

新規プロンプトは [prompt_template.md](prompt_template.md) をコピーして作成できます。

```bash
cp flux2/prompt_template.md prompts/my-flux2-prompt.md
```

Markdownを使わず、短い指示をCLIで直接渡す方法も残しています。

```bash
python3 flux2/generate.py \
  --prompt 'ZK_CHAR, one anime girl, standing in a quiet street at sunset' \
  --lora models/flux2/zk-char-flux2/pytorch_lora_weights.safetensors \
  --output outputs/flux2/generation/zk-char.png
```

蒸留4Bの既定値は4 steps、guidance scale 1.0、512x512です。

### x86 RAM 32GB / CUDA VRAM 8GB

この環境では次をそのまま使います。

```bash
python3 flux2/generate.py \
  --prompt-md prompts/my-flux2-prompt.md \
  --memory-mode auto \
  --quantization auto
```

8GB GPUで `auto` が選ぶ構成は次のとおりです。

- transformerとtext encoderをbitsandbytes NF4 4bitで量子化
- double quantizationを有効化
- checkpointを最初からCPUへ読み込み、量子化時の一時VRAM peakを回避
- sequential CPU offloadで層単位にGPUへ転送
- VAE tilingとVAE slicingを有効化
- `low_cpu_mem_usage=True` でcheckpointを読み込み
- Ampere以降はbf16、それ以前はfp16を自動選択
- prompt tokenを最大256へ制限し、PyTorch SDPAのflash/memory-efficient backendを有効化
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` でCUDA予約領域の断片化を軽減
- 512x512、参照画像1枚・長辺512pxを低メモリ既定値に設定

最低20 GiBの「空き」CPU RAMが必要です。32GB搭載でも、他プロセスがRAMを使って
空きが20 GiB未満ならモデル取得前に停止します。終了時にはpeak VRAM、reserved VRAM、
プロセスpeak RAMを標準出力と生成画像横のJSONへ保存します。自動判定にはGPUの公称容量
ではなく実行直前の空きVRAMを使うため、他のGPUプロセスが動いている場合も安全側へ倒します。

`sequential-offload` は転送回数が多く非常に低速です。一度成功したあと、速度を優先する
場合は `--memory-mode model-offload` を試せます。OOMになる場合はautoへ戻してください。

メモリモードは次のとおりです。

| mode | 長所 | 必要条件・短所 |
|---|---|---|
| `auto` | 8GBでは4bit + sequential offloadを選択 | 空きCPU RAM 20 GiB未満なら停止 |
| `gpu` | CPU/GPU転送が少ない | 非量子化はVRAM約15 GiB以上が目安 |
| `model-offload` | component単位でoffload、比較的速い | 8GBではpeak VRAMが収まらない場合あり |
| `sequential-offload` | 層単位でoffload、VRAMを最小化 | 非常に低速 |
| `group-offload` | 上2方式の中間 | モデル依存の実験的経路 |
| `disk-offload` | CPU RAMもSSDへ逃がせる | 最も低速、十分な高速SSD空き容量が必要 |

量子化は `auto`、`bnb4`、`none` から選べます。VRAM 10GB未満で `none` を使うと
安全判定で停止します。品質比較などで意図的に試す場合だけ `--force` を併用してください。
disk offloadは次のように明示します。

```bash
python3 flux2/generate.py \
  --prompt 'one anime girl, portrait' \
  --memory-mode disk-offload \
  --quantization bnb4 \
  --offload-dir /fast-ssd/flux2-offload
```

VAE対策は既定で有効です。比較目的で無効化するときだけ `--no-vae-tiling` または
`--no-vae-slicing` を使います。slicingは1枚生成では効果が小さいものの、害が少ないため
既定で有効にしています。長いpromptが必要なら `--max-sequence-length 512` へ戻せますが、
text embeddingとattentionのメモリ消費は増えます。

layerwise FP8 castingは4bit量子化の代替であり、LoRA/PEFTとの組み合わせが十分に保証
されていないため既定stackへ重ねていません。CUDA stream付きgroup offloadはCPU RAMを
大きく増やすため32GB環境では無効です。`torch.compile` は速度最適化であってpeak RAM
削減策ではなく初回負荷も増えるため使いません。FLUX transformerにはPyTorch SDPAの
flash/memory-efficient backendを使い、UNet向けattention slicingやxFormersは追加しません。

画像編集では `--input-image` を加え、通常は `--guidance-scale 4.0` を使います。

```bash
python3 flux2/generate.py \
  --prompt 'change the character outfit to a blue jacket' \
  --input-image reference.png \
  --lora models/flux2/zk-char-flux2/pytorch_lora_weights.safetensors \
  --guidance-scale 4.0
```

Notebookは `flux2/generate.ipynb` です。こちらも初期状態では
`RUN_GENERATION = False` です。処理を別プロセスにしているため、生成終了時にNotebook
kernelへ巨大なpipelineを残しません。Notebookの `PROMPT_MD` を変更すれば、CLIと
まったく同じMarkdownパーサーを使用します。
