# FLUX.2 generation prompt

このファイルを `prompts/<name>.md` へコピーして編集する。

## settings
character: character-name
width: 512
height: 512
steps: 4
guidance_scale: 1.0
max_sequence_length: 256
seed: 42
memory_mode: auto
quantization: auto
# disk-offloadを選ぶ場合のみ使用。高速なSSD上を推奨。
offload_dir: ./cache/flux2-offload
max_reference_images: 1
reference_max_side: 512
output: ./outputs/flux2/generation/character-name.png

## references
source_image: ./datasets/works/character-name/portrait/source.png
identity: ./datasets/works/character-name/portrait/identity.png
pose: ./datasets/works/character-name/game/pose.png
costume: ./datasets/works/character-name/game/costume.png
style: ./datasets/works/character-name/illust

## lora
# LoRAなしの場合はuse_rolesを空にする。
use_roles: base
base: ./models/flux2/character-name/pytorch_lora_weights.safetensors
base_weight: 1.0

## generation_preferences
最終画像で優先したい画風を書く。
identityは顔と髪、poseは姿勢、costumeは衣装、styleは画風として扱う。
低品質な参照は形状や配置だけに使い、質感を引き継がない。

## positive
同一キャラクターとして描く。
ここに生成したい構図、衣装、背景、画風を具体的に書く。

## negative
低品質、ぼやけ、文字、透かし、複数人、余分な顔、崩れた手を避ける。
ここに避けたい変化を書く。

## rules
指定されていない特徴は変更しない。
参照画像の役割を混同しない。
生成結果が複数人になる場合は失敗とする。
