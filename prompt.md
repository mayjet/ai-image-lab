# prompt

このファイルは公開用テンプレートです。通常は `./prompts/<任意の名前>.md` へコピーして編集し、
Notebookの `PROMPT_MD_PATH` をそのファイルへ変更してください。`./prompts/` の内容はGit管理されません。

## settings
character: <character-folder-name>
aspect: portrait_3_4
# JetsonでControlNetを使う場合は64の倍数を推奨
width: 384
height: 512

## references
source_image: ./datasets/<works-name>/<character-folder-name>/portrait/<image>.png
identity: ./datasets/<works-name>/<character-folder-name>/portrait/<image>.png
pose: ./datasets/<works-name>/<character-folder-name>/game/<image>.png
costume: ./datasets/<works-name>/<character-folder-name>/game/<image>.png
style: ./datasets/<works-name>/<character-folder-name>/illust

## lora
# 必要なroleだけをカンマ区切りで指定する。
use_roles: base, style
# 現在のフォルダ別学習重みをNotebookのroleへ明示的に割り当てる。
base: ./models/sd15/<works-name>/<character-folder-name>/anime.safetensors
style: ./models/sd15/<works-name>/<character-folder-name>/illust.safetensors
base_weight: 1.00
style_weight: 0.25

## generation_preferences
最終画像で優先したい画風を指定する。
参照画像ごとの役割を分ける。例: identityは顔と髪、poseは姿勢、costumeは衣装、styleは画風。
低品質な参照画像は、形状や配置の参考に限定し、画風や質感としては使わない。
変更してよい部分と、維持したい部分を明確に書く。
数値設定は通常ここに書かない。LoRA重み、img2img strength、ControlNet scale、IP-Adapter scale、seed探索はQwenが自動で提案し、Notebook側で安全範囲に制限する。

## positive
参照画像と同じキャラクターとして描く。
顔立ち、目の色、髪色、髪型の主要構造、全体の雰囲気、キャラクター性を維持する。
ユーザーが指定した変更内容だけを反映する。
指定した構図、ポーズ、服装、背景、画風に従う。
シャープで綺麗な現代アニメ調またはイラスト調にする。
線画は清潔にし、塗りは自然で破綻の少ない仕上げにする。

ここに今回生成したい内容を日本語で具体的に書く。
例:
- 年齢印象を少し若くする
- 指定した制服や衣装へ変更する
- 正面の全身立ち絵にする
- 背景は白にする
- 特定フォルダの画風を優先する

## negative
低品質、ぼやけ、ノイズ、jpeg劣化、文字、透かし、署名。
SNSのUI、コメント欄、アイコン、ユーザー名、キャプション文字。
2人、複数人、双子、重複した人物、余分な顔。
顔違い、髪色違い、目の色違い、キャラクター性の変化。
手の崩れ、指の崩れ、余分な指、融合した指、折れた指。
指定していない背景物、写実調、粗い3D調、低品質なゲーム画面風。

ここに今回避けたい具体的な変化を書く。

## rules
指定されていない領域は原則として変更しない。
identity画像はキャラクター同一性の参照として扱う。
pose画像は姿勢と構図の参照として扱う。
costume画像は服装構造の参照として扱う。
styleフォルダは最終画風の参照として扱う。
参照画像の役割を混同しない。
生成結果が複数人になる場合は失敗とする。
指示されていない背景要素を追加しない。
