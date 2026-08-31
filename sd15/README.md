# SD1.5 workflow

SD1.5、kohya-ss/sd-scripts、ControlNet、IP-Adapterを使う従来ワークフローです。

- `train_lora.py`: フォルダ単位のLoRA学習
- `validate_lora.py`: ベースラインとLoRAの生成比較
- `train_lora.ipynb`: 顔収集、学習、検証のNotebook
- `generate_character.ipynb`: VLM支援付きSD1.5生成Notebook

CLIはプロジェクトルートから`python3 sd15/train_lora.py`、`python3 sd15/validate_lora.py`として実行します。顔収集は`python3 shared/anime_face_collect.py`です。
