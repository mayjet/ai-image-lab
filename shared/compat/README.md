# Runtime compatibility helpers

`jetson_torch_compat.py` supplies missing `torch.distributed` functions in the
Jetson PyTorch build. It also detects NVIDIA PyTorch 2.6 alpha builds that lack
`TransformGetItemToIndex` and makes Transformers select its compatible mask
path. `jetson_torch_compat.pth` loads the module automatically when Python
starts inside an L4T image. These files are not used on ordinary CUDA or Apple
Silicon environments.
