"""Compatibility shims for NVIDIA's Jetson PyTorch build.

The iGPU build exposes ``torch.distributed`` but omits some functions.
Accelerate calls these functions during shutdown even for a one-process job,
before sd-scripts writes the final LoRA file.
"""

try:
    import torch

    distributed = getattr(torch, "distributed", None)
    if distributed is not None and not hasattr(distributed, "is_initialized"):
        distributed.is_initialized = lambda: False
    if distributed is not None and not hasattr(distributed, "destroy_process_group"):
        distributed.destroy_process_group = lambda group=None: None
except ImportError:
    pass
