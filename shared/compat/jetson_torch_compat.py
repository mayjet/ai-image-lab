"""Compatibility shims for NVIDIA's Jetson PyTorch build.

The iGPU build exposes ``torch.distributed`` but omits some functions.
Accelerate calls these functions during shutdown even for a one-process job,
before a training script writes its final LoRA file.

Some NVIDIA 2.6.0 alpha builds are also reported as PyTorch 2.6 while missing
``TransformGetItemToIndex``.  Transformers otherwise selects its PyTorch 2.6
mask path and fails during import.  In that specific case only, report the
2.6 feature check as false so Transformers uses its compatible mask path.
"""

try:
    import importlib

    import torch

    distributed = getattr(torch, "distributed", None)
    if distributed is not None and not hasattr(distributed, "is_initialized"):
        distributed.is_initialized = lambda: False
    if distributed is not None and not hasattr(distributed, "destroy_process_group"):
        distributed.destroy_process_group = lambda group=None: None

    try:
        trace_ops = importlib.import_module("torch._dynamo._trace_wrapped_higher_order_op")
        has_transform_getitem = hasattr(trace_ops, "TransformGetItemToIndex")
    except ImportError:
        has_transform_getitem = False

    if not has_transform_getitem:
        from transformers.utils import import_utils as transformers_import_utils

        original_torch_version_check = transformers_import_utils.is_torch_greater_or_equal

        def jetson_torch_version_check(version, *args, **kwargs):
            if str(version) == "2.6":
                return False
            return original_torch_version_check(version, *args, **kwargs)

        transformers_import_utils.is_torch_greater_or_equal = jetson_torch_version_check
except ImportError:
    pass
