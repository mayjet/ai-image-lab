"""GPU and hardware diagnostics."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GpuInfo:
    available: bool = False
    device_count: int = 0
    devices: list[dict] = field(default_factory=list)
    driver_version: Optional[str] = None
    cuda_version_host: Optional[str] = None
    # pytorch view
    torch_cuda_available: bool = False
    torch_version: Optional[str] = None
    torch_cuda_version: Optional[str] = None


def get_gpu_info() -> GpuInfo:
    """
    Collect GPU information without crashing if no GPU is present.

    Uses nvidia-smi and optionally torch for detailed VRAM stats.
    Returns a best-effort GpuInfo regardless of environment.
    """
    info = GpuInfo()
    _probe_nvidia_smi(info)
    _probe_torch(info)
    return info


def _probe_nvidia_smi(info: GpuInfo) -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return
        devices = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                devices.append(
                    {
                        "name": parts[0],
                        "memory_total_mib": _safe_int(parts[1]),
                        "memory_free_mib": _safe_int(parts[2]),
                        "memory_used_mib": _safe_int(parts[3]),
                        "driver_version": parts[4] if len(parts) > 4 else None,
                    }
                )
        if devices:
            info.available = True
            info.device_count = len(devices)
            info.devices = devices
            info.driver_version = devices[0].get("driver_version")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass


def _probe_torch(info: GpuInfo) -> None:
    try:
        import torch  # noqa: PLC0415

        info.torch_version = torch.__version__
        info.torch_cuda_available = torch.cuda.is_available()
        if info.torch_cuda_available:
            info.torch_cuda_version = torch.version.cuda
            if not info.devices:
                # Populate from torch if nvidia-smi was not available
                info.available = True
                info.device_count = torch.cuda.device_count()
                for i in range(info.device_count):
                    props = torch.cuda.get_device_properties(i)
                    info.devices.append(
                        {
                            "name": props.name,
                            "memory_total_mib": props.total_memory // (1024 * 1024),
                            "memory_free_mib": None,
                            "memory_used_mib": None,
                        }
                    )
    except ImportError:
        pass


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except (ValueError, TypeError):
        return None
