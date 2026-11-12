"""System resource diagnostics: GPU, RAM, disk."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from image_agent.errors import DiskSpaceError, RAMError


@dataclass
class ResourceReport:
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    vram_total_mib: Optional[float] = None
    vram_free_mib: Optional[float] = None
    ram_total_gib: Optional[float] = None
    ram_available_gib: Optional[float] = None
    disk_free_gib: Optional[float] = None
    warnings: list[str] = field(default_factory=list)


def check_resources(
    workspace: Path = Path("/workspace"),
    min_free_ram_gib: float = 20.0,
    max_vram_gib: float = 7.5,
) -> ResourceReport:
    """
    Check GPU, RAM, and disk resources.

    Logs warnings but does not raise unless critically short.
    """
    report = ResourceReport()
    _check_gpu(report, max_vram_gib)
    _check_ram(report, min_free_ram_gib)
    _check_disk(report, workspace)
    return report


def _check_gpu(report: ResourceReport, max_vram_gib: float) -> None:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            report.gpu_available = True
            props = torch.cuda.get_device_properties(0)
            report.gpu_name = props.name
            total_mib = props.total_memory / (1024 * 1024)
            report.vram_total_mib = total_mib
            mem = torch.cuda.mem_get_info(0)
            report.vram_free_mib = mem[0] / (1024 * 1024)
            if total_mib > (max_vram_gib * 1024 * 1.1):
                report.warnings.append(
                    f"VRAM {total_mib:.0f} MiB exceeds expected max "
                    f"{max_vram_gib * 1024:.0f} MiB"
                )
    except ImportError:
        report.warnings.append("torch not available — GPU check skipped")
    except Exception as exc:
        report.warnings.append(f"GPU check error: {exc}")


def _check_ram(report: ResourceReport, min_free_gib: float) -> None:
    try:
        import psutil  # noqa: PLC0415

        vm = psutil.virtual_memory()
        report.ram_total_gib = vm.total / (1024**3)
        report.ram_available_gib = vm.available / (1024**3)
        if report.ram_available_gib < min_free_gib:
            report.warnings.append(
                f"Low RAM: {report.ram_available_gib:.1f} GiB free "
                f"(minimum {min_free_gib} GiB recommended)"
            )
    except ImportError:
        # psutil optional
        try:
            with open("/proc/meminfo") as f:
                meminfo = {
                    line.split(":")[0]: int(line.split(":")[1].strip().split()[0])
                    for line in f
                    if ":" in line
                }
            total_kib = meminfo.get("MemTotal", 0)
            avail_kib = meminfo.get("MemAvailable", 0)
            report.ram_total_gib = total_kib / (1024**2)
            report.ram_available_gib = avail_kib / (1024**2)
        except Exception:
            report.warnings.append("RAM check skipped (psutil not available)")


def _check_disk(report: ResourceReport, workspace: Path) -> None:
    try:
        usage = shutil.disk_usage(workspace)
        report.disk_free_gib = usage.free / (1024**3)
        if report.disk_free_gib < 50:
            report.warnings.append(
                f"Low disk: {report.disk_free_gib:.1f} GiB free under {workspace}"
            )
    except Exception as exc:
        report.warnings.append(f"Disk check error: {exc}")
