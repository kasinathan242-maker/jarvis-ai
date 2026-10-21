"""JARVIS Lite — System Resource Monitor.

Monitors CPU, RAM, Disk, GPU utilization, VRAM usage (via NVML),
Battery status, and top processes by CPU/memory consumption.
Designed specifically for RTX 3050 Laptop GPU (6 GB VRAM) on Windows 11.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import psutil

try:
    import pynvml
    HAS_PYNVML = True
except (ImportError, Exception):
    HAS_PYNVML = False


@dataclass
class GpuMetrics:
    gpu_name: str
    total_vram_mb: float
    used_vram_mb: float
    free_vram_mb: float
    vram_percent: float
    utilization_percent: float
    temperature_c: float | None = None


@dataclass
class ProcessInfo:
    pid: int
    name: str
    cpu_percent: float
    memory_mb: float


@dataclass
class BatteryMetrics:
    percent: float
    is_plugged: bool
    secs_left: int | None = None


@dataclass
class ResourceSnapshot:
    timestamp: float
    cpu_percent: float
    cpu_count_logical: int
    ram_total_mb: float
    ram_used_mb: float
    ram_percent: float
    disk_total_gb: float
    disk_used_gb: float
    disk_free_gb: float
    disk_percent: float
    gpu: GpuMetrics | None
    battery: BatteryMetrics | None
    top_processes: list[ProcessInfo]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResourceMonitor:
    """Collects real-time hardware telemetry on Windows 11."""

    def __init__(self) -> None:
        self._nvml_initialized = False
        self._init_nvml()

    def _init_nvml(self) -> None:
        if not HAS_PYNVML:
            return
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except Exception:
            self._nvml_initialized = False

    def get_gpu_metrics(self) -> GpuMetrics | None:
        """Query GPU and VRAM metrics from NVIDIA NVML."""
        if not self._nvml_initialized:
            self._init_nvml()
        if not self._nvml_initialized:
            return None

        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return None

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name_raw = pynvml.nvmlDeviceGetName(handle)
            name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = round(mem_info.total / (1024 * 1024), 2)
            used_mb = round(mem_info.used / (1024 * 1024), 2)
            free_mb = round(mem_info.free / (1024 * 1024), 2)
            vram_pct = round((mem_info.used / mem_info.total) * 100, 1) if mem_info.total > 0 else 0.0

            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = float(util_info.gpu)

            temp_c: float | None = None
            try:
                temp_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass

            return GpuMetrics(
                gpu_name=name,
                total_vram_mb=total_mb,
                used_vram_mb=used_mb,
                free_vram_mb=free_mb,
                vram_percent=vram_pct,
                utilization_percent=gpu_util,
                temperature_c=temp_c,
            )
        except Exception:
            return None

    def get_battery_metrics(self) -> BatteryMetrics | None:
        """Query laptop battery state via psutil."""
        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return None
            is_unlimited = battery.secsleft == psutil.POWER_TIME_UNLIMITED or battery.secsleft < 0
            secs_left = None if is_unlimited else battery.secsleft
            return BatteryMetrics(
                percent=round(battery.percent, 1),
                is_plugged=battery.power_plugged if battery.power_plugged is not None else True,
                secs_left=secs_left,
            )
        except Exception:
            return None

    def get_top_processes(self, limit: int = 5) -> list[ProcessInfo]:
        """Return the top processes consuming memory/CPU."""
        procs: list[ProcessInfo] = []
        try:
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
                try:
                    info = p.info
                    pid = info.get("pid")
                    name = info.get("name") or "unknown"
                    cpu = float(info.get("cpu_percent") or 0.0)
                    mem_info = info.get("memory_info")
                    mem_bytes = getattr(mem_info, "rss", 0) if mem_info is not None else 0
                    mem_mb = round(mem_bytes / (1024 * 1024), 1)

                    if pid is not None and pid != 0:
                        procs.append(ProcessInfo(pid=pid, name=name, cpu_percent=cpu, memory_mb=mem_mb))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by memory descending
            procs.sort(key=lambda x: x.memory_mb, reverse=True)
            return procs[:limit]
        except Exception:
            return []

    def get_snapshot(self) -> ResourceSnapshot:
        """Capture an atomic snapshot of current system resource consumption."""
        import time

        # CPU & Memory
        cpu_pct = float(psutil.cpu_percent(interval=None))
        cpu_count = psutil.cpu_count(logical=True) or 1

        vmem = psutil.virtual_memory()
        ram_total = round(vmem.total / (1024 * 1024), 1)
        ram_used = round(vmem.used / (1024 * 1024), 1)
        ram_pct = float(vmem.percent)

        # Disk on primary drive C:
        drive_path = os.getenv("SystemDrive", "C:") + "\\"
        disk = psutil.disk_usage(drive_path)
        disk_total_gb = round(disk.total / (1024**3), 1)
        disk_used_gb = round(disk.used / (1024**3), 1)
        disk_free_gb = round(disk.free / (1024**3), 1)
        disk_pct = float(disk.percent)

        gpu = self.get_gpu_metrics()
        battery = self.get_battery_metrics()
        top_procs = self.get_top_processes(limit=5)

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_pct,
            cpu_count_logical=cpu_count,
            ram_total_mb=ram_total,
            ram_used_mb=ram_used,
            ram_percent=ram_pct,
            disk_total_gb=disk_total_gb,
            disk_used_gb=disk_used_gb,
            disk_free_gb=disk_free_gb,
            disk_percent=disk_pct,
            gpu=gpu,
            battery=battery,
            top_processes=top_procs,
        )

    def close(self) -> None:
        """Release NVML bindings."""
        if self._nvml_initialized and HAS_PYNVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False


# Global singleton instance
resource_monitor = ResourceMonitor()
