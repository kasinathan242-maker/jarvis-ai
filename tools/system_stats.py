"""JARVIS Lite — System Telemetry Stats Tool.

Gathers hardware metrics: CPU usage, RAM utilization, disk capacity,
and NVIDIA GPU / VRAM status (RTX 3050 Laptop).
"""

from __future__ import annotations

import shutil

import psutil
from pydantic import BaseModel, Field

from security.permissions import RiskLevel
from tools.base import BaseTool


class SystemStatsInput(BaseModel):
    include_gpu: bool = Field(default=True, description="Query NVIDIA GPU metrics via NVML")


class SystemStatsOutput(BaseModel):
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    disk_free_gb: float
    disk_total_gb: float
    gpu_name: str | None = None
    gpu_vram_used_mb: float | None = None
    gpu_vram_total_mb: float | None = None
    status: str = "nominal"


class SystemStatsTool(BaseTool[SystemStatsInput, SystemStatsOutput]):
    name = "system_stats"
    description = "Retrieve live system telemetry: CPU, RAM, disk space, and NVIDIA GPU VRAM."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 4.0
    input_schema = SystemStatsInput
    output_schema = SystemStatsOutput

    async def _execute(self, params: SystemStatsInput) -> SystemStatsOutput:
        # CPU
        cpu = psutil.cpu_percent(interval=0.1)

        # RAM
        mem = psutil.virtual_memory()
        ram_used = round(mem.used / (1024**3), 2)
        ram_total = round(mem.total / (1024**3), 2)
        ram_pct = mem.percent

        # Disk
        disk = shutil.disk_usage("C:/")
        disk_free = round(disk.free / (1024**3), 2)
        disk_total = round(disk.total / (1024**3), 2)

        # GPU
        gpu_name: str | None = None
        vram_used: float | None = None
        vram_total: float | None = None

        if params.include_gpu:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    import pynvml
                    pynvml.nvmlInit()
                    if pynvml.nvmlDeviceGetCount() > 0:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        raw_name = pynvml.nvmlDeviceGetName(handle)
                        gpu_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        vram_used = round(mem_info.used / (1024**2), 1)
                        vram_total = round(mem_info.total / (1024**2), 1)
                    pynvml.nvmlShutdown()
            except Exception:
                pass

        return SystemStatsOutput(
            cpu_percent=cpu,
            ram_used_gb=ram_used,
            ram_total_gb=ram_total,
            ram_percent=ram_pct,
            disk_free_gb=disk_free,
            disk_total_gb=disk_total,
            gpu_name=gpu_name,
            gpu_vram_used_mb=vram_used,
            gpu_vram_total_mb=vram_total,
            status="nominal",
        )

    async def verify(self, params: SystemStatsInput, result: SystemStatsOutput) -> bool:
        # Verify physical sanity bounds
        is_cpu_valid = 0.0 <= result.cpu_percent <= 100.0
        is_ram_valid = 0.0 < result.ram_total_gb < 1024.0 and 0.0 <= result.ram_used_gb <= result.ram_total_gb
        is_disk_valid = result.disk_total_gb > 0.0
        return is_cpu_valid and is_ram_valid and is_disk_valid
