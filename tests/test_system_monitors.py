"""JARVIS Lite — System Monitors and Telemetry Alert Tests."""

import pytest

from system.health import SystemHealthAggregator
from system.network_monitor import NetworkMonitor
from system.resource_monitor import GpuMetrics, ResourceMonitor, ResourceSnapshot


@pytest.mark.asyncio
async def test_resource_monitor_snapshot() -> None:
    monitor = ResourceMonitor()
    snap = monitor.get_snapshot()

    assert snap.timestamp > 0
    assert 0.0 <= snap.cpu_percent <= 100.0
    assert snap.cpu_count_logical >= 1
    assert snap.ram_total_mb > 0
    assert snap.ram_percent > 0
    assert snap.disk_total_gb > 0
    assert isinstance(snap.top_processes, list)

    # Check GPU if present on RTX 3050 Laptop
    if snap.gpu is not None:
        assert isinstance(snap.gpu.gpu_name, str)
        assert snap.gpu.total_vram_mb > 0
        assert snap.gpu.used_vram_mb >= 0
        assert 0.0 <= snap.gpu.vram_percent <= 100.0

    monitor.close()


@pytest.mark.asyncio
async def test_network_monitor_latency_and_metrics() -> None:
    monitor = NetworkMonitor(timeout=1.0)
    metrics = await monitor.get_metrics()

    assert metrics.timestamp > 0
    assert isinstance(metrics.is_connected, bool)
    assert metrics.total_bytes_sent_mb >= 0.0
    assert metrics.total_bytes_recv_mb >= 0.0


@pytest.mark.asyncio
async def test_health_aggregator_vram_threshold_alert() -> None:
    # Set VRAM threshold to 5500 MB (5.5 GB) on 6 GB RTX 3050
    aggregator = SystemHealthAggregator(vram_limit_mb=5500.0, cpu_threshold_pct=90.0)

    # Create dummy snapshot simulating VRAM exceeding threshold
    snap = ResourceSnapshot(
        timestamp=100.0,
        cpu_percent=45.0,
        cpu_count_logical=8,
        ram_total_mb=16000.0,
        ram_used_mb=8000.0,
        ram_percent=50.0,
        disk_total_gb=512.0,
        disk_used_gb=200.0,
        disk_free_gb=312.0,
        disk_percent=40.0,
        gpu=GpuMetrics(
            gpu_name="NVIDIA GeForce RTX 3050 Laptop GPU",
            total_vram_mb=6144.0,
            used_vram_mb=5800.0,  # Exceeds 5500 MB
            free_vram_mb=344.0,
            vram_percent=94.4,
            utilization_percent=85.0,
        ),
        battery=None,
        top_processes=[],
    )

    alerts = aggregator._evaluate_thresholds(snap)
    assert len(alerts) >= 1
    vram_alert = next((a for a in alerts if a.name == "gpu_vram_high"), None)
    assert vram_alert is not None
    assert vram_alert.severity == "critical"
    assert "High risk of Out-Of-Memory" in vram_alert.message
