"""JARVIS Lite — Network Health and Latency Monitor.

Tracks network connection latency, active interface stats,
and bandwidth consumption (bytes/sec transmitted and received).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Any

import psutil


@dataclass
class NetworkMetrics:
    timestamp: float
    is_connected: bool
    latency_ms: float | None
    bytes_sent_per_sec: float
    bytes_recv_per_sec: float
    total_bytes_sent_mb: float
    total_bytes_recv_mb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NetworkMonitor:
    """Monitors network latency and throughput."""

    def __init__(self, ping_host: str = "1.1.1.1", ping_port: int = 53, timeout: float = 1.5) -> None:
        self.ping_host = ping_host
        self.ping_port = ping_port
        self.timeout = timeout
        self._last_time: float = time.time()
        net_io = psutil.net_io_counters()
        self._last_bytes_sent: int = net_io.bytes_sent if net_io else 0
        self._last_bytes_recv: int = net_io.bytes_recv if net_io else 0

    async def check_latency_ms(self) -> float | None:
        """Measure TCP connect latency to DNS host."""
        t0 = time.perf_counter()
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(self.ping_host, self.ping_port),
                timeout=self.timeout,
            )
            elapsed = (time.perf_counter() - t0) * 1000.0
            writer.close()
            await writer.wait_closed()
            return round(elapsed, 1)
        except Exception:
            # Fallback check to 8.8.8.8
            try:
                t0_fb = time.perf_counter()
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection("8.8.8.8", 53),
                    timeout=self.timeout,
                )
                elapsed = (time.perf_counter() - t0_fb) * 1000.0
                writer.close()
                await writer.wait_closed()
                return round(elapsed, 1)
            except Exception:
                return None

    async def get_metrics(self) -> NetworkMetrics:
        """Capture network health metrics and current throughput."""
        now = time.time()
        time_delta = max(0.001, now - self._last_time)

        net_io = psutil.net_io_counters()
        bytes_sent = net_io.bytes_sent if net_io else 0
        bytes_recv = net_io.bytes_recv if net_io else 0

        sent_delta = max(0, bytes_sent - self._last_bytes_sent)
        recv_delta = max(0, bytes_recv - self._last_bytes_recv)

        sent_rate = round(sent_delta / time_delta, 1)
        recv_rate = round(recv_delta / time_delta, 1)

        self._last_time = now
        self._last_bytes_sent = bytes_sent
        self._last_bytes_recv = bytes_recv

        latency = await self.check_latency_ms()
        connected = latency is not None

        return NetworkMetrics(
            timestamp=now,
            is_connected=connected,
            latency_ms=latency,
            bytes_sent_per_sec=sent_rate,
            bytes_recv_per_sec=recv_rate,
            total_bytes_sent_mb=round(bytes_sent / (1024 * 1024), 2),
            total_bytes_recv_mb=round(bytes_recv / (1024 * 1024), 2),
        )
