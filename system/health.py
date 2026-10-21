"""JARVIS Lite — System Health Aggregator & Alert Engine.

Continuously tracks telemetry, checks critical safety thresholds
(e.g., VRAM > 5.5 GB on 6 GB RTX 3050, CPU > 90%, Battery < 20%),
and dispatches events onto the EventBus for the UI and voice alerts.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Any

from core.event_bus import bus
from core.logging import get_logger
from system.network_monitor import NetworkMonitor
from system.resource_monitor import ResourceMonitor, ResourceSnapshot

logger = get_logger("system.health")

EVENT_SYSTEM_METRICS = "system.metrics"
EVENT_SYSTEM_ALERT = "system.alert"


@dataclass
class AlertCondition:
    name: str
    severity: str  # "warning", "critical", "info"
    message: str
    metric_value: float
    threshold_value: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SystemHealthAggregator:
    """Aggregates resource and network metrics, evaluating threshold rules."""

    def __init__(
        self,
        vram_limit_mb: float = 5600.0,  # 5.5 GB threshold on 6 GB RTX 3050
        cpu_threshold_pct: float = 90.0,
        ram_threshold_pct: float = 92.0,
        battery_low_pct: float = 20.0,
    ) -> None:
        self.vram_limit_mb = vram_limit_mb
        self.cpu_threshold_pct = cpu_threshold_pct
        self.ram_threshold_pct = ram_threshold_pct
        self.battery_low_pct = battery_low_pct

        self.resource_monitor = ResourceMonitor()
        self.network_monitor = NetworkMonitor()

        self._active_alerts: dict[str, AlertCondition] = {}
        self._history: list[dict[str, Any]] = []
        self._max_history = 60  # Keep last 60 snapshots
        self._running = False
        self._task: asyncio.Task[None] | None = None

    @property
    def active_alerts(self) -> list[AlertCondition]:
        return list(self._active_alerts.values())

    def get_active_alerts(self) -> list[AlertCondition]:
        return list(self._active_alerts.values())

    async def get_latest_health(self) -> dict[str, Any]:
        """Fetch a combined telemetry and network snapshot."""
        resource_snap = self.resource_monitor.get_snapshot()
        network_snap = await self.network_monitor.get_metrics()

        alerts = self._evaluate_thresholds(resource_snap)

        data = {
            "timestamp": time.time(),
            "resources": resource_snap.to_dict(),
            "network": network_snap.to_dict(),
            "alerts": [a.to_dict() for a in alerts],
        }

        # Save to circular history buffer
        self._history.append(data)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return data

    def _evaluate_thresholds(self, snap: ResourceSnapshot) -> list[AlertCondition]:
        current_alerts: list[AlertCondition] = []
        now = time.time()

        # 1. GPU VRAM Check
        if snap.gpu is not None:
            if snap.gpu.used_vram_mb >= self.vram_limit_mb:
                alert = AlertCondition(
                    name="gpu_vram_high",
                    severity="critical",
                    message=(
                        f"VRAM usage critical: {snap.gpu.used_vram_mb:.0f} MB / "
                        f"{snap.gpu.total_vram_mb:.0f} MB (> {self.vram_limit_mb:.0f} MB). "
                        "High risk of Out-Of-Memory."
                    ),
                    metric_value=snap.gpu.used_vram_mb,
                    threshold_value=self.vram_limit_mb,
                    timestamp=now,
                )
                current_alerts.append(alert)

        # 2. CPU Utilization Check
        if snap.cpu_percent >= self.cpu_threshold_pct:
            alert = AlertCondition(
                name="cpu_util_high",
                severity="warning",
                message=f"CPU utilization sustained above {self.cpu_threshold_pct:.0f}%: {snap.cpu_percent:.1f}%",
                metric_value=snap.cpu_percent,
                threshold_value=self.cpu_threshold_pct,
                timestamp=now,
            )
            current_alerts.append(alert)

        # 3. RAM Utilization Check
        if snap.ram_percent >= self.ram_threshold_pct:
            alert = AlertCondition(
                name="ram_util_high",
                severity="warning",
                message=f"System RAM utilization above {self.ram_threshold_pct:.0f}%: {snap.ram_percent:.1f}%",
                metric_value=snap.ram_percent,
                threshold_value=self.ram_threshold_pct,
                timestamp=now,
            )
            current_alerts.append(alert)

        # 4. Battery Low Check
        if snap.battery is not None and not snap.battery.is_plugged:
            if snap.battery.percent <= self.battery_low_pct:
                alert = AlertCondition(
                    name="battery_critical",
                    severity="critical" if snap.battery.percent <= 10.0 else "warning",
                    message=f"Battery low ({snap.battery.percent:.0f}%) and running on battery power.",
                    metric_value=snap.battery.percent,
                    threshold_value=self.battery_low_pct,
                    timestamp=now,
                )
                current_alerts.append(alert)

        # Track and log newly triggered or resolved alerts
        new_active = {a.name: a for a in current_alerts}
        for name, alert in new_active.items():
            if name not in self._active_alerts:
                logger.warn("system_threshold_alert_triggered", alert=alert.to_dict())
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(bus.publish(EVENT_SYSTEM_ALERT, alert.to_dict()))
                except RuntimeError:
                    pass

        for name in list(self._active_alerts.keys()):
            if name not in new_active:
                logger.info("system_threshold_alert_resolved", alert_name=name)

        self._active_alerts = new_active
        return current_alerts

    async def start_periodic_monitoring(self, interval_seconds: float = 3.0) -> None:
        """Start background loop publishing metrics to EventBus."""
        if self._running:
            return
        self._running = True

        async def _loop() -> None:
            logger.info("system_health_monitoring_started", interval_seconds=interval_seconds)
            while self._running:
                try:
                    data = await self.get_latest_health()
                    await bus.publish(EVENT_SYSTEM_METRICS, data)
                except Exception as exc:
                    logger.error("system_health_poll_error", error=str(exc))
                await asyncio.sleep(interval_seconds)

        self._task = asyncio.create_task(_loop())

    def stop_monitoring(self) -> None:
        """Stop background telemetry loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self.resource_monitor.close()
        logger.info("system_health_monitoring_stopped")

    def get_history(self) -> list[dict[str, Any]]:
        """Return cached history for charts."""
        return list(self._history)


# Global singleton instance
system_health = SystemHealthAggregator()
health_aggregator = system_health
