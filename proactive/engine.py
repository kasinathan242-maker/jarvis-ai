"""JARVIS Lite — Proactive Assistance Engine.

Periodically evaluates system state, hardware metrics, and environment signals.
Never acts without user approval: any suggested action is created as a pending approval ticket
or emitted as an unobtrusive notification.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from core.event_bus import bus
from core.logging import get_logger
from security.approval import approval_manager
from security.permissions import RiskLevel
from system.resource_monitor import resource_monitor

logger = get_logger("proactive.engine")


class ProactiveEngine:
    """Monitors system conditions and surfaces intelligent suggestions with approval gating."""

    def __init__(
        self,
        quiet_hours_start: int = 22,
        quiet_hours_end: int = 7,
    ) -> None:
        self.quiet_hours_start = quiet_hours_start
        self.quiet_hours_end = quiet_hours_end

        # Granular feature toggles
        self.toggles: dict[str, bool] = {
            "battery_warning": True,
            "vram_warning": True,
            "disk_cleanup_suggestion": True,
            "morning_briefing_suggestion": True,
        }

        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._notified_cache: dict[str, float] = {}

    def is_quiet_hours(self, dt: datetime | None = None) -> bool:
        """Return True if the current local time falls within configured quiet hours."""
        now = dt or datetime.now()
        current_hour = now.hour
        if self.quiet_hours_start > self.quiet_hours_end:
            # Over midnight: e.g. 22:00 to 07:00
            return current_hour >= self.quiet_hours_start or current_hour < self.quiet_hours_end
        return self.quiet_hours_start <= current_hour < self.quiet_hours_end

    def set_toggle(self, feature: str, enabled: bool) -> None:
        """Enable or disable a specific proactive trigger."""
        self.toggles[feature] = enabled
        logger.info("proactive_toggle_updated", feature=feature, enabled=enabled)

    async def evaluate_system_state(self) -> list[dict[str, Any]]:
        """Evaluate current system metrics and return triggered suggestions or approval tickets.

        Safety invariant: Any state-modifying action creates an approval ticket, never mutates autonomously.
        """
        suggestions: list[dict[str, Any]] = []
        snap = resource_monitor.get_snapshot()
        is_quiet = self.is_quiet_hours()

        # 1. Battery Health Evaluation
        if self.toggles.get("battery_warning", True) and snap.battery is not None:
            if not snap.battery.is_plugged and snap.battery.percent < 20.0:
                sug = {
                    "type": "battery_low",
                    "severity": "warning",
                    "message": f"Laptop battery is at {snap.battery.percent:.0f}% on battery power. Consider plugging in AC power.",
                    "requires_approval": False,
                    "silenced_in_quiet_hours": is_quiet,
                }
                suggestions.append(sug)
                if not is_quiet:
                    await bus.publish("proactive_alert", sug)

        # 2. GPU VRAM Consumption Evaluation
        if self.toggles.get("vram_warning", True) and snap.gpu is not None:
            if snap.gpu.used_vram_mb > 5200.0:
                sug = {
                    "type": "gpu_vram_critical",
                    "severity": "critical",
                    "message": f"GPU VRAM usage is at {snap.gpu.used_vram_mb:.0f}/{snap.gpu.total_vram_mb:.0f} MB. High risk of OOM.",
                    "requires_approval": False,
                    "silenced_in_quiet_hours": is_quiet,
                }
                suggestions.append(sug)
                if not is_quiet:
                    await bus.publish("proactive_alert", sug)

        # 3. Disk Space & Screenshots Cleanup Suggestion (Gated by Approval Ticket)
        if self.toggles.get("disk_cleanup_suggestion", True):
            screenshots_dir = Path("data/screenshots")
            if screenshots_dir.exists():
                shots = list(screenshots_dir.glob("*.png"))
                if len(shots) > 50:
                    # Never delete automatically! Create an approval ticket for user review
                    ticket = approval_manager.create_ticket(
                        action_type="clean_temporary_screenshots",
                        target_resource=str(screenshots_dir),
                        details={
                            "stale_screenshot_count": len(shots),
                            "rationale": "Free disk space by purging historical visual captures",
                        },
                        risk_level=RiskLevel.CONFIRM,
                    )
                    sug = {
                        "type": "cleanup_approval_needed",
                        "severity": "info",
                        "ticket_id": ticket.ticket_id,
                        "message": f"Found {len(shots)} historical screen captures. Created approval ticket {ticket.ticket_id} to confirm cleanup.",
                        "requires_approval": True,
                        "silenced_in_quiet_hours": is_quiet,
                    }
                    suggestions.append(sug)

        return suggestions

    async def start(self, interval_seconds: float = 60.0) -> None:
        """Start background proactive monitoring loop."""
        if self._running:
            return
        self._running = True

        async def _loop() -> None:
            while self._running:
                try:
                    await self.evaluate_system_state()
                except Exception as exc:
                    logger.warn("proactive_loop_error", error=str(exc))
                await asyncio.sleep(interval_seconds)

        self._task = asyncio.create_task(_loop())
        logger.info("proactive_engine_started", interval_seconds=interval_seconds)

    def stop(self) -> None:
        """Stop background proactive monitoring loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None
        logger.info("proactive_engine_stopped")


proactive_engine = ProactiveEngine()
