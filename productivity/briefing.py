"""JARVIS Lite — Daily Executive Briefing Generator.

Aggregates system telemetry, scheduled reminders, calendar events,
and pending task pipelines into an executive audio/text briefing.
"""

from __future__ import annotations

from datetime import datetime

from ai.model_router import router
from core.logging import get_logger
from productivity.scheduler import scheduler
from system.health import health_aggregator
from system.resource_monitor import resource_monitor

logger = get_logger("productivity.briefing")


class DailyBriefingGenerator:
    """Generates an executive daily briefing."""

    async def generate_briefing(self) -> str:
        """Aggregate data and generate morning/daily briefing text."""
        now = datetime.now()
        date_str = now.strftime("%A, %B %d, %Y")
        time_str = now.strftime("%I:%M %p")

        # Telemetry
        snap = resource_monitor.get_snapshot()
        gpu_info = f"GPU: {snap.gpu.gpu_name} ({snap.gpu.used_vram_mb:.0f}/{snap.gpu.total_vram_mb:.0f} MB VRAM)" if snap.gpu else "Integrated GPU"
        battery_info = f"Battery: {snap.battery.percent}% ({'Plugged in' if snap.battery.is_plugged else 'On battery'})" if snap.battery else "AC Power"

        # Reminders
        active_reminders = scheduler.list_items(active_only=True)
        reminders_text = (
            "\n".join(f"- {r.name}" for r in active_reminders[:5])
            if active_reminders
            else "No pending reminders scheduled for today."
        )

        # Health
        alerts = health_aggregator.get_active_alerts()
        alerts_text = (
            f"{len(alerts)} active alert(s): " + ", ".join(a.message for a in alerts[:2])
            if alerts
            else "All hardware operating within normal thresholds."
        )

        prompt = f"""You are JARVIS Lite delivering the daily executive briefing to the user.
Tone: Confident, crisp, polite, professional, concise (under 120 words).

Context:
- Current Time: {time_str} on {date_str}
- Hardware Status: CPU {snap.cpu_percent:.1f}%, RAM {snap.ram_used_mb / 1024:.1f}/{snap.ram_total_mb / 1024:.1f} GB, {gpu_info}, {battery_info}
- Health & Safety: {alerts_text}
- Pending Reminders & Tasks:
{reminders_text}

Deliver the verbal briefing as if speaking directly to the user.
"""

        try:
            return await router.chat(
                messages=[
                    {"role": "system", "content": "You are JARVIS Lite, an executive AI operating system assistant."},
                    {"role": "user", "content": prompt},
                ],
                task_type="chat",
                temperature=0.3,
            )
        except Exception:
            # Deterministic fallback briefing
            return (
                f"Good day. It is {time_str} on {date_str}. "
                f"Your system is running smoothly with CPU at {snap.cpu_percent:.1f}% and RAM at {snap.ram_percent:.1f}%. "
                f"{gpu_info}. {alerts_text} "
                f"You have {len(active_reminders)} scheduled reminder(s). How may I assist you today?"
            )


daily_briefing = DailyBriefingGenerator()
