"""JARVIS Lite — AsyncIOScheduler & Routine Automation Manager.

Manages reminders, scheduled triggers, and recurring routines with persistent storage.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from core.event_bus import bus
from core.logging import get_logger

logger = get_logger("productivity.scheduler")
SCHEDULES_FILE = Path("data/schedules.json")


@dataclass
class ScheduledItem:
    item_id: str
    name: str
    item_type: str  # "reminder", "routine"
    trigger_type: str  # "date", "interval", "cron"
    trigger_value: str
    created_at: float
    due_at: float | None = None
    completed: bool = False
    metadata: dict[str, Any] | None = None


class ProductivityScheduler:
    """Async scheduler wrapping APScheduler with state persistence."""

    def __init__(self) -> None:
        SCHEDULES_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._scheduler = AsyncIOScheduler()
        self._items: dict[str, ScheduledItem] = {}
        self._is_started = False
        self._load_state()

    def _load_state(self) -> None:
        if SCHEDULES_FILE.exists():
            try:
                data = json.loads(SCHEDULES_FILE.read_text(encoding="utf-8"))
                for k, v in data.items():
                    self._items[k] = ScheduledItem(**v)
            except Exception as exc:
                logger.warn("failed_to_load_schedules_state", error=str(exc))

    def _save_state(self) -> None:
        try:
            serialized = {k: asdict(v) for k, v in self._items.items()}
            SCHEDULES_FILE.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("failed_to_save_schedules_state", error=str(exc))

    def start(self) -> None:
        """Start the async scheduler background loop."""
        if not self._is_started:
            if not self._scheduler.running:
                self._scheduler.start()
            self._is_started = True
            logger.info("productivity_scheduler_started")

    def shutdown(self) -> None:
        """Stop scheduler."""
        if self._is_started and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            self._is_started = False
            logger.info("productivity_scheduler_stopped")

    async def add_reminder(
        self,
        text: str,
        delay_minutes: float | None = None,
        run_at: datetime | None = None,
    ) -> ScheduledItem:
        """Schedule a one-off notification reminder."""
        self.start()
        now = datetime.now(UTC)
        if run_at is not None:
            target_time = run_at
        elif delay_minutes is not None:
            target_time = now + timedelta(minutes=delay_minutes)
        else:
            target_time = now + timedelta(minutes=1.0)

        item_id = f"rem_{int(time.time() * 1000)}"
        due_timestamp = target_time.timestamp()

        item = ScheduledItem(
            item_id=item_id,
            name=text,
            item_type="reminder",
            trigger_type="date",
            trigger_value=target_time.isoformat(),
            created_at=time.time(),
            due_at=due_timestamp,
            completed=False,
        )

        async def _trigger_reminder() -> None:
            logger.info("reminder_due", reminder=text, item_id=item_id)
            item.completed = True
            self._save_state()
            await bus.publish("reminder_alert", {"item_id": item_id, "text": text})

        self._scheduler.add_job(
            _trigger_reminder,
            trigger=DateTrigger(run_date=target_time),
            id=item_id,
            replace_existing=True,
        )

        self._items[item_id] = item
        self._save_state()
        return item

    async def add_routine(
        self,
        name: str,
        cron_expr: str,
        callback: Any,
    ) -> ScheduledItem:
        """Add a recurring routine job via cron expression."""
        self.start()
        item_id = f"rt_{int(time.time() * 1000)}"
        fields = cron_expr.split()
        if len(fields) != 5:
            raise ValueError(f"Cron expression must contain 5 fields: '{cron_expr}'")

        trigger = CronTrigger.from_crontab(cron_expr)

        self._scheduler.add_job(
            callback,
            trigger=trigger,
            id=item_id,
            replace_existing=True,
        )

        item = ScheduledItem(
            item_id=item_id,
            name=name,
            item_type="routine",
            trigger_type="cron",
            trigger_value=cron_expr,
            created_at=time.time(),
            due_at=None,
            completed=False,
        )

        self._items[item_id] = item
        self._save_state()
        return item

    def list_items(self, active_only: bool = True) -> list[ScheduledItem]:
        """List scheduled items."""
        if active_only:
            return [it for it in self._items.values() if not it.completed]
        return list(self._items.values())

    def cancel_item(self, item_id: str) -> bool:
        """Cancel and remove a scheduled item."""
        if item_id in self._items:
            try:
                self._scheduler.remove_job(item_id)
            except Exception:
                pass
            self._items[item_id].completed = True
            self._save_state()
            return True
        return False


scheduler = ProductivityScheduler()
