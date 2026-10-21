"""JARVIS Lite — Reminder and Briefing Tools.

Exposes reminder scheduling, pending reminder listing, and executive daily briefings
to the agent tool catalog.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from productivity.briefing import daily_briefing
from productivity.scheduler import scheduler
from security.permissions import RiskLevel
from tools.base import BaseTool


class SetReminderInput(BaseModel):
    message: str = Field(description="Reminder message content")
    delay_minutes: float = Field(default=5.0, ge=0.5, le=1440.0, description="Delay in minutes until reminder triggers")


class SetReminderOutput(BaseModel):
    item_id: str
    message: str
    due_at_iso: str
    verified: bool


class SetReminderTool(BaseTool[SetReminderInput, SetReminderOutput]):
    name = "set_reminder"
    description = "Schedule a reminder that alerts the user after a specified number of minutes."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 10.0
    input_schema = SetReminderInput
    output_schema = SetReminderOutput

    async def _execute(self, params: SetReminderInput) -> SetReminderOutput:
        item = await scheduler.add_reminder(text=params.message, delay_minutes=params.delay_minutes)
        return SetReminderOutput(
            item_id=item.item_id,
            message=item.name,
            due_at_iso=item.trigger_value,
            verified=True,
        )

    async def verify(self, params: SetReminderInput, result: SetReminderOutput) -> bool:
        items = scheduler.list_items(active_only=True)
        found = any(it.item_id == result.item_id for it in items)
        result.verified = found
        return found


class ListRemindersInput(BaseModel):
    active_only: bool = Field(default=True, description="Only list uncompleted reminders")


class ReminderItemSchema(BaseModel):
    item_id: str
    message: str
    due_at: str | None


class ListRemindersOutput(BaseModel):
    count: int
    reminders: list[ReminderItemSchema]
    verified: bool


class ListRemindersTool(BaseTool[ListRemindersInput, ListRemindersOutput]):
    name = "list_reminders"
    description = "List all active or completed scheduled reminders."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 10.0
    input_schema = ListRemindersInput
    output_schema = ListRemindersOutput

    async def _execute(self, params: ListRemindersInput) -> ListRemindersOutput:
        items = scheduler.list_items(active_only=params.active_only)
        schemas = [
            ReminderItemSchema(item_id=it.item_id, message=it.name, due_at=it.trigger_value)
            for it in items
        ]
        return ListRemindersOutput(
            count=len(schemas),
            reminders=schemas,
            verified=True,
        )

    async def verify(self, params: ListRemindersInput, result: ListRemindersOutput) -> bool:
        result.verified = True
        return True


class DailyBriefingInput(BaseModel):
    pass


class DailyBriefingOutput(BaseModel):
    briefing_text: str
    verified: bool


class DailyBriefingTool(BaseTool[DailyBriefingInput, DailyBriefingOutput]):
    name = "daily_briefing"
    description = "Generate an executive daily audio and text briefing summarizing system status, schedule, and tasks."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 20.0
    input_schema = DailyBriefingInput
    output_schema = DailyBriefingOutput

    async def _execute(self, params: DailyBriefingInput) -> DailyBriefingOutput:
        text = await daily_briefing.generate_briefing()
        return DailyBriefingOutput(
            briefing_text=text,
            verified=bool(len(text) > 20),
        )

    async def verify(self, params: DailyBriefingInput, result: DailyBriefingOutput) -> bool:
        verified = bool(len(result.briefing_text) > 20)
        result.verified = verified
        return verified
