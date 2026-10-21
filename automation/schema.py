"""JARVIS Lite — Declarative Workflow & Trigger Schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Trigger(BaseModel):
    type: str = Field(description="Trigger type: 'voice', 'hotkey', 'schedule', or 'folder_change'")
    pattern: str = Field(description="Phrase, hotkey sequence, cron pattern, or path to watch")


class WorkflowStep(BaseModel):
    name: str
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    condition: str | None = Field(default=None, description="Optional condition expression")
    retries: int = Field(default=0, ge=0, le=5)
    loop_count: int = Field(default=1, ge=1, le=10)


class WorkflowDefinition(BaseModel):
    name: str
    description: str = ""
    enabled: bool = True
    triggers: list[Trigger] = Field(default_factory=list)
    steps: list[WorkflowStep] = Field(default_factory=list)
