"""JARVIS Lite — Intent Schemas.

Pydantic data models for structured intent classification.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class IntentType(enum.StrEnum):
    GET_TIME = "get_time"
    OPEN_APPLICATION = "open_application"
    SYSTEM_STATS = "system_stats"
    CHAT = "chat"
    REASONING = "reasoning"
    MEMORY_STORE = "memory_store"
    MEMORY_QUERY = "memory_query"
    GREETING = "greeting"
    EMERGENCY_STOP = "emergency_stop"
    ANALYZE_SCREEN = "analyze_screen"
    EXIT = "exit"
    UNKNOWN = "unknown"


class Intent(BaseModel):
    type: IntentType = Field(description="Categorized intent type")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence score")
    slots: dict[str, Any] = Field(default_factory=dict, description="Extracted entity slots (e.g. app_name, timezone)")
    raw_query: str = Field(description="Original user text query")
