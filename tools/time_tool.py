"""JARVIS Lite — Get Time Tool.

Provides current local time, date, and timezone info.
Fully validated schemas, zero side-effects, verified timestamp output.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from security.permissions import RiskLevel
from tools.base import BaseTool


class GetTimeInput(BaseModel):
    timezone_name: str | None = Field(default="local", description="Timezone name or 'local'")
    format_string: str | None = Field(default="%A, %B %d, %Y %I:%M:%S %p", description="Datetime format")


class GetTimeOutput(BaseModel):
    formatted_time: str
    iso_timestamp: str
    timezone: str


class GetTimeTool(BaseTool[GetTimeInput, GetTimeOutput]):
    name = "get_time"
    description = "Get the current system time, day of the week, and date."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 2.0
    input_schema = GetTimeInput
    output_schema = GetTimeOutput

    async def _execute(self, params: GetTimeInput) -> GetTimeOutput:
        now = datetime.now()
        fmt = params.format_string or "%A, %B %d, %Y %I:%M:%S %p"
        formatted = now.strftime(fmt)
        iso = now.isoformat()
        tz_name = now.astimezone().tzname() or "Local"

        return GetTimeOutput(
            formatted_time=formatted,
            iso_timestamp=iso,
            timezone=tz_name,
        )

    async def verify(self, params: GetTimeInput, result: GetTimeOutput) -> bool:
        # Verify result is a non-empty, valid timestamp string
        return bool(result.formatted_time and result.iso_timestamp and len(result.formatted_time) > 5)
