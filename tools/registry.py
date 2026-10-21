"""JARVIS Lite — Tool Registry & Dispatch Catalog.

Central registry managing tool registration, schema introspection,
permission gating, and verified dispatch.
"""

from __future__ import annotations

from typing import Any

from core.errors import ToolNotFoundError
from security.permissions import RiskLevel
from tools.applications import CloseApplicationTool, OpenApplicationTool
from tools.base import BaseTool, ToolCall, ToolResult
from tools.browser import WebBrowseTool, WebSearchResearchTool
from tools.clipboard import ClipboardReadTool, ClipboardWriteTool
from tools.developer import (
    ApplyEditTool,
    GitCommitTool,
    GitStatusTool,
    ProposeEditTool,
    RepoReadTool,
    RunProjectTestsTool,
    SymbolSearchTool,
)
from tools.files import DeleteFileTool, OpenPathTool, SearchFilesTool
from tools.google import (
    CreateCalendarEventTool,
    DraftEmailReplyTool,
    ListCalendarEventsTool,
    SearchEmailsTool,
    SendEmailTool,
)
from tools.organizer import BatchRenameTool, FileOrganizerTool
from tools.reminders import DailyBriefingTool, ListRemindersTool, SetReminderTool
from tools.screenshot import ScreenshotTool
from tools.shell import ShellTool
from tools.system_stats import SystemStatsTool
from tools.time_tool import GetTimeTool
from tools.vision import AnalyzeScreenTool
from tools.windows import ListWindowsTool, WindowManagementTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool[Any, Any]] = {}

        # Register Phase 1 tools
        self.register(GetTimeTool())
        self.register(OpenApplicationTool())
        self.register(SystemStatsTool())

        # Register Phase 2 Desktop Intelligence tools
        self.register(CloseApplicationTool())
        self.register(WindowManagementTool())
        self.register(ListWindowsTool())
        self.register(ClipboardReadTool())
        self.register(ClipboardWriteTool())
        self.register(ScreenshotTool())
        self.register(OpenPathTool())
        self.register(SearchFilesTool())
        self.register(DeleteFileTool())
        self.register(ShellTool())

        # Register Phase 3 Productivity & Agent tools
        self.register(WebBrowseTool())
        self.register(WebSearchResearchTool())
        self.register(FileOrganizerTool())
        self.register(BatchRenameTool())
        self.register(SetReminderTool())
        self.register(ListRemindersTool())
        self.register(DailyBriefingTool())
        self.register(SearchEmailsTool())
        self.register(DraftEmailReplyTool())
        self.register(SendEmailTool())
        self.register(ListCalendarEventsTool())
        self.register(CreateCalendarEventTool())
        self.register(RepoReadTool())
        self.register(SymbolSearchTool())
        self.register(ProposeEditTool())
        self.register(ApplyEditTool())
        self.register(RunProjectTestsTool())
        self.register(GitStatusTool())
        self.register(GitCommitTool())

        # Register Phase 4 Vision & Advanced tools
        self.register(AnalyzeScreenTool())

    def register(self, tool: BaseTool[Any, Any]) -> None:
        """Register a tool instance in the catalog."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool[Any, Any]:
        """Look up tool by exact name."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' is not registered in the system.")
        return self._tools[name]

    def list_tools(self) -> list[dict[str, Any]]:
        """Return catalog of available tools with their documentation and schemas."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "risk_level": tool.risk_level.value,
                "timeout_seconds": tool.timeout_seconds,
                "parameters_schema": tool.input_schema.model_json_schema(),
            }
            for tool in self._tools.values()
        ]

    async def execute(
        self,
        call: ToolCall,
        user_approved: bool | None = None,
    ) -> ToolResult[Any]:
        """Execute a tool call with permission enforcement, interactive approval, timeout, and verification."""
        from security.approval import approval_manager

        tool = self.get_tool(call.tool_name)
        effective_risk = tool.get_risk_level(call.parameters)

        if effective_risk == RiskLevel.FORBIDDEN:
            # Cannot be approved under any circumstances
            return await tool.run(call.parameters, user_approved=False)

        if effective_risk in (RiskLevel.CONFIRM, RiskLevel.DANGEROUS) and user_approved is None:
            # Request explicit user authorization via UI dialog or voice
            desc = f"Authorize action '{call.tool_name}' ({effective_risk.value}) with parameters: {call.parameters}"
            approved = await approval_manager.request_approval(
                tool_name=call.tool_name,
                risk_level=effective_risk,
                parameters=call.parameters,
                description=desc,
            )
            user_approved = approved

        is_approved = True if user_approved is None else user_approved
        return await tool.run(call.parameters, user_approved=is_approved)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    async def dispatch(
        self,
        name: str,
        raw_params: dict[str, Any],
        user_approved: bool | None = None,
    ) -> ToolResult[Any]:
        """Dispatch tool by name and parameters directly."""
        call = ToolCall(tool_name=name, parameters=raw_params)
        return await self.execute(call, user_approved=user_approved)


# Global singleton registry
registry = ToolRegistry()
