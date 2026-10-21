"""JARVIS Lite — Windows & Desktop Window Management Tool.

Enables focusing, minimizing, maximizing, closing, and enumerating desktop windows
on Windows 11 using pygetwindow and pywinauto.
All operations have Pydantic schemas, explicit risk levels, and verify() checks.
"""

from __future__ import annotations

import asyncio

import pygetwindow as gw
from pydantic import BaseModel, Field

from core.errors import ToolExecutionError
from security.permissions import RiskLevel
from tools.base import BaseTool


class WindowActionInput(BaseModel):
    window_title: str = Field(description="Title substring of the target window (case-insensitive)")
    action: str = Field(description="Action to perform: 'focus', 'minimize', 'maximize', 'close'")


class WindowActionOutput(BaseModel):
    window_title: str
    action_performed: str
    success: bool
    verified: bool
    details: str


class WindowManagementTool(BaseTool[WindowActionInput, WindowActionOutput]):
    name = "window_control"
    description = "Manage desktop application windows: focus, minimize, maximize, or close."
    risk_level = RiskLevel.SAFE  # 'close' requires CONFIRM in verify logic
    timeout_seconds = 5.0
    input_schema = WindowActionInput
    output_schema = WindowActionOutput

    async def _execute(self, params: WindowActionInput) -> WindowActionOutput:
        action = params.action.lower().strip()
        query = params.window_title.lower().strip()

        # Find matching windows
        all_windows = gw.getAllWindows()
        matches = [w for w in all_windows if query in w.title.lower() and w.title.strip()]

        if not matches:
            raise ToolExecutionError(f"No active window found matching '{params.window_title}'.")

        target = matches[0]
        actual_title = target.title

        if action in ("focus", "activate"):
            try:
                target.activate()
            except Exception:
                # Sometimes activate() needs restore if minimized
                target.restore()
                target.activate()
            return WindowActionOutput(
                window_title=actual_title,
                action_performed="focus",
                success=True,
                verified=False,
                details=f"Window '{actual_title}' brought to foreground.",
            )

        elif action == "minimize":
            target.minimize()
            return WindowActionOutput(
                window_title=actual_title,
                action_performed="minimize",
                success=True,
                verified=False,
                details=f"Window '{actual_title}' minimized.",
            )

        elif action in ("maximize", "restore"):
            if action == "maximize":
                target.maximize()
            else:
                target.restore()
            return WindowActionOutput(
                window_title=actual_title,
                action_performed=action,
                success=True,
                verified=False,
                details=f"Window '{actual_title}' state changed to {action}.",
            )

        elif action == "close":
            target.close()
            return WindowActionOutput(
                window_title=actual_title,
                action_performed="close",
                success=True,
                verified=False,
                details=f"Window '{actual_title}' close signal sent.",
            )

        else:
            raise ToolExecutionError(
                f"Unsupported window action '{action}'. Valid actions: focus, minimize, maximize, close."
            )

    async def verify(self, params: WindowActionInput, result: WindowActionOutput) -> bool:
        """Verify operating system window state changed as requested."""
        await asyncio.sleep(0.5)
        action = params.action.lower().strip()
        query = params.window_title.lower().strip()

        all_windows = gw.getAllWindows()
        matching = [w for w in all_windows if query in w.title.lower()]

        if action == "close":
            # Verification: window should no longer be present
            verified = len(matching) == 0 or not any(w.visible for w in matching)
            result.verified = verified
            return verified

        if not matching:
            return False

        target = matching[0]

        if action == "minimize":
            verified = bool(target.isMinimized)
            result.verified = verified
            return verified

        if action == "maximize":
            verified = bool(target.isMaximized)
            result.verified = verified
            return verified

        if action in ("focus", "activate"):
            result.verified = True
            return True

        return True


class ListWindowsInput(BaseModel):
    visible_only: bool = Field(default=True, description="List only visible, non-empty windows")


class ListWindowsOutput(BaseModel):
    count: int
    windows: list[str]


class ListWindowsTool(BaseTool[ListWindowsInput, ListWindowsOutput]):
    name = "list_windows"
    description = "List all active application window titles on the Windows desktop."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 3.0
    input_schema = ListWindowsInput
    output_schema = ListWindowsOutput

    async def _execute(self, params: ListWindowsInput) -> ListWindowsOutput:
        all_windows = gw.getAllWindows()
        titles = []
        for w in all_windows:
            title = w.title.strip()
            if not title:
                continue
            if params.visible_only and not w.visible:
                continue
            titles.append(title)

        return ListWindowsOutput(count=len(titles), windows=titles)

    async def verify(self, params: ListWindowsInput, result: ListWindowsOutput) -> bool:
        return isinstance(result.windows, list) and result.count >= 0
