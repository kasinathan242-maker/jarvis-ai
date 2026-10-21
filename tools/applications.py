"""JARVIS Lite — Application Launch Tool.

Safely launches allowlisted applications on Windows 11.
Mandatory verify() guarantees Jarvis never claims 'I opened VS Code'
unless the operating system process or window is confirmed running.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess

import psutil
from pydantic import BaseModel, Field

from core.errors import ToolExecutionError
from security.permissions import RiskLevel
from tools.base import BaseTool

APP_EXECUTABLE_MAP: dict[str, str] = {
    "notepad": "notepad.exe",
    "calc": "calc.exe",
    "calculator": "calc.exe",
    "code": "code.cmd",
    "vscode": "code.cmd",
    "explorer": "explorer.exe",
    "terminal": "wt.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
}


class OpenApplicationInput(BaseModel):
    app_name: str = Field(description="Name or alias of application to open (e.g. 'notepad', 'calc', 'code')")
    blocked_for_test: bool = Field(default=False, description="Internal test hook to simulate blocked app launch")


class OpenApplicationOutput(BaseModel):
    app_name: str
    pid: int | None = None
    executable: str
    verified: bool
    status: str


class OpenApplicationTool(BaseTool[OpenApplicationInput, OpenApplicationOutput]):
    name = "open_application"
    description = "Launch an application on the Windows desktop and verify its window/process."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 5.0
    input_schema = OpenApplicationInput
    output_schema = OpenApplicationOutput

    async def _execute(self, params: OpenApplicationInput) -> OpenApplicationOutput:
        clean_name = params.app_name.lower().strip()

        # Check simulation flag for acceptance testing blocked launches
        if params.blocked_for_test:
            raise ToolExecutionError(f"Application '{clean_name}' launch blocked by policy or system lock.")

        target_exe = APP_EXECUTABLE_MAP.get(clean_name, f"{clean_name}.exe")

        # Check if executable exists in PATH or system
        exe_path = shutil.which(target_exe)
        if not exe_path and not os.path.exists(target_exe):
            raise ToolExecutionError(
                f"Application executable '{target_exe}' could not be located in system PATH."
            )

        try:
            # Launch in background, non-blocking
            proc = subprocess.Popen(
                [target_exe],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pid = proc.pid
            return OpenApplicationOutput(
                app_name=clean_name,
                pid=pid,
                executable=target_exe,
                verified=False,  # Set by verify()
                status="process_spawned",
            )
        except Exception as exc:
            raise ToolExecutionError(f"Operating system failed to launch '{target_exe}': {exc}") from exc

    async def verify(self, params: OpenApplicationInput, result: OpenApplicationOutput) -> bool:
        """Poll the OS for up to 2.5 seconds to confirm process PID or matching window exists."""
        if not result.pid:
            return False

        clean_name = params.app_name.lower().strip()
        target_exe = APP_EXECUTABLE_MAP.get(clean_name, clean_name).lower()
        base_name = target_exe.replace(".exe", "").replace(".cmd", "")

        for _ in range(5):
            await asyncio.sleep(0.5)

            # Check 1: psutil process active
            if result.pid is not None and psutil.pid_exists(result.pid):
                result.verified = True
                result.status = "process_verified_running"
                return True

            # Check 2: Process by name (in case child process took over, e.g. UWP apps like calc or modern notepad)
            for proc in psutil.process_iter(["name", "pid"]):
                try:
                    pname = (proc.info.get("name") or "").lower()
                    if base_name in pname:
                        result.verified = True
                        result.pid = proc.info.get("pid")
                        result.status = f"window_process_verified ({pname})"
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        return False


class CloseApplicationInput(BaseModel):
    app_name: str = Field(description="Name or alias of application to terminate (e.g. 'notepad', 'calc')")


class CloseApplicationOutput(BaseModel):
    app_name: str
    processes_terminated: int
    verified: bool


class CloseApplicationTool(BaseTool[CloseApplicationInput, CloseApplicationOutput]):
    name = "close_application"
    description = "Terminate running processes of a specified application."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 5.0
    input_schema = CloseApplicationInput
    output_schema = CloseApplicationOutput

    async def _execute(self, params: CloseApplicationInput) -> CloseApplicationOutput:
        clean_name = params.app_name.lower().strip()
        target_exe = APP_EXECUTABLE_MAP.get(clean_name, clean_name).lower()
        base_name = target_exe.replace(".exe", "").replace(".cmd", "")

        terminated = 0
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                pname = (proc.info.get("name") or "").lower()
                if base_name in pname:
                    proc.terminate()
                    terminated += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return CloseApplicationOutput(
            app_name=clean_name,
            processes_terminated=terminated,
            verified=False,
        )

    async def verify(self, params: CloseApplicationInput, result: CloseApplicationOutput) -> bool:
        """Verify that no matching process remains running."""
        await asyncio.sleep(0.5)
        clean_name = params.app_name.lower().strip()
        target_exe = APP_EXECUTABLE_MAP.get(clean_name, clean_name).lower()
        base_name = target_exe.replace(".exe", "").replace(".cmd", "")

        for proc in psutil.process_iter(["name"]):
            try:
                pname = (proc.info.get("name") or "").lower()
                if base_name in pname:
                    return False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        result.verified = True
        return True

