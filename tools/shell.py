"""JARVIS Lite — Sandboxed Shell Execution Tool.

Enforces Rule 8:
- Allowlisted commands (`git`, `python`, `pip`, `npm`, `node`, `pytest`, `dir`, `where`, `code`) are SAFE.
- Non-allowlisted commands default to CONFIRM.
- Blocklisted commands (`format`, `diskpart`, `reg delete`, `bcdedit`, `del /s`, `rm -rf /`, `C:\\Windows`)
  are FORBIDDEN and cannot be approved.
- Working directory sandboxed. Timeout enforced. Output captured.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.errors import SecurityViolationError, ToolExecutionError
from security.permissions import RiskLevel, classify_shell_command, validate_path_sandbox
from tools.base import BaseTool

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ShellExecutionInput(BaseModel):
    command: str = Field(description="Shell command line to execute")
    working_directory: str | None = Field(
        default=None, description="Working directory (sandboxed to project/user data)"
    )
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0, description="Maximum execution timeout in seconds")


class ShellExecutionOutput(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    verified: bool


class ShellTool(BaseTool[ShellExecutionInput, ShellExecutionOutput]):
    name = "shell_execute"
    description = "Execute a shell command with strict allowlist/blocklist sandboxing."
    risk_level = RiskLevel.SAFE  # Fallback baseline
    timeout_seconds = 35.0
    input_schema = ShellExecutionInput
    output_schema = ShellExecutionOutput

    def get_risk_level(self, raw_params: dict[str, Any]) -> RiskLevel:
        raw_cmd = str(raw_params.get("command", "")).strip()
        risk, _ = classify_shell_command(raw_cmd)
        return risk

    async def _execute(self, params: ShellExecutionInput) -> ShellExecutionOutput:
        raw_cmd = params.command.strip()

        # 1. Enforce Risk Classification
        risk, reason = classify_shell_command(raw_cmd)
        if risk == RiskLevel.FORBIDDEN:
            raise SecurityViolationError(f"Command execution blocked: {reason}")

        # 2. Validate Sandbox Working Directory
        cwd = Path(params.working_directory).resolve() if params.working_directory else PROJECT_ROOT
        validate_path_sandbox(cwd)
        if not cwd.exists() or not cwd.is_dir():
            raise ToolExecutionError(f"Working directory does not exist: '{cwd}'")

        start_time = asyncio.get_event_loop().time()

        try:
            # Use PowerShell on Windows
            proc = await asyncio.create_subprocess_shell(
                raw_cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=params.timeout_seconds,
            )

            elapsed = asyncio.get_event_loop().time() - start_time
            stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            exit_code = proc.returncode if proc.returncode is not None else -1

            return ShellExecutionOutput(
                command=raw_cmd,
                exit_code=exit_code,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_seconds=round(elapsed, 3),
                verified=False,
            )
        except TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            raise ToolExecutionError(f"Shell command timed out after {params.timeout_seconds}s.") from None
        except Exception as exc:
            raise ToolExecutionError(f"Shell process execution failed: {exc}") from exc

    async def verify(self, params: ShellExecutionInput, result: ShellExecutionOutput) -> bool:
        """Verify the shell process terminated and returned a valid exit code."""
        verified = isinstance(result.exit_code, int) and result.duration_seconds >= 0.0
        result.verified = verified
        return verified
