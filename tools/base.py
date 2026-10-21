"""JARVIS Lite — Tool Base Architecture.

Defines the contract for every executable tool in JARVIS Lite:
- Pydantic input and output schemas
- Explicit risk levels (SAFE, CONFIRM, DANGEROUS, FORBIDDEN)
- Execution timeouts
- Mandatory verify() method to confirm reality after execution
- Integrated audit logging and error reporting
"""

from __future__ import annotations

import abc
import asyncio
import time
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from core.errors import SecurityViolationError
from security.audit_log import log_tool_invocation
from security.permissions import RiskLevel

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class ToolCall(BaseModel):
    """Structured tool invocation proposed by the AI reasoning layer."""
    tool_name: str = Field(description="Exact registered name of the tool to execute")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Input parameters matching tool schema")


class ToolResult(BaseModel, Generic[TOutput]):
    """Standardized result returned by tool execution layer."""
    tool_name: str
    success: bool
    data: TOutput | None = None
    verification_passed: bool = False
    error: str | None = None
    duration_ms: float = 0.0


class BaseTool(abc.ABC, Generic[TInput, TOutput]):
    name: str
    description: str
    risk_level: RiskLevel = RiskLevel.SAFE
    timeout_seconds: float = 5.0
    input_schema: type[TInput]
    output_schema: type[TOutput]

    @abc.abstractmethod
    async def _execute(self, params: TInput) -> TOutput:
        """Core execution logic. Subclasses must implement."""
        raise NotImplementedError

    @abc.abstractmethod
    async def verify(self, params: TInput, result: TOutput) -> bool:
        """Mandatory post-execution reality check.

        Validates that the change actually took effect in the real world (e.g.
        window exists, file created, process running).
        """
        raise NotImplementedError

    def get_risk_level(self, raw_params: dict[str, Any]) -> RiskLevel:
        """Evaluate effective risk level dynamically based on input parameters."""
        return self.risk_level

    async def run(
        self,
        raw_params: dict[str, Any],
        user_approved: bool = True,
    ) -> ToolResult[TOutput]:
        """Validate schemas, execute with timeout, verify reality, and record audit log."""
        start_time = time.monotonic()
        duration_ms = 0.0
        effective_risk = self.get_risk_level(raw_params)

        # 1. Permission check
        if effective_risk == RiskLevel.FORBIDDEN:
            duration_ms = (time.monotonic() - start_time) * 1000
            err_msg = f"Tool '{self.name}' is classified as FORBIDDEN and cannot be run."
            log_tool_invocation(
                tool_name=self.name,
                risk_level=effective_risk.value,
                parameters=raw_params,
                user_approved=False,
                execution_status="BLOCKED",
                verification_passed=False,
                error_message=err_msg,
                duration_ms=duration_ms,
            )
            raise SecurityViolationError(err_msg)

        if effective_risk in (RiskLevel.CONFIRM, RiskLevel.DANGEROUS) and not user_approved:
            duration_ms = (time.monotonic() - start_time) * 1000
            err_msg = f"Execution of '{self.name}' denied by user (Risk: {effective_risk.value})."
            log_tool_invocation(
                tool_name=self.name,
                risk_level=effective_risk.value,
                parameters=raw_params,
                user_approved=False,
                execution_status="REJECTED",
                verification_passed=False,
                error_message=err_msg,
                duration_ms=duration_ms,
            )
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=err_msg,
                duration_ms=duration_ms,
            )

        # 2. Schema validation
        try:
            validated_input = self.input_schema(**raw_params)
        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            err_msg = f"Invalid parameters for tool '{self.name}': {exc}"
            log_tool_invocation(
                tool_name=self.name,
                risk_level=self.risk_level.value,
                parameters=raw_params,
                user_approved=user_approved,
                execution_status="FAILED",
                verification_passed=False,
                error_message=err_msg,
                duration_ms=duration_ms,
            )
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=err_msg,
                duration_ms=duration_ms,
            )

        # 3. Execution with bounded timeout
        try:
            output_obj = await asyncio.wait_for(
                self._execute(validated_input),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            duration_ms = (time.monotonic() - start_time) * 1000
            err_msg = f"Tool '{self.name}' timed out after {self.timeout_seconds}s."
            log_tool_invocation(
                tool_name=self.name,
                risk_level=self.risk_level.value,
                parameters=raw_params,
                user_approved=user_approved,
                execution_status="FAILED",
                verification_passed=False,
                error_message=err_msg,
                duration_ms=duration_ms,
            )
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=err_msg,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            err_msg = f"Tool '{self.name}' execution failed: {exc}"
            log_tool_invocation(
                tool_name=self.name,
                risk_level=self.risk_level.value,
                parameters=raw_params,
                user_approved=user_approved,
                execution_status="FAILED",
                verification_passed=False,
                error_message=err_msg,
                duration_ms=duration_ms,
            )
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=err_msg,
                duration_ms=duration_ms,
            )

        # 4. Mandatory reality check verify()
        try:
            verified = await self.verify(validated_input, output_obj)
        except Exception:
            verified = False

        duration_ms = (time.monotonic() - start_time) * 1000

        status = "SUCCESS" if verified else "FAILED"
        err = None if verified else f"Action completed but reality check verify() failed for '{self.name}'."

        log_tool_invocation(
            tool_name=self.name,
            risk_level=self.risk_level.value,
            parameters=raw_params,
            user_approved=user_approved,
            execution_status=status,
            verification_passed=verified,
            output_preview=str(output_obj.model_dump() if hasattr(output_obj, "model_dump") else output_obj),
            error_message=err,
            duration_ms=duration_ms,
        )

        return ToolResult(
            tool_name=self.name,
            success=verified,
            data=output_obj if verified else None,
            verification_passed=verified,
            error=err,
            duration_ms=duration_ms,
        )
