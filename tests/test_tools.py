"""Tests for Tool Registry, ToolBase contract, and Verified Safe Tools."""

import psutil
import pytest

from security.audit_log import get_recent_audit_records
from tools.applications import OpenApplicationTool
from tools.base import ToolCall
from tools.registry import registry
from tools.system_stats import SystemStatsTool
from tools.time_tool import GetTimeTool


@pytest.mark.asyncio
async def test_get_time_tool() -> None:
    tool = GetTimeTool()
    result = await tool.run({})
    assert result.success is True
    assert result.verification_passed is True
    assert result.data is not None
    assert len(result.data.formatted_time) > 0
    assert result.data.timezone != ""


@pytest.mark.asyncio
async def test_system_stats_tool() -> None:
    tool = SystemStatsTool()
    result = await tool.run({"include_gpu": True})
    assert result.success is True
    assert result.verification_passed is True
    assert result.data is not None
    assert 0.0 <= result.data.cpu_percent <= 100.0
    assert result.data.ram_total_gb > 0.0
    assert result.data.disk_free_gb > 0.0


@pytest.mark.asyncio
async def test_open_application_tool_success_and_verify() -> None:
    """Verify 'open notepad' spawns, verify() confirms it, and clean up process."""
    tool = OpenApplicationTool()
    result = await tool.run({"app_name": "notepad"})

    assert result.success is True
    assert result.verification_passed is True
    assert result.data is not None
    assert result.data.pid is not None
    pid = result.data.pid

    # Clean up spawned test process
    try:
        proc = psutil.Process(pid)
        proc.terminate()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_open_application_tool_blocked_honest_failure() -> None:
    """Verify that when an application launch is blocked, an honest failure is reported."""
    tool = OpenApplicationTool()
    result = await tool.run({"app_name": "notepad", "blocked_for_test": True})

    assert result.success is False
    assert result.verification_passed is False
    assert result.error is not None
    assert "blocked" in result.error.lower()


@pytest.mark.asyncio
async def test_registry_dispatch_and_audit_logging() -> None:
    call = ToolCall(tool_name="get_time", parameters={})
    res = await registry.execute(call)
    assert res.success is True

    # Confirm audit log entry was created
    logs = get_recent_audit_records(limit=5)
    matching = [record for record in logs if record["tool_name"] == "get_time"]
    assert len(matching) > 0
    assert matching[0]["execution_status"] == "SUCCESS"
    assert matching[0]["verification_passed"] is True
