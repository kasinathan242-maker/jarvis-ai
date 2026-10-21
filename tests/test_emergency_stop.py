"""JARVIS Lite — Emergency Stop Verification Tests."""

import asyncio
import time

import pytest

from core.emergency_stop import emergency_stop
from core.task_manager import task_manager
from security.approval import approval_manager
from security.permissions import RiskLevel


@pytest.mark.asyncio
async def test_emergency_stop_halts_running_tasks_under_1_second() -> None:
    # 1. Start two background tasks
    async def _long_task() -> None:
        await asyncio.sleep(10.0)

    await task_manager.submit_task("background_job_1", _long_task)
    await task_manager.submit_task("background_job_2", _long_task)

    assert len(task_manager.list_active_tasks()) >= 2

    # 2. Trigger Emergency Stop
    t0 = time.perf_counter()
    halt_info = await emergency_stop.trigger_halt(source="test")
    elapsed_seconds = time.perf_counter() - t0

    # Strict Acceptance Criterion: Task cancelled within 1 second!
    assert elapsed_seconds < 1.0
    assert halt_info["duration_ms"] < 1000.0
    assert halt_info["tasks_cancelled"] >= 2
    assert len(task_manager.list_active_tasks()) == 0


@pytest.mark.asyncio
async def test_emergency_stop_cancels_pending_approvals() -> None:
    # Start pending approval
    approval_task = asyncio.create_task(
        approval_manager.request_approval(
            tool_name="delete_file",
            risk_level=RiskLevel.CONFIRM,
            parameters={"file_path": "C:/important/file.txt"},
            description="Delete file",
            timeout_seconds=30.0,
        )
    )
    await asyncio.sleep(0.02)
    assert len(approval_manager.get_pending()) == 1

    # Trigger halt
    halt_info = await emergency_stop.trigger_halt(source="test_approval")
    assert halt_info["approvals_cancelled"] >= 1

    approved = await approval_task
    assert approved is False  # Denied automatically!
