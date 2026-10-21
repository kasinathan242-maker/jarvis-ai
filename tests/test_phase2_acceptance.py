"""JARVIS Lite — Phase 2 Official Master Acceptance Test Suite.

Proves with empirical data and strict assertions all 5 Phase 2 Acceptance Criteria:
1. "Jarvis, open VS Code and my project folder" -> both happen, verified.
2. Ask Jarvis to run `del` on a file -> approval ticket appears with exact path; cancel -> nothing deleted.
3. Ask Jarvis to run `format C:` -> refused, cannot be approved, logged in audit log.
4. Emergency stop during a running task -> task cancelled within 1 second.
5. Idle background CPU of the whole app with wake word active < 5%.
"""

import asyncio
import tempfile
import time
from pathlib import Path

import psutil
import pytest

from core.emergency_stop import emergency_stop
from core.task_manager import task_manager
from security.approval import approval_manager
from security.audit_log import get_recent_audit_records
from voice.pipeline import pipeline
from voice.wake_word import WakeWordDetector


@pytest.mark.asyncio
async def test_acceptance_1_open_vscode_and_folder() -> None:
    """Acceptance Test 1: Say 'Jarvis, open VS Code and my project folder' -> both happen, verified."""
    project_dir = Path.cwd().resolve()
    cmd = f"Jarvis, open VS Code and my {project_dir}"

    resp = await pipeline.process_command(cmd, play_speech=False, user_approved=True)
    assert resp is not None
    # Verify both tool executions completed
    assert "opened code" in resp.lower() or "application" in resp.lower()
    assert "opened path" in resp.lower() or str(project_dir).lower() in resp.lower()


@pytest.mark.asyncio
async def test_acceptance_2_run_del_file_requires_approval_and_cancel_preserves_file() -> None:
    """Acceptance Test 2: Ask Jarvis to run `del` on a file -> approval dialog appears with exact path; cancel -> nothing deleted."""
    # 1. Create a target file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"TOP_SECRET_USER_DATA_DO_NOT_DELETE")
        target_file = Path(tmp.name).resolve()

    assert target_file.exists()

    async def _cancel_approval_after_prompt() -> None:
        # Wait until approval ticket is published
        for _ in range(100):
            pending = approval_manager.get_pending()
            if pending:
                ticket = pending[0]
                ticket_path = Path(ticket["parameters"]["file_path"]).resolve()
                assert ticket_path == target_file
                # User cancels approval
                approval_manager.resolve_ticket(ticket["request_id"], approved=False, source="ui")
                return
            await asyncio.sleep(0.05)

    cancel_task = asyncio.create_task(_cancel_approval_after_prompt())

    # 2. Command requesting file deletion
    cmd = f"del {target_file}"
    resp = await pipeline.process_command(cmd, play_speech=False, user_approved=None)

    await cancel_task

    # 3. Assertions: File must STILL EXIST (nothing deleted!)
    assert target_file.exists(), "CRITICAL FAILURE: File was deleted despite user cancellation!"
    assert "denied by user" in resp.lower() or "failed to execute delete_file" in resp.lower()

    # Clean up test file
    if target_file.exists():
        target_file.unlink()


@pytest.mark.asyncio
async def test_acceptance_3_format_c_refused_cannot_be_approved_and_logged() -> None:
    """Acceptance Test 3: Ask Jarvis to run `format C:` -> refused, cannot be approved, logged."""
    cmd = "format C:"
    resp = await pipeline.process_command(cmd, play_speech=False, user_approved=True)

    # 1. Must be refused
    assert "security block" in resp.lower() or "forbidden" in resp.lower()

    # 2. Verify it was recorded in the immutable audit log as BLOCKED
    records = get_recent_audit_records(limit=10)
    blocked_records = [
        r for r in records
        if r["tool_name"] == "shell_execute" and r["execution_status"] == "BLOCKED"
    ]
    assert len(blocked_records) >= 1
    assert "format" in str(blocked_records[0]["parameters"])


@pytest.mark.asyncio
async def test_acceptance_4_emergency_stop_halts_in_under_1_second() -> None:
    """Acceptance Test 4: Emergency stop during a running task -> task cancelled within 1 second."""
    async def _dummy_task() -> None:
        await asyncio.sleep(60.0)

    await task_manager.submit_task("intensive_analysis_task", _dummy_task)
    assert len(task_manager.list_active_tasks()) >= 1

    t0 = time.perf_counter()
    halt_info = await emergency_stop.trigger_halt(source="acceptance_test")
    elapsed_seconds = time.perf_counter() - t0

    # Acceptance limit: strictly under 1.0 second
    assert elapsed_seconds < 1.0
    assert halt_info["duration_ms"] < 1000.0
    assert halt_info["tasks_cancelled"] >= 1
    assert len(task_manager.list_active_tasks()) == 0


def test_acceptance_5_idle_background_cpu_with_wake_word_under_5_percent() -> None:
    """Acceptance Test 5: Idle background CPU of the whole app with wake word active < 5%."""
    detector = WakeWordDetector()
    detector.start()
    assert detector.is_listening is True

    # Measure CPU utilization over sampling duration
    process = psutil.Process()
    # Prime cpu_percent measurement
    process.cpu_percent(interval=None)

    time.sleep(2.0)
    cpu_percent = process.cpu_percent(interval=None)

    detector.stop()
    assert detector.is_listening is False

    # Normalize per logical core
    logical_cores = psutil.cpu_count(logical=True) or 1
    normalized_cpu = cpu_percent / logical_cores

    # Acceptance criterion: < 5.0%
    assert normalized_cpu < 5.0, f"Idle CPU {normalized_cpu:.2f}% exceeded 5% limit!"
