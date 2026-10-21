"""JARVIS Lite — Interactive Security Approval Flow Tests."""

import asyncio

import pytest

from core.errors import SecurityViolationError
from security.approval import ApprovalManager
from security.permissions import RiskLevel


@pytest.mark.asyncio
async def test_approval_flow_ui_authorization() -> None:
    mgr = ApprovalManager()

    async def _resolve_later() -> None:
        await asyncio.sleep(0.05)
        pending = mgr.get_pending()
        assert len(pending) == 1
        mgr.resolve_ticket(pending[0]["request_id"], approved=True, source="ui")

    task = asyncio.create_task(_resolve_later())
    approved = await mgr.request_approval(
        tool_name="delete_file",
        risk_level=RiskLevel.CONFIRM,
        parameters={"file_path": "C:/data/old.txt"},
        description="Delete file",
        timeout_seconds=5.0,
    )
    await task
    assert approved is True


@pytest.mark.asyncio
async def test_approval_flow_user_rejection() -> None:
    mgr = ApprovalManager()

    async def _reject_later() -> None:
        await asyncio.sleep(0.05)
        pending = mgr.get_pending()
        assert len(pending) == 1
        mgr.resolve_ticket(pending[0]["request_id"], approved=False, source="ui")

    task = asyncio.create_task(_reject_later())
    approved = await mgr.request_approval(
        tool_name="delete_file",
        risk_level=RiskLevel.CONFIRM,
        parameters={"file_path": "C:/data/old.txt"},
        description="Delete file",
        timeout_seconds=5.0,
    )
    await task
    assert approved is False


@pytest.mark.asyncio
async def test_approval_flow_voice_confirmation() -> None:
    mgr = ApprovalManager()

    async def _voice_confirm_later() -> None:
        await asyncio.sleep(0.05)
        mgr.resolve_latest_by_voice(approved=True)

    task = asyncio.create_task(_voice_confirm_later())
    approved = await mgr.request_approval(
        tool_name="shell_execute",
        risk_level=RiskLevel.CONFIRM,
        parameters={"command": "powershell script.ps1"},
        description="Run unlisted script",
        timeout_seconds=5.0,
    )
    await task
    assert approved is True


@pytest.mark.asyncio
async def test_forbidden_action_cannot_be_approved() -> None:
    mgr = ApprovalManager()

    with pytest.raises(SecurityViolationError) as exc_info:
        await mgr.request_approval(
            tool_name="shell_execute",
            risk_level=RiskLevel.FORBIDDEN,
            parameters={"command": "format C:"},
            description="Format drive",
        )

    assert "FORBIDDEN and cannot be approved" in str(exc_info.value)
    assert len(mgr.get_pending()) == 0


@pytest.mark.asyncio
async def test_approval_timeout_defaults_to_rejection() -> None:
    mgr = ApprovalManager()

    # Short timeout 0.1s
    approved = await mgr.request_approval(
        tool_name="delete_file",
        risk_level=RiskLevel.CONFIRM,
        parameters={"file_path": "C:/tmp/file.txt"},
        description="Delete file with timeout",
        timeout_seconds=0.1,
    )
    assert approved is False
