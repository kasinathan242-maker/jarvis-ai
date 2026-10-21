"""JARVIS Lite — Phase 2 Desktop Tools Verification Tests."""

import tempfile
from pathlib import Path

import pytest

from core.errors import SecurityViolationError
from security.permissions import RiskLevel
from tools.clipboard import ClipboardReadTool, ClipboardWriteTool
from tools.files import DeleteFileTool, SearchFilesTool
from tools.screenshot import ScreenshotTool
from tools.shell import ShellTool
from tools.windows import ListWindowsTool


@pytest.mark.asyncio
async def test_clipboard_read_write_verification() -> None:
    write_tool = ClipboardWriteTool()
    read_tool = ClipboardReadTool()

    test_content = "JARVIS_TEST_CLIPBOARD_SECRET_42"
    write_res = await write_tool.run({"text": test_content}, user_approved=True)

    assert write_res.success is True
    assert write_res.verification_passed is True

    read_res = await read_tool.run({}, user_approved=True)
    assert read_res.success is True
    assert read_res.verification_passed is True
    assert read_res.data is not None
    assert read_res.data.content == test_content


@pytest.mark.asyncio
async def test_screenshot_tool_creates_and_verifies_png() -> None:
    tool = ScreenshotTool()
    res = await tool.run({}, user_approved=True)

    assert res.success is True
    assert res.verification_passed is True
    assert res.data is not None
    saved_path = Path(res.data.file_path)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 1000  # Valid PNG image


@pytest.mark.asyncio
async def test_list_windows_tool() -> None:
    tool = ListWindowsTool()
    res = await tool.run({"visible_only": False}, user_approved=True)

    assert res.success is True
    assert res.verification_passed is True
    assert res.data is not None
    assert res.data.count >= 0


@pytest.mark.asyncio
async def test_search_files_tool() -> None:
    tool = SearchFilesTool()
    res = await tool.run(
        {"directory": ".", "pattern": "*.py", "max_results": 5},
        user_approved=True,
    )

    assert res.success is True
    assert res.verification_passed is True
    assert res.data is not None
    assert res.data.total_found > 0


@pytest.mark.asyncio
async def test_delete_file_tool_requires_confirmation_and_deletes() -> None:
    # 1. Create a temporary file to delete
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"delete me")
        tmp_path = Path(f.name)

    assert tmp_path.exists()

    tool = DeleteFileTool()
    assert tool.risk_level == RiskLevel.CONFIRM

    # Without user approval -> execution rejected, file stays intact
    res_rejected = await tool.run({"file_path": str(tmp_path)}, user_approved=False)
    assert res_rejected.success is False
    assert tmp_path.exists()  # Nothing deleted!

    # With user approval -> file deleted, verify confirms removal from disk
    res_approved = await tool.run({"file_path": str(tmp_path)}, user_approved=True)
    assert res_approved.success is True
    assert res_approved.verification_passed is True
    assert not tmp_path.exists()  # Verified removed!


@pytest.mark.asyncio
async def test_shell_tool_allowlisted_command() -> None:
    tool = ShellTool()
    # 'dir' or 'python --version' are in safe allowlist
    res = await tool.run({"command": "python --version"}, user_approved=True)

    assert res.success is True
    assert res.verification_passed is True
    assert res.data is not None
    assert res.data.exit_code == 0
    assert "Python" in res.data.stdout or "Python" in res.data.stderr


@pytest.mark.asyncio
async def test_shell_tool_forbidden_command_refused() -> None:
    tool = ShellTool()
    # 'format C:' is FORBIDDEN and cannot be approved
    with pytest.raises(SecurityViolationError) as exc_info:
        await tool.run({"command": "format C:"}, user_approved=True)

    assert "FORBIDDEN" in str(exc_info.value) or "contains forbidden pattern" in str(exc_info.value)
