"""JARVIS Lite — Phase 3 Tools & Capabilities Unit and Integration Test Suite.

Tests:
- File organizer (dry-run & confirmed execution)
- Batch rename (dry-run & confirmed execution)
- Reminder scheduling & listing
- Daily briefing generation
- Gmail & Google Calendar tools
- Developer copilot tools (repo_read, symbol_search, propose_edit, apply_edit, run_tests, git_status)
- Declarative automation engine workflows
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from automation.engine import automation_engine
from tools.developer import (
    ApplyEditInput,
    ApplyEditTool,
    GitStatusInput,
    GitStatusTool,
    ProposeEditInput,
    ProposeEditTool,
    RepoReadInput,
    RepoReadTool,
    SymbolSearchInput,
    SymbolSearchTool,
)
from tools.google import (
    CreateCalendarEventInput,
    CreateCalendarEventTool,
    DraftEmailReplyInput,
    DraftEmailReplyTool,
    ListCalendarEventsInput,
    ListCalendarEventsTool,
    SearchEmailsInput,
    SearchEmailsTool,
)
from tools.organizer import (
    BatchRenameInput,
    BatchRenameTool,
    FileOrganizerInput,
    FileOrganizerTool,
)
from tools.reminders import (
    DailyBriefingInput,
    DailyBriefingTool,
    ListRemindersInput,
    ListRemindersTool,
    SetReminderInput,
    SetReminderTool,
)


@pytest.mark.asyncio
async def test_file_organizer_dry_run_and_execution() -> None:
    """Test file organizer dry-run preview and confirmed categorization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        # Create test files
        (dir_path / "doc1.pdf").write_text("dummy pdf", encoding="utf-8")
        (dir_path / "img1.png").write_text("dummy png", encoding="utf-8")
        (dir_path / "notes.txt").write_text("dummy notes", encoding="utf-8")

        tool = FileOrganizerTool()

        # 1. Test Dry Run (files MUST NOT be moved)
        dry_res = await tool._execute(FileOrganizerInput(directory_path=str(dir_path), dry_run=True))
        assert dry_res.dry_run is True
        assert dry_res.total_files_evaluated == 3
        assert len(dry_res.moves_planned) == 3
        assert (dir_path / "doc1.pdf").exists()  # Untouched

        # 2. Test Execution
        exec_res = await tool._execute(FileOrganizerInput(directory_path=str(dir_path), dry_run=False))
        assert exec_res.executed_moves_count == 3
        assert (dir_path / "Documents" / "PDFs" / "doc1.pdf").exists()
        assert (dir_path / "Media" / "Images" / "img1.png").exists()
        assert (dir_path / "Documents" / "Text" / "notes.txt").exists()


@pytest.mark.asyncio
async def test_batch_rename_dry_run_and_execution() -> None:
    """Test batch file rename with regex matching and dry-run preview."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        (dir_path / "report_2026_q1.txt").write_text("q1 report", encoding="utf-8")
        (dir_path / "report_2026_q2.txt").write_text("q2 report", encoding="utf-8")

        tool = BatchRenameTool()

        # 1. Dry run preview
        dry_res = await tool._execute(
            BatchRenameInput(
                directory_path=str(dir_path),
                find_pattern=r"report_2026_(q\d)",
                replacement=r"dossier_\1",
                dry_run=True,
            )
        )
        assert dry_res.dry_run is True
        assert dry_res.matches_count == 2
        assert (dir_path / "report_2026_q1.txt").exists()

        # 2. Execution
        exec_res = await tool._execute(
            BatchRenameInput(
                directory_path=str(dir_path),
                find_pattern=r"report_2026_(q\d)",
                replacement=r"dossier_\1",
                dry_run=False,
            )
        )
        assert exec_res.renamed_count == 2
        assert (dir_path / "dossier_q1.txt").exists()
        assert (dir_path / "dossier_q2.txt").exists()


@pytest.mark.asyncio
async def test_reminders_and_daily_briefing() -> None:
    """Test reminder scheduling, listing, and daily executive briefing."""
    set_tool = SetReminderTool()
    res = await set_tool._execute(SetReminderInput(message="Review pull request #12", delay_minutes=15.0))
    assert res.verified is True
    assert res.item_id.startswith("rem_")

    list_tool = ListRemindersTool()
    l_res = await list_tool._execute(ListRemindersInput(active_only=True))
    assert l_res.count >= 1
    assert any("Review pull request" in r.message for r in l_res.reminders)

    briefing_tool = DailyBriefingTool()
    b_res = await briefing_tool._execute(DailyBriefingInput())
    assert b_res.verified is True
    assert len(b_res.briefing_text) > 30


@pytest.mark.asyncio
async def test_google_services_tools() -> None:
    """Test Gmail search, draft creation, and Calendar event creation."""
    # Search emails
    search_tool = SearchEmailsTool()
    s_res = await search_tool._execute(SearchEmailsInput(query="Architecture"))
    assert s_res.count >= 1
    assert "Architecture" in s_res.emails[0].subject

    # Draft reply
    draft_tool = DraftEmailReplyTool()
    d_res = await draft_tool._execute(
        DraftEmailReplyInput(
            to_email="colleague@example.com",
            subject="Status Sync",
            reply_body="Looking good, approved.",
        )
    )
    assert d_res.draft_id.startswith("draft_")
    assert d_res.requires_approval_to_send is True

    # Calendar events
    list_cal_tool = ListCalendarEventsTool()
    c_res = await list_cal_tool._execute(ListCalendarEventsInput())
    assert c_res.verified is True

    create_cal_tool = CreateCalendarEventTool()
    cr_res = await create_cal_tool._execute(
        CreateCalendarEventInput(
            summary="Team Sync Meeting",
            start_iso="2026-10-25T15:00:00Z",
            end_iso="2026-10-25T16:00:00Z",
        )
    )
    assert cr_res.event_id.startswith("evt_")


@pytest.mark.asyncio
async def test_developer_copilot_tools() -> None:
    """Test developer copilot repo_read, symbol_search, propose_edit, apply_edit, git_status."""
    # 1. Repo Read
    read_tool = RepoReadTool()
    r_res = await read_tool._execute(RepoReadInput(file_path="pyproject.toml", start_line=1, end_line=10))
    assert r_res.verified is True
    assert "project" in r_res.content

    # 2. Symbol Search
    search_tool = SymbolSearchTool()
    s_res = await search_tool._execute(SymbolSearchInput(query="ToolRegistry", file_extension=".py"))
    assert s_res.matches_count >= 1

    # 3. Propose Edit & Apply Edit
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as f:
        f.write("def calculate_metric():\n    return 42\n")
        tmp_file = Path(f.name).resolve()

    try:
        # Test propose diff
        propose_tool = ProposeEditTool()
        p_res = await propose_tool._execute(
            ProposeEditInput(
                file_path=str(tmp_file),
                target_content="return 42",
                replacement_content="return 100",
            )
        )
        assert p_res.matched is True
        assert "+    return 100" in p_res.diff

        # Test apply edit
        apply_tool = ApplyEditTool()
        a_res = await apply_tool._execute(
            ApplyEditInput(
                file_path=str(tmp_file),
                target_content="return 42",
                replacement_content="return 100",
            )
        )
        assert a_res.applied is True
        assert "return 100" in tmp_file.read_text(encoding="utf-8")
    finally:
        if tmp_file.exists():
            tmp_file.unlink()

    # 4. Git Status
    git_tool = GitStatusTool()
    g_res = await git_tool._execute(GitStatusInput())
    assert g_res.verified is True
    assert len(g_res.branch) > 0


@pytest.mark.asyncio
async def test_automation_engine_declarative_workflow() -> None:
    """Test declarative automation engine loading and executing workflow."""
    count = automation_engine.load_all()
    assert count >= 1

    # Test executing dry-run
    dry_res = await automation_engine.execute_workflow("Morning Briefing", dry_run=True)
    assert dry_res["status"] == "completed"
    assert dry_res["dry_run"] is True
    assert len(dry_res["steps"]) >= 1

    # Test executing live
    live_res = await automation_engine.execute_workflow("Morning Briefing", dry_run=False)
    assert live_res["status"] == "completed"
    assert live_res["steps"][0]["status"] == "success"
