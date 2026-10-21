"""JARVIS Lite — File Organizer and Batch Renaming Tools.

Organizes directories by file extension or modification date with mandatory dry-run preview,
path sandboxing, and confirmation verification.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.errors import ToolExecutionError
from security.permissions import RiskLevel, validate_path_sandbox
from tools.base import BaseTool

EXTENSION_CATEGORIES: dict[str, str] = {
    ".pdf": "Documents/PDFs",
    ".docx": "Documents/Word",
    ".doc": "Documents/Word",
    ".txt": "Documents/Text",
    ".md": "Documents/Markdown",
    ".csv": "Data/Spreadsheets",
    ".xlsx": "Data/Spreadsheets",
    ".png": "Media/Images",
    ".jpg": "Media/Images",
    ".jpeg": "Media/Images",
    ".gif": "Media/Images",
    ".mp4": "Media/Videos",
    ".mkv": "Media/Videos",
    ".mp3": "Media/Audio",
    ".wav": "Media/Audio",
    ".zip": "Archives",
    ".tar": "Archives",
    ".gz": "Archives",
    ".py": "Code/Python",
    ".js": "Code/JavaScript",
    ".ts": "Code/TypeScript",
    ".html": "Code/Web",
    ".css": "Code/Web",
}


class PlannedFileMove(BaseModel):
    source: str
    destination: str
    category: str


class FileOrganizerInput(BaseModel):
    directory_path: str = Field(description="Directory to organize")
    rule_type: str = Field(default="extension", description="'extension' or 'date'")
    dry_run: bool = Field(default=True, description="When true, only previews planned moves without modifying files")


class FileOrganizerOutput(BaseModel):
    directory: str
    dry_run: bool
    total_files_evaluated: int
    moves_planned: list[PlannedFileMove]
    executed_moves_count: int
    verified: bool


class FileOrganizerTool(BaseTool[FileOrganizerInput, FileOrganizerOutput]):
    name = "organize_files"
    description = "Organize a directory into categorized subfolders by file type or date. Defaults to dry-run."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 60.0
    input_schema = FileOrganizerInput
    output_schema = FileOrganizerOutput

    def get_risk_level(self, raw_params: dict[str, Any]) -> RiskLevel:
        is_dry_run = raw_params.get("dry_run", True)
        if not is_dry_run:
            return RiskLevel.CONFIRM
        return RiskLevel.SAFE

    async def _execute(self, params: FileOrganizerInput) -> FileOrganizerOutput:
        target_dir = Path(params.directory_path).resolve()
        validate_path_sandbox(target_dir)

        if not target_dir.exists() or not target_dir.is_dir():
            raise ToolExecutionError(f"Target directory does not exist or is not a directory: {target_dir}")

        files = [f for f in target_dir.iterdir() if f.is_file()]
        planned_moves: list[PlannedFileMove] = []

        for f in files:
            ext = f.suffix.lower()
            if params.rule_type == "date":
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                sub_dir = mtime.strftime("%Y-%m")
                category = f"Date/{sub_dir}"
            else:
                category = EXTENSION_CATEGORIES.get(ext, "Miscellaneous")

            dest_folder = target_dir / category
            dest_file = dest_folder / f.name

            planned_moves.append(
                PlannedFileMove(
                    source=str(f),
                    destination=str(dest_file),
                    category=category,
                )
            )

        executed_count = 0
        if not params.dry_run:
            for move in planned_moves:
                dest_path = Path(move.destination)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(move.source, str(dest_path))
                executed_count += 1

        return FileOrganizerOutput(
            directory=str(target_dir),
            dry_run=params.dry_run,
            total_files_evaluated=len(files),
            moves_planned=planned_moves,
            executed_moves_count=executed_count,
            verified=True,
        )

    async def verify(self, params: FileOrganizerInput, result: FileOrganizerOutput) -> bool:
        if params.dry_run:
            result.verified = True
            return True
        # Verify moved files actually exist in destination
        all_exist = all(Path(m.destination).exists() for m in result.moves_planned)
        result.verified = all_exist
        return all_exist


class BatchRenameInput(BaseModel):
    directory_path: str = Field(description="Directory containing files to rename")
    find_pattern: str = Field(description="Regex or substring to match in file names")
    replacement: str = Field(description="String to replace matches with")
    dry_run: bool = Field(default=True, description="When true, previews changes without renaming")


class RenamePreview(BaseModel):
    old_name: str
    new_name: str
    path: str


class BatchRenameOutput(BaseModel):
    directory: str
    dry_run: bool
    matches_count: int
    renames: list[RenamePreview]
    renamed_count: int
    verified: bool


class BatchRenameTool(BaseTool[BatchRenameInput, BatchRenameOutput]):
    name = "batch_rename"
    description = "Batch rename files matching a regex pattern with dry-run verification."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 30.0
    input_schema = BatchRenameInput
    output_schema = BatchRenameOutput

    def get_risk_level(self, raw_params: dict[str, Any]) -> RiskLevel:
        is_dry_run = raw_params.get("dry_run", True)
        if not is_dry_run:
            return RiskLevel.CONFIRM
        return RiskLevel.SAFE

    async def _execute(self, params: BatchRenameInput) -> BatchRenameOutput:
        target_dir = Path(params.directory_path).resolve()
        validate_path_sandbox(target_dir)

        if not target_dir.exists() or not target_dir.is_dir():
            raise ToolExecutionError(f"Directory not found: {target_dir}")

        pattern = re.compile(params.find_pattern)
        files = [f for f in target_dir.iterdir() if f.is_file()]
        previews: list[RenamePreview] = []

        for f in files:
            if pattern.search(f.name):
                new_name = pattern.sub(params.replacement, f.name)
                if new_name != f.name:
                    previews.append(
                        RenamePreview(
                            old_name=f.name,
                            new_name=new_name,
                            path=str(f),
                        )
                    )

        renamed_count = 0
        if not params.dry_run:
            for p in previews:
                old_path = Path(p.path)
                new_path = old_path.parent / p.new_name
                old_path.rename(new_path)
                renamed_count += 1

        return BatchRenameOutput(
            directory=str(target_dir),
            dry_run=params.dry_run,
            matches_count=len(previews),
            renames=previews,
            renamed_count=renamed_count,
            verified=True,
        )

    async def verify(self, params: BatchRenameInput, result: BatchRenameOutput) -> bool:
        if params.dry_run:
            result.verified = True
            return True
        target_dir = Path(params.directory_path)
        all_renamed = all((target_dir / p.new_name).exists() for p in result.renames)
        result.verified = all_renamed
        return all_renamed
