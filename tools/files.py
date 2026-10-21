"""JARVIS Lite — File & Directory Operations Tool.

Enforces Rule 6 & Rule 7:
- Open files and folders with default system handler.
- File search by name, extension, and content.
- Deletion or destructive operations require explicit user approval (CONFIRM).
- Sandboxed to prevent tampering with system directories (C:\\Windows).
"""

from __future__ import annotations

import os
from datetime import UTC
from pathlib import Path

from pydantic import BaseModel, Field

from core.errors import ToolExecutionError
from security.permissions import RiskLevel, validate_path_sandbox
from tools.base import BaseTool


class OpenPathInput(BaseModel):
    path: str = Field(description="Absolute or relative path to file or folder to open")


class OpenPathOutput(BaseModel):
    resolved_path: str
    is_directory: bool
    opened: bool
    verified: bool


class OpenPathTool(BaseTool[OpenPathInput, OpenPathOutput]):
    name = "open_path"
    description = "Open a file or directory in the default Windows application or Explorer."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 5.0
    input_schema = OpenPathInput
    output_schema = OpenPathOutput

    async def _execute(self, params: OpenPathInput) -> OpenPathOutput:
        target = Path(params.path).resolve()
        validate_path_sandbox(target)

        if not target.exists():
            raise ToolExecutionError(f"Target path does not exist: '{target}'")

        try:
            # os.startfile opens in default application or explorer on Windows
            os.startfile(str(target))
            return OpenPathOutput(
                resolved_path=str(target),
                is_directory=target.is_dir(),
                opened=True,
                verified=False,
            )
        except Exception as exc:
            raise ToolExecutionError(f"Failed to open '{target}': {exc}") from exc

    async def verify(self, params: OpenPathInput, result: OpenPathOutput) -> bool:
        # Verified if the target path exists and was dispatched
        path = Path(result.resolved_path)
        verified = path.exists() and result.opened
        result.verified = verified
        return verified


class SearchFilesInput(BaseModel):
    directory: str = Field(default=".", description="Root directory to search within")
    pattern: str = Field(default="*", description="Glob pattern (e.g. '*.py', '*report*', '*.pdf')")
    content_search: str | None = Field(default=None, description="Optional text content to grep inside files")
    max_results: int = Field(default=20, ge=1, le=100, description="Maximum number of search results to return")


class FileSearchResultItem(BaseModel):
    path: str
    size_bytes: int
    modified_time: str
    matching_line: str | None = None


class SearchFilesOutput(BaseModel):
    root_directory: str
    total_found: int
    results: list[FileSearchResultItem]


class SearchFilesTool(BaseTool[SearchFilesInput, SearchFilesOutput]):
    name = "search_files"
    description = "Search for files by name glob, extension, and optional content text."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 10.0
    input_schema = SearchFilesInput
    output_schema = SearchFilesOutput

    async def _execute(self, params: SearchFilesInput) -> SearchFilesOutput:
        root = Path(params.directory).resolve()
        validate_path_sandbox(root)

        if not root.exists() or not root.is_dir():
            raise ToolExecutionError(f"Directory not found: '{root}'")

        items: list[FileSearchResultItem] = []

        try:
            for path in root.rglob(params.pattern):
                if len(items) >= params.max_results:
                    break
                if not path.is_file():
                    continue

                # Skip version control and virtualenv dirs
                parts = set(path.parts)
                if any(p in parts for p in (".git", ".venv", "node_modules", "__pycache__")):
                    continue

                matching_snippet: str | None = None
                if params.content_search:
                    try:
                        content = path.read_text(encoding="utf-8", errors="ignore")
                        if params.content_search.lower() not in content.lower():
                            continue
                        for line in content.splitlines():
                            if params.content_search.lower() in line.lower():
                                matching_snippet = line.strip()[:120]
                                break
                    except Exception:
                        continue

                stat = path.stat()
                from datetime import datetime
                mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()

                items.append(
                    FileSearchResultItem(
                        path=str(path),
                        size_bytes=stat.st_size,
                        modified_time=mtime_iso,
                        matching_line=matching_snippet,
                    )
                )
        except Exception as exc:
            raise ToolExecutionError(f"File search failed in '{root}': {exc}") from exc

        return SearchFilesOutput(
            root_directory=str(root),
            total_found=len(items),
            results=items,
        )

    async def verify(self, params: SearchFilesInput, result: SearchFilesOutput) -> bool:
        return isinstance(result.results, list) and result.total_found >= 0


class DeleteFileInput(BaseModel):
    file_path: str = Field(description="Path to file to delete")


class DeleteFileOutput(BaseModel):
    target_path: str
    deleted: bool
    verified: bool


class DeleteFileTool(BaseTool[DeleteFileInput, DeleteFileOutput]):
    name = "delete_file"
    description = "Delete a single file. Classified as CONFIRM (requires explicit user confirmation)."
    risk_level = RiskLevel.CONFIRM  # Rule 6: Destructive action
    timeout_seconds = 5.0
    input_schema = DeleteFileInput
    output_schema = DeleteFileOutput

    async def _execute(self, params: DeleteFileInput) -> DeleteFileOutput:
        target = Path(params.file_path).resolve()
        validate_path_sandbox(target)

        if not target.exists():
            raise ToolExecutionError(f"Cannot delete non-existent file: '{target}'")
        if target.is_dir():
            raise ToolExecutionError(f"Cannot use delete_file on directory: '{target}'. Use bulk file tools.")

        try:
            os.remove(target)
            return DeleteFileOutput(
                target_path=str(target),
                deleted=True,
                verified=False,
            )
        except Exception as exc:
            raise ToolExecutionError(f"Failed to delete file '{target}': {exc}") from exc

    async def verify(self, params: DeleteFileInput, result: DeleteFileOutput) -> bool:
        """Verify that the file no longer exists on disk."""
        path = Path(result.target_path)
        verified = not path.exists()
        result.verified = verified
        return verified
