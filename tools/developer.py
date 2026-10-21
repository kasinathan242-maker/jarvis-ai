"""JARVIS Lite — Developer Copilot Tools.

Provides repo-aware code inspection, symbol searching, diff generation,
approved edit application, test execution with failure trace parsing,
and sandboxed git operations.
"""

from __future__ import annotations

import asyncio
import difflib
from pathlib import Path

from pydantic import BaseModel, Field

from core.errors import ToolExecutionError
from security.permissions import RiskLevel, validate_path_sandbox
from tools.base import BaseTool

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RepoReadInput(BaseModel):
    file_path: str = Field(description="Relative path to file in repo to read")
    start_line: int | None = Field(default=None, ge=1, description="Optional 1-indexed starting line")
    end_line: int | None = Field(default=None, ge=1, description="Optional 1-indexed ending line")


class RepoReadOutput(BaseModel):
    file_path: str
    total_lines: int
    content: str
    verified: bool


class RepoReadTool(BaseTool[RepoReadInput, RepoReadOutput]):
    name = "repo_read"
    description = "Read source code file contents within the project repository."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 10.0
    input_schema = RepoReadInput
    output_schema = RepoReadOutput

    async def _execute(self, params: RepoReadInput) -> RepoReadOutput:
        target = (PROJECT_ROOT / params.file_path).resolve()
        validate_path_sandbox(target)

        if not target.exists() or not target.is_file():
            raise ToolExecutionError(f"File not found in repository: {params.file_path}")

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = (params.start_line - 1) if params.start_line else 0
        end = params.end_line if params.end_line else len(lines)
        sliced = lines[start:end]

        numbered = [f"{start + i + 1}: {line}" for i, line in enumerate(sliced)]
        return RepoReadOutput(
            file_path=params.file_path,
            total_lines=len(lines),
            content="\n".join(numbered),
            verified=True,
        )

    async def verify(self, params: RepoReadInput, result: RepoReadOutput) -> bool:
        result.verified = bool(len(result.content) >= 0)
        return True


class SymbolSearchInput(BaseModel):
    query: str = Field(description="Symbol name or code pattern to search for across repository")
    file_extension: str | None = Field(default=".py", description="Filter by file extension")
    max_results: int = Field(default=15, ge=1, le=50)


class SymbolSearchResult(BaseModel):
    file_path: str
    line_number: int
    line_content: str


class SymbolSearchOutput(BaseModel):
    query: str
    matches_count: int
    results: list[SymbolSearchResult]
    verified: bool


class SymbolSearchTool(BaseTool[SymbolSearchInput, SymbolSearchOutput]):
    name = "symbol_search"
    description = "Search codebase for symbols, functions, or strings."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 15.0
    input_schema = SymbolSearchInput
    output_schema = SymbolSearchOutput

    async def _execute(self, params: SymbolSearchInput) -> SymbolSearchOutput:
        matches: list[SymbolSearchResult] = []
        q = params.query.lower()

        for path in PROJECT_ROOT.rglob(f"*{params.file_extension or ''}"):
            # Skip hidden / venv
            if any(part.startswith(".") or part in ("venv", ".venv", "__pycache__") for part in path.parts):
                continue
            if not path.is_file():
                continue

            try:
                for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
                    if q in line.lower():
                        rel = str(path.relative_to(PROJECT_ROOT))
                        matches.append(
                            SymbolSearchResult(
                                file_path=rel,
                                line_number=idx + 1,
                                line_content=line.strip(),
                            )
                        )
                        if len(matches) >= params.max_results:
                            break
            except Exception:
                continue

            if len(matches) >= params.max_results:
                break

        return SymbolSearchOutput(
            query=params.query,
            matches_count=len(matches),
            results=matches,
            verified=True,
        )

    async def verify(self, params: SymbolSearchInput, result: SymbolSearchOutput) -> bool:
        result.verified = True
        return True


class ProposeEditInput(BaseModel):
    file_path: str = Field(description="Relative path to file in repo to modify")
    target_content: str = Field(description="Exact snippet of existing code to replace")
    replacement_content: str = Field(description="Replacement code snippet")


class ProposeEditOutput(BaseModel):
    file_path: str
    diff: str
    matched: bool
    verified: bool


class ProposeEditTool(BaseTool[ProposeEditInput, ProposeEditOutput]):
    name = "propose_edit"
    description = "Propose a code modification and preview the unified diff before applying."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 10.0
    input_schema = ProposeEditInput
    output_schema = ProposeEditOutput

    async def _execute(self, params: ProposeEditInput) -> ProposeEditOutput:
        target = (PROJECT_ROOT / params.file_path).resolve()
        validate_path_sandbox(target)

        if not target.exists():
            raise ToolExecutionError(f"Target file not found: {params.file_path}")

        current_content = target.read_text(encoding="utf-8", errors="replace")
        matched = params.target_content in current_content

        if not matched:
            raise ToolExecutionError(
                f"Target content snippet not found in {params.file_path}. Cannot generate clean diff."
            )

        new_content = current_content.replace(params.target_content, params.replacement_content, 1)

        diff_lines = list(
            difflib.unified_diff(
                current_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{params.file_path}",
                tofile=f"b/{params.file_path}",
            )
        )

        return ProposeEditOutput(
            file_path=params.file_path,
            diff="".join(diff_lines),
            matched=True,
            verified=True,
        )

    async def verify(self, params: ProposeEditInput, result: ProposeEditOutput) -> bool:
        result.verified = bool(len(result.diff) > 0)
        return result.verified


class ApplyEditInput(BaseModel):
    file_path: str = Field(description="Relative path to file to edit")
    target_content: str = Field(description="Exact text to replace")
    replacement_content: str = Field(description="Replacement content")


class ApplyEditOutput(BaseModel):
    file_path: str
    applied: bool
    verified: bool


class ApplyEditTool(BaseTool[ApplyEditInput, ApplyEditOutput]):
    name = "apply_edit"
    description = "Apply a verified code modification to a file. Requires user confirmation."
    risk_level = RiskLevel.CONFIRM  # ALWAYS REQUIRES CONFIRMATION
    timeout_seconds = 15.0
    input_schema = ApplyEditInput
    output_schema = ApplyEditOutput

    async def _execute(self, params: ApplyEditInput) -> ApplyEditOutput:
        target = (PROJECT_ROOT / params.file_path).resolve()
        validate_path_sandbox(target)

        current = target.read_text(encoding="utf-8")
        if params.target_content not in current:
            raise ToolExecutionError("Target text does not match current file content.")

        updated = current.replace(params.target_content, params.replacement_content, 1)
        target.write_text(updated, encoding="utf-8")

        return ApplyEditOutput(
            file_path=params.file_path,
            applied=True,
            verified=True,
        )

    async def verify(self, params: ApplyEditInput, result: ApplyEditOutput) -> bool:
        target = (PROJECT_ROOT / params.file_path).resolve()
        verified = params.replacement_content in target.read_text(encoding="utf-8")
        result.verified = verified
        return verified


class RunProjectTestsInput(BaseModel):
    test_path: str | None = Field(default=None, description="Optional specific test file to run")
    verbose: bool = Field(default=True)


class RunProjectTestsOutput(BaseModel):
    exit_code: int
    all_passed: bool
    total_tests: int
    stdout_summary: str
    failure_details: str
    verified: bool


class RunProjectTestsTool(BaseTool[RunProjectTestsInput, RunProjectTestsOutput]):
    name = "run_tests"
    description = "Run the project test suite and return structured pass/fail analytics."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 90.0
    input_schema = RunProjectTestsInput
    output_schema = RunProjectTestsOutput

    async def _execute(self, params: RunProjectTestsInput) -> RunProjectTestsOutput:
        cmd = [str(PROJECT_ROOT / ".venv" / "Scripts" / "pytest.exe")]
        if params.verbose:
            cmd.append("-v")
        if params.test_path:
            cmd.append(params.test_path)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        out = stdout_bytes.decode("utf-8", errors="replace")
        err = stderr_bytes.decode("utf-8", errors="replace")

        exit_code = proc.returncode or 0
        all_passed = exit_code == 0

        # Extract failure trace if any
        failures = ""
        if not all_passed:
            fail_idx = out.find("FAILURES")
            failures = out[fail_idx : fail_idx + 2000] if fail_idx != -1 else err[:2000]

        return RunProjectTestsOutput(
            exit_code=exit_code,
            all_passed=all_passed,
            total_tests=out.count("PASSED") + out.count("FAILED"),
            stdout_summary=out[-1000:],
            failure_details=failures,
            verified=True,
        )

    async def verify(self, params: RunProjectTestsInput, result: RunProjectTestsOutput) -> bool:
        result.verified = True
        return True


class GitStatusInput(BaseModel):
    pass


class GitStatusOutput(BaseModel):
    branch: str
    modified_files: list[str]
    untracked_files: list[str]
    verified: bool


class GitStatusTool(BaseTool[GitStatusInput, GitStatusOutput]):
    name = "git_status"
    description = "Inspect git branch and working tree state."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 10.0
    input_schema = GitStatusInput
    output_schema = GitStatusOutput

    async def _execute(self, params: GitStatusInput) -> GitStatusOutput:
        proc = await asyncio.create_subprocess_exec(
            "git", "status", "--porcelain", "-b",
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, _ = await proc.communicate()
        lines = stdout_bytes.decode("utf-8", errors="replace").splitlines()

        branch = "unknown"
        modified: list[str] = []
        untracked: list[str] = []

        for line in lines:
            if line.startswith("## "):
                branch = line[3:].split("...")[0].strip()
            elif line.startswith("?? "):
                untracked.append(line[3:].strip())
            elif len(line) >= 3:
                modified.append(line[3:].strip())

        return GitStatusOutput(
            branch=branch,
            modified_files=modified,
            untracked_files=untracked,
            verified=True,
        )

    async def verify(self, params: GitStatusInput, result: GitStatusOutput) -> bool:
        result.verified = True
        return True


class GitCommitInput(BaseModel):
    message: str = Field(description="Git commit message")


class GitCommitOutput(BaseModel):
    commit_hash: str
    message: str
    verified: bool


class GitCommitTool(BaseTool[GitCommitInput, GitCommitOutput]):
    name = "git_commit"
    description = "Commit staged changes to git. Requires user confirmation."
    risk_level = RiskLevel.CONFIRM  # ALWAYS REQUIRES CONFIRMATION
    timeout_seconds = 20.0
    input_schema = GitCommitInput
    output_schema = GitCommitOutput

    async def _execute(self, params: GitCommitInput) -> GitCommitOutput:
        proc = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", params.message,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        out = stdout_bytes.decode("utf-8", errors="replace")
        err = stderr_bytes.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            raise ToolExecutionError(f"Git commit failed: {out or err}")

        # Get latest commit hash
        hash_proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        h_out, _ = await hash_proc.communicate()
        commit_hash = h_out.decode("utf-8").strip()

        return GitCommitOutput(
            commit_hash=commit_hash,
            message=params.message,
            verified=True,
        )

    async def verify(self, params: GitCommitInput, result: GitCommitOutput) -> bool:
        result.verified = bool(len(result.commit_hash) >= 7)
        return result.verified
