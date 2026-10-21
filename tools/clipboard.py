"""JARVIS Lite — Clipboard Tool.

Read and write text to the Windows system clipboard.
Write operations verify reality by reading back the clipboard buffer.
"""

from __future__ import annotations

import pyperclip
from pydantic import BaseModel, Field

from core.errors import ToolExecutionError
from security.permissions import RiskLevel
from tools.base import BaseTool


class ClipboardReadInput(BaseModel):
    pass


class ClipboardReadOutput(BaseModel):
    content: str
    length: int


class ClipboardReadTool(BaseTool[ClipboardReadInput, ClipboardReadOutput]):
    name = "clipboard_read"
    description = "Read the current text content from the Windows clipboard."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 2.0
    input_schema = ClipboardReadInput
    output_schema = ClipboardReadOutput

    async def _execute(self, params: ClipboardReadInput) -> ClipboardReadOutput:
        try:
            content = pyperclip.paste()
            return ClipboardReadOutput(content=content, length=len(content))
        except Exception as exc:
            raise ToolExecutionError(f"Failed to read from clipboard: {exc}") from exc

    async def verify(self, params: ClipboardReadInput, result: ClipboardReadOutput) -> bool:
        return isinstance(result.content, str) and result.length >= 0


class ClipboardWriteInput(BaseModel):
    text: str = Field(description="Text to copy to the clipboard")


class ClipboardWriteOutput(BaseModel):
    characters_written: int
    verified: bool


class ClipboardWriteTool(BaseTool[ClipboardWriteInput, ClipboardWriteOutput]):
    name = "clipboard_write"
    description = "Copy specified text into the Windows system clipboard."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 2.0
    input_schema = ClipboardWriteInput
    output_schema = ClipboardWriteOutput

    async def _execute(self, params: ClipboardWriteInput) -> ClipboardWriteOutput:
        try:
            pyperclip.copy(params.text)
            return ClipboardWriteOutput(
                characters_written=len(params.text),
                verified=False,
            )
        except Exception as exc:
            raise ToolExecutionError(f"Failed to write to clipboard: {exc}") from exc

    async def verify(self, params: ClipboardWriteInput, result: ClipboardWriteOutput) -> bool:
        """Verify that the clipboard actually contains the exact text that was written."""
        current = str(pyperclip.paste())
        verified = bool(current == params.text)
        result.verified = verified
        return verified
