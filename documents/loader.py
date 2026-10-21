"""JARVIS Lite — Universal Document Loader.

Extracts text, structural metadata, and exact page numbers from PDF, DOCX, TXT, and Markdown.
Applies security sanitization to guard against prompt injection in documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pypdf
from docx import Document as DocxDocument

from core.errors import ToolExecutionError
from core.logging import get_logger
from security.injection_guard import is_adversarial, sanitize_for_prompt

logger = get_logger("documents.loader")


@dataclass
class DocumentChunk:
    text: str
    file_path: str
    file_name: str
    page_number: int | None
    chunk_index: int
    has_injection_risk: bool = False


class DocumentLoader:
    """Universal parser for PDF, DOCX, TXT, and Markdown files."""

    def load(self, file_path: Path, chunk_size: int = 1000) -> list[DocumentChunk]:
        """Parse file into structured chunks with exact page numbering."""
        path = file_path.resolve()
        if not path.exists():
            raise ToolExecutionError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(path, chunk_size)
        elif suffix == ".docx":
            return self._load_docx(path, chunk_size)
        elif suffix in (".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml"):
            return self._load_text(path, chunk_size)
        else:
            raise ToolExecutionError(f"Unsupported document format '{suffix}'. Supported: PDF, DOCX, TXT, MD.")

    def _load_pdf(self, path: Path, chunk_size: int) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        try:
            reader = pypdf.PdfReader(str(path))
            chunk_idx = 0
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                page_text = page.extract_text() or ""
                clean_text = page_text.strip()
                if not clean_text:
                    continue

                # Split long pages into chunk_size windows
                for i in range(0, len(clean_text), chunk_size):
                    window = clean_text[i : i + chunk_size]
                    injection = is_adversarial(window)
                    sanitized = sanitize_for_prompt(window)
                    chunks.append(
                        DocumentChunk(
                            text=sanitized,
                            file_path=str(path),
                            file_name=path.name,
                            page_number=page_num,
                            chunk_index=chunk_idx,
                            has_injection_risk=injection,
                        )
                    )
                    chunk_idx += 1
            return chunks
        except Exception as exc:
            raise ToolExecutionError(f"Failed to parse PDF document {path.name}: {exc}") from exc

    def _load_docx(self, path: Path, chunk_size: int) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        try:
            doc = DocxDocument(str(path))
            full_parts: list[str] = []

            # Paragraphs
            for p in doc.paragraphs:
                if p.text.strip():
                    full_parts.append(p.text.strip())

            # Tables
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        full_parts.append(" | ".join(row_cells))

            all_text = "\n\n".join(full_parts)
            chunk_idx = 0
            for i in range(0, max(len(all_text), 1), chunk_size):
                window = all_text[i : i + chunk_size]
                if not window.strip():
                    continue
                injection = is_adversarial(window)
                chunks.append(
                    DocumentChunk(
                        text=sanitize_for_prompt(window),
                        file_path=str(path),
                        file_name=path.name,
                        page_number=1,  # DOCX has flow pagination
                        chunk_index=chunk_idx,
                        has_injection_risk=injection,
                    )
                )
                chunk_idx += 1
            return chunks
        except Exception as exc:
            raise ToolExecutionError(f"Failed to parse DOCX document {path.name}: {exc}") from exc

    def _load_text(self, path: Path, chunk_size: int) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            chunk_idx = 0
            for i in range(0, max(len(raw), 1), chunk_size):
                window = raw[i : i + chunk_size]
                if not window.strip():
                    continue
                injection = is_adversarial(window)
                chunks.append(
                    DocumentChunk(
                        text=sanitize_for_prompt(window),
                        file_path=str(path),
                        file_name=path.name,
                        page_number=1,
                        chunk_index=chunk_idx,
                        has_injection_risk=injection,
                    )
                )
                chunk_idx += 1
            return chunks
        except Exception as exc:
            raise ToolExecutionError(f"Failed to read text file {path.name}: {exc}") from exc


document_loader = DocumentLoader()
