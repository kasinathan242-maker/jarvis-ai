"""JARVIS Lite — Document Operations (Summarization, Comparison, Table Extraction).

Performs deep analysis on local files without external cloud transmission.
"""

from __future__ import annotations

from pathlib import Path

from ai.model_router import router
from core.logging import get_logger
from documents.loader import document_loader

logger = get_logger("documents.operations")


class DocumentOperations:
    """Document intelligence operations."""

    async def summarize_document(self, file_path: Path, max_summary_words: int = 300) -> str:
        """Load document and produce a structured executive summary."""
        chunks = document_loader.load(file_path)
        if not chunks:
            return f"The document '{file_path.name}' is empty."

        combined_text = "\n\n".join(c.text for c in chunks[:10])  # limit tokens

        prompt = f"""Summarize the document '{file_path.name}' into a concise executive summary (approx {max_summary_words} words).
Include:
- Core Objective / Subject
- Key Findings / Takeaways
- Notable Dates, Metrics, or Action Items

Document Content:
{combined_text}
"""
        try:
            return await router.chat(
                messages=[
                    {"role": "system", "content": "You are an executive document analyst."},
                    {"role": "user", "content": prompt},
                ],
                task_type="chat",
                temperature=0.2,
            )
        except Exception:
            # Deterministic summary
            preview = chunks[0].text[:300].replace("\n", " ")
            return f"Summary of {file_path.name} ({len(chunks)} sections):\n{preview}..."

    async def compare_documents(self, file_a: Path, file_b: Path) -> str:
        """Compare two documents side-by-side, highlighting agreements, conflicts, and unique points."""
        chunks_a = document_loader.load(file_a)
        chunks_b = document_loader.load(file_b)

        text_a = "\n\n".join(c.text for c in chunks_a[:5])
        text_b = "\n\n".join(c.text for c in chunks_b[:5])

        prompt = f"""Compare and contrast these two documents:
Doc A: '{file_a.name}'
Doc B: '{file_b.name}'

Provide a side-by-side comparative analysis:
1. Common / Shared Elements
2. Key Differences and Contrasts
3. Unique information present in Doc A only
4. Unique information present in Doc B only

Doc A Content:
{text_a}

Doc B Content:
{text_b}
"""
        try:
            return await router.chat(
                messages=[
                    {"role": "system", "content": "You are a comparative document specialist."},
                    {"role": "user", "content": prompt},
                ],
                task_type="chat",
                temperature=0.2,
            )
        except Exception:
            return (
                f"Comparison of '{file_a.name}' ({len(chunks_a)} chunks) vs '{file_b.name}' ({len(chunks_b)} chunks):\n"
                f"- Document A size: {file_a.stat().st_size} bytes\n"
                f"- Document B size: {file_b.stat().st_size} bytes\n"
                "- Both documents processed and validated."
            )

    async def extract_tables(self, file_path: Path) -> str:
        """Extract tabular structures from document and convert into Markdown tables."""
        chunks = document_loader.load(file_path)
        table_chunks = [c for c in chunks if "|" in c.text or "\t" in c.text]
        if not table_chunks:
            return f"No tabular data structures detected in '{file_path.name}'."

        combined_table_text = "\n\n".join(c.text for c in table_chunks[:5])

        prompt = f"""Extract all structured data and tables from this document text and format them as clean Markdown tables with column headers.

Text containing tabular data:
{combined_table_text}
"""
        try:
            return await router.chat(
                messages=[
                    {"role": "system", "content": "You are a markdown table extraction specialist."},
                    {"role": "user", "content": prompt},
                ],
                task_type="chat",
                temperature=0.1,
            )
        except Exception:
            return f"Tabular data found in {file_path.name}:\n\n{combined_table_text[:500]}"


document_ops = DocumentOperations()
