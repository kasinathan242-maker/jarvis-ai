"""JARVIS Lite — Memory Extractor.

Extracts candidate user preferences, facts, or instructions from conversation
and proposes them for explicit user approval before storing in long-term memory.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class MemoryCandidate(BaseModel):
    category: str = Field(description="Category (preference, project, user_fact, rule)")
    content: str = Field(description="Extracted statement to remember")
    confidence: float = Field(default=0.8, description="Extraction confidence")


# Heuristic patterns for common memory extraction
MEMORY_PATTERNS = [
    (re.compile(r"(?:remember that|note that|keep in mind that)\s+(.+)", re.IGNORECASE), "user_instruction"),
    (re.compile(r"(?:my name is|i am)\s+([A-Z][a-z]+)", re.IGNORECASE), "user_fact"),
    (re.compile(r"i prefer\s+(.+)", re.IGNORECASE), "preference"),
    (re.compile(r"always\s+(?:use|format|reply with)\s+(.+)", re.IGNORECASE), "rule"),
]


def extract_memory_candidates(text: str) -> list[MemoryCandidate]:
    """Scan user input text for statements that should be proposed for memory storage."""
    candidates: list[MemoryCandidate] = []
    clean = text.strip()

    for pattern, category in MEMORY_PATTERNS:
        match = pattern.search(clean)
        if match:
            captured = match.group(1).strip().rstrip(".?!")
            candidates.append(
                MemoryCandidate(
                    category=category,
                    content=captured,
                    confidence=0.9,
                )
            )

    return candidates
