"""JARVIS Lite — Unified Memory Retrieval.

Combines short-term conversation context (SQLite) and long-term
semantic memory (ChromaDB) to construct prompt context with full
source citations.
"""

from __future__ import annotations

from typing import Any

from memory.conversation_store import get_recent_history
from memory.long_term_memory import search_memories


def build_augmented_context(
    query: str,
    recent_history_limit: int = 5,
    long_term_memory_limit: int = 3,
) -> dict[str, Any]:
    """Assemble a combined context bundle including recent chat history and semantic long-term memory."""
    recent_turns = get_recent_history(limit=recent_history_limit)
    long_term = search_memories(query=query, n_results=long_term_memory_limit)

    return {
        "recent_history": recent_turns,
        "recalled_memories": long_term,
    }


def format_context_for_prompt(context_bundle: dict[str, Any]) -> str:
    """Format recalled memories and conversation context into structured system prompt sections."""
    lines: list[str] = []

    long_term = context_bundle.get("recalled_memories", [])
    if long_term:
        lines.append("--- RECALLED LONG-TERM MEMORIES ---")
        for mem in long_term:
            source = mem.get("source", "conversation")
            ts = mem.get("timestamp", "")
            lines.append(f"• [Source: {source} | Time: {ts}] {mem.get('content')}")
        lines.append("")

    return "\n".join(lines)
