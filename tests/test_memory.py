"""Tests for SQLite Conversation Store and ChromaDB Long-Term Memory."""

import uuid

from memory.conversation_store import get_recent_history, save_message
from memory.long_term_memory import delete_memory, list_all_memories, save_long_term_memory, search_memories


def test_conversation_persistence_and_restore() -> None:
    """Verify conversation messages persist and restore across sessions."""
    unique_query = f"test_query_{uuid.uuid4().hex[:8]}"
    unique_response = f"test_response_{uuid.uuid4().hex[:8]}"

    save_message("user", unique_query, intent="test")
    save_message("assistant", unique_response, intent="test")

    # Simulate app restart by querying new session
    history = get_recent_history(limit=5)
    matching = [h for h in history if h["content"] == unique_query]
    assert len(matching) == 1
    assert matching[0]["role"] == "user"
    assert matching[0]["intent"] == "test"


def test_long_term_memory_approved_and_citations() -> None:
    """Verify long-term memory saves only with approval, searches, and cites source."""
    fact = f"User preferred programming language is Rust_{uuid.uuid4().hex[:6]}"

    # Unapproved memory must be rejected
    unapproved_id = save_long_term_memory(fact, approved=False)
    assert unapproved_id is None

    # Approved memory persists
    mem_id = save_long_term_memory(
        content=fact,
        category="preferences",
        source="unit_test_session",
        approved=True,
    )
    assert mem_id is not None

    # Search recalled memories
    results = search_memories(query="programming language", n_results=5)
    matching = [m for m in results if m["id"] == mem_id]
    assert len(matching) == 1
    assert matching[0]["source"] == "unit_test_session"
    assert matching[0]["category"] == "preferences"

    # User can list and delete
    deleted = delete_memory(mem_id)
    assert deleted is True

    # Post-delete check
    all_mems = list_all_memories()
    assert not any(m["id"] == mem_id for m in all_mems)
