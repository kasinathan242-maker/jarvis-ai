"""JARVIS Lite — Long-Term Vector Memory (ChromaDB).

Implements persistent semantic memory storage.
Rule 11 Compliance:
- Memory writes are visible and deletable by the user.
- Every recalled memory carries its original source and timestamp.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.config import Settings

from core.logging import get_logger

logger = get_logger("memory.long_term")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize persistent ChromaDB client
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False, allow_reset=True),
)

MEMORY_COLLECTION = "jarvis_long_term_memories"


def get_memory_collection() -> chromadb.Collection:
    """Get or create the main long-term memory collection using custom embeddings."""
    try:
        from ai.custom_model_engine import custom_chroma_embed_fn

        return chroma_client.get_or_create_collection(
            name=MEMORY_COLLECTION,
            embedding_function=cast(Any, custom_chroma_embed_fn),
            metadata={"description": "User preferences, learned facts, and approved memories"},
        )
    except Exception:
        return chroma_client.get_or_create_collection(
            name=MEMORY_COLLECTION,
            metadata={"description": "User preferences, learned facts, and approved memories"},
        )


def save_long_term_memory(
    content: str,
    category: str = "general",
    source: str = "user_conversation",
    approved: bool = True,
    memory_id: str | None = None,
) -> str | None:
    """Store an approved memory item in ChromaDB.

    If approved is False, the write is rejected to enforce Rule 11.
    """
    if not approved:
        logger.info("memory_write_rejected_unapproved", content=content)
        return None

    coll = get_memory_collection()
    mid = memory_id or f"mem_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(UTC).isoformat()

    metadata: dict[str, str | bool] = {
        "category": category,
        "source": source,
        "timestamp": now_iso,
        "approved": approved,
    }

    coll.upsert(
        ids=[mid],
        documents=[content],
        metadatas=[metadata],
    )
    logger.info("long_term_memory_saved", id=mid, category=category, source=source)
    return mid


def search_memories(query: str, n_results: int = 5) -> list[dict[str, Any]]:
    """Retrieve top-N semantically relevant memories with their source citations."""
    coll = get_memory_collection()
    count = coll.count()
    if count == 0:
        return []

    limit = min(n_results, count)
    results = coll.query(
        query_texts=[query],
        n_results=limit,
    )

    memories: list[dict[str, Any]] = []
    docs_outer = results.get("documents")
    docs = docs_outer[0] if docs_outer else []

    metas_outer = results.get("metadatas")
    metas = metas_outer[0] if metas_outer else []

    ids_outer = results.get("ids")
    ids = ids_outer[0] if ids_outer else []

    dists_outer = results.get("distances")
    distances = dists_outer[0] if dists_outer else [0.0] * len(docs)

    for i in range(len(docs)):
        memories.append({
            "id": ids[i],
            "content": docs[i],
            "category": metas[i].get("category", "general"),
            "source": metas[i].get("source", "unknown"),
            "timestamp": metas[i].get("timestamp", ""),
            "relevance_distance": distances[i] if i < len(distances) else None,
        })
    return memories


def list_all_memories() -> list[dict[str, Any]]:
    """List all long-term memories for user inspection and auditing."""
    coll = get_memory_collection()
    items = coll.get()
    memories: list[dict[str, Any]] = []
    docs = items.get("documents") or []
    metas = items.get("metadatas") or []
    ids = items.get("ids") or []

    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        memories.append({
            "id": ids[i],
            "content": docs[i],
            "category": meta.get("category", "general"),
            "source": meta.get("source", "unknown"),
            "timestamp": meta.get("timestamp", ""),
        })
    return memories


def delete_memory(memory_id: str) -> bool:
    """Delete a specific memory by ID."""
    coll = get_memory_collection()
    try:
        coll.delete(ids=[memory_id])
        logger.info("memory_deleted", id=memory_id)
        return True
    except Exception as exc:
        logger.warn("memory_delete_failed", id=memory_id, error=str(exc))
        return False
