"""JARVIS Lite — Conversation Store (SQLite / SQLAlchemy 2.0).

Stores conversation turns, user queries, intent tags, responses, and token stats.
Persisted in data/jarvis.db with thread-safe session management.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "jarvis.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC), index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    intent = Column(String(50), nullable=True, index=True)
    tokens_used = Column(Integer, default=0)


# Initialize tables
Base.metadata.create_all(bind=engine)


def save_message(
    role: str,
    content: str,
    intent: str | None = None,
    tokens_used: int = 0,
) -> int:
    """Save a single conversation message to the database."""
    with SessionLocal() as session:
        msg = ConversationMessage(
            role=role,
            content=content,
            intent=intent,
            tokens_used=tokens_used,
        )
        session.add(msg)
        session.commit()
        session.refresh(msg)
        return int(msg.id)


def get_recent_history(limit: int = 10) -> list[dict[str, Any]]:
    """Retrieve recent conversation history formatted for model prompt context."""
    with SessionLocal() as session:
        records = (
            session.query(ConversationMessage)
            .order_by(ConversationMessage.id.desc())
            .limit(limit)
            .all()
        )
        # Return in chronological order
        ordered = list(reversed(records))
        return [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "intent": r.intent,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in ordered
        ]


def clear_history() -> int:
    """Clear conversation history (user-driven reset). Returns number of rows deleted."""
    with SessionLocal() as session:
        deleted = session.query(ConversationMessage).delete()
        session.commit()
        return deleted
