"""JARVIS Lite — Security Audit Log.

Provides an immutable, persistent audit log table for every proposed and
executed tool action, recording parameters, risk levels, approvals,
verification results, and execution metrics.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
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


class ToolAuditRecord(Base):
    __tablename__ = "tool_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC), index=True)
    tool_name = Column(String(100), nullable=False, index=True)
    risk_level = Column(String(20), nullable=False)
    parameters_json = Column(Text, nullable=False)
    user_approved = Column(Boolean, default=True)
    execution_status = Column(String(20), nullable=False)  # SUCCESS, FAILED, BLOCKED, REJECTED
    verification_passed = Column(Boolean, default=False)
    output_preview = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    duration_ms = Column(Float, nullable=False, default=0.0)


# Initialize tables
Base.metadata.create_all(bind=engine)


def log_tool_invocation(
    tool_name: str,
    risk_level: str,
    parameters: dict[str, Any],
    user_approved: bool,
    execution_status: str,
    verification_passed: bool,
    output_preview: str | None = None,
    error_message: str | None = None,
    duration_ms: float = 0.0,
) -> int:
    """Record an audit log entry for a tool execution."""
    # Redact potential secrets before serializing to audit table
    clean_params = {}
    for k, v in parameters.items():
        if any(secret_term in k.lower() for secret_term in ("key", "token", "password", "secret")):
            clean_params[k] = "[REDACTED]"
        else:
            clean_params[k] = v

    record = ToolAuditRecord(
        tool_name=tool_name,
        risk_level=risk_level,
        parameters_json=json.dumps(clean_params, default=str),
        user_approved=user_approved,
        execution_status=execution_status,
        verification_passed=verification_passed,
        output_preview=str(output_preview)[:1000] if output_preview else None,
        error_message=str(error_message)[:1000] if error_message else None,
        duration_ms=duration_ms,
    )

    with SessionLocal() as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        return int(record.id)


def get_recent_audit_records(limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve recent audit logs for UI dashboard or inspection."""
    with SessionLocal() as session:
        records = (
            session.query(ToolAuditRecord)
            .order_by(ToolAuditRecord.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "tool_name": r.tool_name,
                "risk_level": r.risk_level,
                "parameters": json.loads(r.parameters_json),
                "user_approved": r.user_approved,
                "execution_status": r.execution_status,
                "verification_passed": r.verification_passed,
                "output_preview": r.output_preview,
                "error_message": r.error_message,
                "duration_ms": r.duration_ms,
            }
            for r in records
        ]


def record_audit_event(
    tool_name: str,
    parameters: dict[str, Any],
    execution_status: str,
    risk_level: str = "HIGH",
    user_approved: bool = False,
    verification_passed: bool = False,
    output_preview: str | None = None,
    error_message: str | None = None,
    duration_ms: float = 0.0,
) -> int:
    """Convenience helper to record security events into immutable audit log."""
    return log_tool_invocation(
        tool_name=tool_name,
        risk_level=risk_level,
        parameters=parameters,
        user_approved=user_approved,
        execution_status=execution_status,
        verification_passed=verification_passed,
        output_preview=output_preview,
        error_message=error_message,
        duration_ms=duration_ms,
    )

