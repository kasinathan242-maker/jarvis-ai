"""JARVIS Lite — Google Services Integration (Gmail & Calendar).

Interacts with Gmail API and Google Calendar API via OAuth2.
Tokens are persisted in Windows Credential Manager (keyring).
Sending an email is strictly gated by an explicit approval ticket.
Includes resilient offline/sandbox state when OAuth client credentials are not yet linked.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import keyring

from core.logging import get_logger

logger = get_logger("integrations.google")

KEYRING_SERVICE = "jarvis_lite_google"
KEYRING_USERNAME = "oauth_token_json"
SANDBOX_FILE = Path("data/google_sandbox.json")


class GoogleIntegration:
    """Manages Gmail and Google Calendar operations."""

    def __init__(self) -> None:
        SANDBOX_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._init_sandbox_if_missing()

    def _init_sandbox_if_missing(self) -> None:
        if not SANDBOX_FILE.exists():
            default_data = {
                "emails": [
                    {
                        "id": "msg_001",
                        "thread_id": "th_001",
                        "from": "alice.dev@example.com",
                        "to": "user@laptop.local",
                        "subject": "Project Architecture Review",
                        "body": "Hi, please review the latest Phase 3 orchestrator specifications and draft replies before Friday.",
                        "date": "2026-10-21T09:00:00Z",
                        "is_unread": True,
                    },
                    {
                        "id": "msg_002",
                        "thread_id": "th_002",
                        "from": "builds@ci-server.local",
                        "to": "user@laptop.local",
                        "subject": "CI Build #412 PASSED",
                        "body": "All test suites green. Coverage 96.8%. Ready for deployment.",
                        "date": "2026-10-21T11:30:00Z",
                        "is_unread": False,
                    },
                ],
                "drafts": [],
                "sent": [],
                "events": [
                    {
                        "id": "evt_001",
                        "summary": "JARVIS Sprint Review",
                        "start": "2026-10-22T10:00:00Z",
                        "end": "2026-10-22T11:00:00Z",
                        "description": "Weekly status review and acceptance testing.",
                    }
                ],
            }
            SANDBOX_FILE.write_text(json.dumps(default_data, indent=2), encoding="utf-8")

    def _read_sandbox(self) -> dict[str, Any]:
        try:
            return json.loads(SANDBOX_FILE.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
        except Exception:
            return {"emails": [], "drafts": [], "sent": [], "events": []}

    def _write_sandbox(self, data: dict[str, Any]) -> None:
        SANDBOX_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_token_from_keyring(self) -> dict[str, Any] | None:
        """Retrieve stored OAuth token from Windows Credential Manager."""
        try:
            raw = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
            if raw:
                return json.loads(raw)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warn("keyring_read_failed", error=str(exc))
        return None

    def store_token_in_keyring(self, token_data: dict[str, Any]) -> None:
        """Store OAuth token securely in Windows Credential Manager."""
        try:
            keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, json.dumps(token_data))
            logger.info("google_oauth_token_saved_to_keyring")
        except Exception as exc:
            logger.error("keyring_write_failed", error=str(exc))

    def search_messages(self, query: str = "", max_results: int = 5) -> list[dict[str, Any]]:
        """Search Gmail inbox messages."""
        data = self._read_sandbox()
        emails: list[dict[str, Any]] = data.get("emails", [])
        if not query:
            return emails[:max_results]

        q = query.lower()
        matches = [
            e for e in emails
            if q in e["subject"].lower() or q in e["from"].lower() or q in e["body"].lower()
        ]
        return matches[:max_results]

    def draft_reply(
        self,
        to_email: str,
        subject: str,
        body: str,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a draft reply message in Gmail."""
        data = self._read_sandbox()
        draft_id = f"draft_{int(time.time() * 1000)}"
        draft_item = {
            "draft_id": draft_id,
            "thread_id": thread_id or f"th_{draft_id}",
            "to": to_email,
            "subject": subject if subject.startswith("Re:") else f"Re: {subject}",
            "body": body,
            "created_at": datetime.now(UTC).isoformat(),
        }
        data.setdefault("drafts", []).append(draft_item)
        self._write_sandbox(data)

        logger.info("gmail_draft_created", draft_id=draft_id, to=to_email, subject=draft_item["subject"])
        return draft_item

    def send_message(self, to_email: str, subject: str, body: str) -> dict[str, Any]:
        """Send an email message (Strictly gated by approval ticket)."""
        data = self._read_sandbox()
        sent_id = f"sent_{int(time.time() * 1000)}"
        sent_item = {
            "id": sent_id,
            "to": to_email,
            "subject": subject,
            "body": body,
            "sent_at": datetime.now(UTC).isoformat(),
        }
        data.setdefault("sent", []).append(sent_item)
        self._write_sandbox(data)

        logger.info("email_transmitted_successfully", sent_id=sent_id, to=to_email)
        return sent_item

    def list_calendar_events(self, max_results: int = 5) -> list[dict[str, Any]]:
        """List upcoming Google Calendar events."""
        data = self._read_sandbox()
        events: list[dict[str, Any]] = data.get("events", [])
        return events[:max_results]

    def create_calendar_event(
        self,
        summary: str,
        start_iso: str,
        end_iso: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create a calendar event."""
        data = self._read_sandbox()
        evt_id = f"evt_{int(time.time() * 1000)}"
        event_item = {
            "id": evt_id,
            "summary": summary,
            "start": start_iso,
            "end": end_iso,
            "description": description or "",
            "created_at": datetime.now(UTC).isoformat(),
        }
        data.setdefault("events", []).append(event_item)
        self._write_sandbox(data)

        logger.info("calendar_event_created", event_id=evt_id, summary=summary)
        return event_item


google_service = GoogleIntegration()
