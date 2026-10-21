"""JARVIS Lite — Adaptive Personalization Manager.

Maintains user-configured behavioral and style preferences learned strictly from
explicit user feedback. Fully reviewable, auditable, and instantly resettable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.logging import get_logger

logger = get_logger("personalization.manager")

PERSONALIZATION_FILE = Path("data/personalization.json")


@dataclass
class UserPreference:
    key: str
    value: Any
    category: str
    source: str
    created_at: str
    notes: str = ""


class PersonalizationManager:
    """Manages explicit preferences with auditability, inspection, and reset guarantees."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path or PERSONALIZATION_FILE
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._preferences: dict[str, UserPreference] = {}
        self._load()

    def _load(self) -> None:
        """Load stored preferences from disk."""
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("preferences", []):
                    pref = UserPreference(
                        key=item["key"],
                        value=item["value"],
                        category=item.get("category", "general"),
                        source=item.get("source", "explicit_feedback"),
                        created_at=item.get("created_at", datetime.now(UTC).isoformat()),
                        notes=item.get("notes", ""),
                    )
                    self._preferences[pref.key] = pref
            logger.info("personalization_loaded", count=len(self._preferences))
        except Exception as exc:
            logger.warn("personalization_load_error", error=str(exc))

    def _save(self) -> None:
        """Persist preferences atomically to disk."""
        try:
            data = {
                "updated_at": datetime.now(UTC).isoformat(),
                "preferences": [asdict(p) for p in self._preferences.values()],
            }
            tmp_path = self.storage_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(self.storage_path)
        except Exception as exc:
            logger.error("personalization_save_error", error=str(exc))

    def record_explicit_feedback(
        self,
        key: str,
        value: Any,
        category: str = "general",
        notes: str = "",
        source: str = "explicit_feedback",
    ) -> UserPreference:
        """Record a new preference derived strictly from explicit user input."""
        pref = UserPreference(
            key=key.strip().lower(),
            value=value,
            category=category.strip().lower(),
            source=source,
            created_at=datetime.now(UTC).isoformat(),
            notes=notes,
        )
        self._preferences[pref.key] = pref
        self._save()
        logger.info("explicit_preference_recorded", key=pref.key, value=pref.value)
        return pref

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Retrieve a specific user preference value."""
        pref = self._preferences.get(key.strip().lower())
        return pref.value if pref is not None else default

    def list_preferences(self) -> list[dict[str, Any]]:
        """List all current preferences for user review."""
        return [asdict(p) for p in self._preferences.values()]

    def delete_preference(self, key: str) -> bool:
        """Remove a single preference by key."""
        k = key.strip().lower()
        if k in self._preferences:
            del self._preferences[k]
            self._save()
            logger.info("preference_deleted", key=k)
            return True
        return False

    def reset_personalization(self) -> int:
        """Wipe all learned preferences and return count of deleted items."""
        count = len(self._preferences)
        self._preferences.clear()
        if self.storage_path.exists():
            try:
                self.storage_path.unlink()
            except Exception:
                pass
        self._save()
        logger.info("personalization_reset_complete", cleared_count=count)
        return count

    def get_system_prompt_personalization(self) -> str:
        """Format active preferences into system prompt guidelines."""
        if not self._preferences:
            return ""
        lines = ["User Preferences (Explicitly Learned):"]
        for p in self._preferences.values():
            lines.append(f"- {p.key}: {p.value}")
        return "\n".join(lines)


personalization_manager = PersonalizationManager()
