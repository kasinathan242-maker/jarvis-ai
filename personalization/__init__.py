"""JARVIS Lite — Personalization Subsystem.

Adapts to user working preferences strictly based on explicit user feedback.
Fully transparent, reviewable via API/UI, and instantly resettable.
"""

from __future__ import annotations

from personalization.manager import personalization_manager

__all__ = ["personalization_manager"]
