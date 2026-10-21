"""JARVIS Lite — Proactive Assistance Subsystem.

Evaluates system health, schedules, battery, and context in the background.
Adheres strictly to safety rules:
- Respects user-defined quiet hours.
- Per-feature granular toggles.
- Never executes destructive or mutation actions without explicit user approval.
"""

from __future__ import annotations

from proactive.engine import proactive_engine

__all__ = ["proactive_engine"]
