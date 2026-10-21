"""JARVIS Lite — Asynchronous In-Process Event Bus.

Decouples voice input, AI reasoning, tool execution, telemetry monitors,
and WebSocket streams via type-safe pub/sub messaging.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

from core.logging import get_logger

logger = get_logger("core.event_bus")

EventHandler = Callable[[str, Any], Coroutine[Any, Any, None]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[EventHandler]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register an async callback for a specific event type."""
        async with self._lock:
            self._subscribers[event_type].add(handler)

    async def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a previously registered handler."""
        async with self._lock:
            if event_type in self._subscribers and handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)

    async def publish(self, event_type: str, data: Any = None) -> None:
        """Dispatch event concurrently to all active subscribers."""
        handlers: list[EventHandler] = []
        async with self._lock:
            # Copy handlers to avoid mutation during dispatch
            handlers = list(self._subscribers.get(event_type, set()))
            # Also dispatch to wildcard listeners
            handlers.extend(list(self._subscribers.get("*", set())))

        if not handlers:
            return

        tasks = [asyncio.create_task(self._safe_invoke(h, event_type, data)) for h in handlers]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_invoke(self, handler: EventHandler, event_type: str, data: Any) -> None:
        try:
            await handler(event_type, data)
        except Exception as exc:
            logger.error("event_handler_failed", event_name=event_type, error=str(exc))


bus = EventBus()
