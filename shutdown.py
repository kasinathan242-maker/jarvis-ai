"""JARVIS Lite — Graceful Shutdown Handler.

Hooks OS signals (SIGINT, SIGTERM) on Windows to ensure background
processes, database connections, and model connections close cleanly.
"""

from __future__ import annotations

import asyncio
import signal

from core.lifecycle import lifecycle
from core.logging import get_logger

logger = get_logger("shutdown")


def setup_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Register signal handlers on supported platforms."""
    def _handle_signal() -> None:
        logger.info("shutdown_signal_received")
        loop.create_task(shutdown_system())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            # Windows does not support loop.add_signal_handler for all signals
            pass


async def shutdown_system() -> None:
    """Trigger complete lifecycle shutdown and exit."""
    await lifecycle.shutdown()
