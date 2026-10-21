"""JARVIS Lite — System Lifecycle Coordinator.

Manages application startup sequence, component initialization,
health status probes, and clean shutdown handling.
"""

from __future__ import annotations

from bootstrap import run_bootstrap
from config.settings import settings
from core.event_bus import bus
from core.logging import get_logger, setup_logging
from core.task_manager import task_manager
from memory.conversation_store import engine as sql_engine
from security.secrets import get_or_create_api_token

logger = get_logger("core.lifecycle")


class SystemLifecycle:
    def __init__(self) -> None:
        self.is_running = False
        self.api_token: str = ""

    async def startup(self) -> bool:
        """Execute full startup sequence."""
        # 1. Setup structured logging
        setup_logging(log_level=settings.system.log_level)
        logger.info("jarvis_lite_starting", version=settings.system.version)

        # 2. Run bootstrap diagnostics
        bootstrap_passed = run_bootstrap()
        if not bootstrap_passed:
            logger.error("bootstrap_preflight_failed")

        # 3. Secure API Bearer token in Windows Credential Manager
        self.api_token = get_or_create_api_token()
        logger.info("api_bearer_token_ready")

        self.is_running = True
        await bus.publish("system_started", {"version": settings.system.version})
        return True

    async def shutdown(self) -> None:
        """Execute graceful shutdown sequence."""
        if not self.is_running:
            return

        logger.info("jarvis_lite_shutting_down")
        self.is_running = False

        # 1. Abort any running tasks
        await task_manager.emergency_stop()

        # 2. Notify subscribers
        await bus.publish("system_shutting_down", {})

        # 3. Dispose SQL engine connection pools
        try:
            sql_engine.dispose()
        except Exception:
            pass

        logger.info("jarvis_lite_shutdown_complete")


lifecycle = SystemLifecycle()
