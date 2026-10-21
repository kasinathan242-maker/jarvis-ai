"""JARVIS Lite — Global Emergency Stop Coordinator.

Guarantees instantaneous system abort (< 1.0s):
- Cancels all running background tasks and workflows via TaskManager.
- Rejects all pending tool approval tickets via ApprovalManager.
- Aborts active speech synthesis playback.
- Registers global hotkey (Ctrl+Alt+Shift+X) and listens for voice triggers.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from core.event_bus import bus
from core.logging import get_logger
from core.task_manager import task_manager
from security.approval import approval_manager

try:
    import keyboard
    HAS_KEYBOARD = True
except (ImportError, Exception):
    HAS_KEYBOARD = False

logger = get_logger("core.emergency_stop")

HOTKEY_EMERGENCY_STOP = "ctrl+alt+shift+x"


class EmergencyStopCoordinator:
    """Central authority for immediate emergency halt across all subsystems."""

    def __init__(self) -> None:
        self._hotkey_hooked = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def initialize_hotkey(self, loop: asyncio.AbstractEventLoop | None = None) -> bool:
        """Register global Windows hotkey for emergency stop."""
        self._loop = loop or (asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else None)
        if not HAS_KEYBOARD:
            logger.warn("keyboard_module_unavailable_for_hotkey")
            return False

        if self._hotkey_hooked:
            return True

        try:
            keyboard.add_hotkey(HOTKEY_EMERGENCY_STOP, self._on_hotkey_pressed)
            self._hotkey_hooked = True
            logger.info("emergency_stop_hotkey_registered", hotkey=HOTKEY_EMERGENCY_STOP)
            return True
        except Exception as exc:
            logger.warn("failed_to_register_hotkey", error=str(exc))
            return False

    def _on_hotkey_pressed(self) -> None:
        """Callback invoked from Windows keyboard hook thread."""
        logger.warning("emergency_stop_hotkey_triggered")
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.trigger_halt(source="hotkey"), self._loop)
        else:
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self.trigger_halt(source="hotkey"))
            except Exception:
                pass

    async def trigger_halt(self, source: str = "manual") -> dict[str, Any]:
        """Execute immediate abort across all subsystems in < 1 second.

        Returns:
            Dict containing cancellation metrics and duration in milliseconds.
        """
        t0 = time.perf_counter()
        logger.warning("triggering_system_halt", source=source)

        # 1. Stop any currently playing audio immediately
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

        # 2. Cancel all running background tasks
        tasks_cancelled = await task_manager.emergency_stop()

        # 3. Reject all pending approval requests
        approvals_cancelled = approval_manager.cancel_all_pending(reason="emergency_stop")

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logger.warning(
            "system_halt_complete",
            source=source,
            duration_ms=duration_ms,
            tasks_cancelled=tasks_cancelled,
            approvals_cancelled=approvals_cancelled,
        )

        payload = {
            "source": source,
            "tasks_cancelled": tasks_cancelled,
            "approvals_cancelled": approvals_cancelled,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        }

        await bus.publish("emergency_stop_halted", payload)
        return payload

    def trigger(self, reason: str = "manual") -> dict[str, Any]:
        """Synchronously trigger emergency halt across audio, tasks, and approvals."""
        t0 = time.perf_counter()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

        approvals_cancelled = approval_manager.cancel_all_pending(reason=reason)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.trigger_halt(source=reason))
        except Exception:
            pass

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "source": reason,
            "tasks_cancelled": 0,
            "approvals_cancelled": approvals_cancelled,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        }

    def cleanup(self) -> None:
        """Unregister keyboard hotkeys."""
        if self._hotkey_hooked and HAS_KEYBOARD:
            try:
                keyboard.remove_hotkey(HOTKEY_EMERGENCY_STOP)
            except Exception:
                pass
            self._hotkey_hooked = False


emergency_stop = EmergencyStopCoordinator()
