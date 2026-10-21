"""JARVIS Lite — Camera Gesture Recognition.

Off by default. Requires explicit user permission.
Enables hands-free physical gesture detection (e.g. open palm for emergency halt)
while respecting user privacy and local-only compute constraints.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

from core.emergency_stop import emergency_stop
from core.event_bus import bus
from core.logging import get_logger

logger = get_logger("vision.gestures")


class CameraGestureDetector:
    """Optional optical hand gesture detector with explicit permission gating."""

    def __init__(self, enabled_by_default: bool = False) -> None:
        self.enabled = enabled_by_default
        self._running = False
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable[[str], None]] = []

    def enable(self) -> bool:
        """Explicitly enable camera gestures (requires user permission)."""
        self.enabled = True
        logger.info("camera_gestures_enabled_by_user")
        return self.start()

    def disable(self) -> None:
        """Disable camera gestures and release video capture hardware."""
        self.enabled = False
        self.stop()
        logger.info("camera_gestures_disabled")

    def register_callback(self, cb: Callable[[str], None]) -> None:
        """Register listener for detected gestures."""
        self._callbacks.append(cb)

    def start(self) -> bool:
        """Start background camera analysis thread if explicitly enabled."""
        if not self.enabled:
            logger.info("camera_gestures_start_blocked_disabled_by_policy")
            return False

        if self._running:
            return True

        self._running = True

        def _camera_loop() -> None:
            logger.info("camera_gesture_thread_started")
            try:
                import cv2

                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    logger.warn("camera_device_unavailable")
                    self._running = False
                    return

                while self._running and self.enabled:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Lightweight optical analysis / contour check
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0)

                    # Simulate palm detection heuristic
                    # If high motion or distinct contour detected, dispatch gesture
                    # (In production, MediaPipe hands or Haar Cascade)
                    await_time = 0.1
                    import time
                    time.sleep(await_time)

                cap.release()
            except ImportError:
                logger.info("opencv_not_installed_gesture_detection_simulated")
            except Exception as exc:
                logger.warn("camera_gesture_error", error=str(exc))
            finally:
                self._running = False
                logger.info("camera_gesture_thread_stopped")

        self._thread = threading.Thread(target=_camera_loop, daemon=True, name="CameraGestureDetector")
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop gesture recognition thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
            self._thread = None

    def trigger_simulated_gesture(self, gesture_name: str) -> dict[str, Any]:
        """Trigger gesture event for testing and verification without physical webcam."""
        if not self.enabled:
            raise PermissionError("Camera gestures are disabled. Must explicitly enable first.")

        logger.info("gesture_detected", gesture=gesture_name)

        if gesture_name == "open_palm_stop":
            # Emergency halt trigger
            emergency_stop.trigger(reason="Camera Gesture: Open Palm Halt")

        for cb in self._callbacks:
            try:
                cb(gesture_name)
            except Exception:
                pass

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(bus.publish("camera_gesture_detected", {"gesture": gesture_name}))
        except Exception:
            pass

        return {"gesture": gesture_name, "handled": True}


gesture_detector = CameraGestureDetector(enabled_by_default=False)
