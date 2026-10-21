"""JARVIS Lite — Wake Word Detection Engine (openWakeWord).

Listens continuously for the "Hey Jarvis" / "Jarvis" wake phrase using
an ONNX-quantized openWakeWord acoustic model on CPU.
Maintains extremely low idle CPU usage (< 5% on laptop) via 80ms ring-buffer streaming.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    from openwakeword.model import Model as OWWModel  # type: ignore
    HAS_OWW = True
except (ImportError, Exception):
    HAS_OWW = False

try:
    import sounddevice as sd  # type: ignore
    HAS_SD = True
except (ImportError, Exception):
    HAS_SD = False

from core.event_bus import bus
from core.logging import get_logger

logger = get_logger("voice.wake_word")

EVENT_WAKE_WORD_DETECTED = "voice.wake_word_detected"


class WakeWordDetector:
    """Continuous background wake word detector."""

    def __init__(
        self,
        model_name: str = "hey_jarvis_v0.1.onnx",
        threshold: float = 0.5,
        sample_rate: int = 16000,
        chunk_size: int = 1280,  # 80ms chunks at 16kHz
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        self._model: OWWModel | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._stream: Any | None = None
        self._callbacks: list[Callable[[float], None]] = []

    def _init_model(self) -> bool:
        if not HAS_OWW:
            logger.warn("openwakeword_not_installed")
            return False

        if self._model is not None:
            return True

        try:
            self._model = OWWModel(
                wakeword_models=[self.model_name],
                inference_framework="onnx",
            )
            logger.info("wake_word_model_loaded", model=self.model_name)
            return True
        except Exception as exc:
            logger.error("wake_word_model_load_failed", error=str(exc))
            return False

    def predict_chunk(self, audio_chunk_int16: np.ndarray) -> float:
        """Run inference on a single audio chunk. Returns confidence score [0.0, 1.0]."""
        if not self._init_model() or self._model is None:
            return 0.0

        try:
            preds: dict[str, float] = self._model.predict(audio_chunk_int16)
            # Find score for target model
            for k, score in preds.items():
                if "jarvis" in k.lower():
                    return float(score)
            return max(preds.values()) if preds else 0.0
        except Exception as exc:
            logger.debug("wake_word_predict_error", error=str(exc))
            return 0.0

    def add_callback(self, callback: Callable[[float], None]) -> None:
        """Register a callback for when wake word is detected."""
        self._callbacks.append(callback)

    def start(self, input_device: int | None = None) -> bool:
        """Start background microphone stream for wake word detection."""
        if self._running:
            return True

        if not self._init_model():
            return False

        if not HAS_SD:
            logger.warn("sounddevice_unavailable")
            return False

        self._running = True

        def _audio_loop() -> None:
            logger.info("wake_word_listener_thread_started", sample_rate=self.sample_rate)
            try:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=self.chunk_size,
                    device=input_device,
                ) as stream:
                    self._stream = stream
                    last_trigger_time = 0.0
                    cooldown_seconds = 2.0  # Prevent multiple rapid triggers

                    while self._running:
                        data, overflowed = stream.read(self.chunk_size)
                        if overflowed:
                            logger.debug("wake_word_buffer_overflow")

                        chunk = data.reshape(-1)
                        score = self.predict_chunk(chunk)

                        now = time.time()
                        if score >= self.threshold and (now - last_trigger_time) > cooldown_seconds:
                            last_trigger_time = now
                            logger.info("wake_word_triggered", score=round(score, 3))

                            # Fire registered callbacks
                            for cb in self._callbacks:
                                try:
                                    cb(score)
                                except Exception as exc:
                                    logger.warn("wake_word_callback_failed", error=str(exc))

                            # Dispatch event onto EventBus
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    loop.create_task(bus.publish(EVENT_WAKE_WORD_DETECTED, {"score": score}))
                            except Exception:
                                pass

            except Exception as exc:
                logger.warn("wake_word_stream_error", error=str(exc))
            finally:
                self._running = False
                logger.info("wake_word_listener_thread_stopped")

        self._thread = threading.Thread(target=_audio_loop, daemon=True, name="WakeWordListener")
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop background audio stream cleanly without cross-thread PortAudio collisions."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._stream = None
        logger.info("wake_word_detection_stopped")

    @property
    def is_listening(self) -> bool:
        return self._running


wake_word = WakeWordDetector()
