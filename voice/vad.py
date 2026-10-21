"""JARVIS Lite — Voice Activity Detector (VAD).

Detects speech presence in real-time audio streams using Silero VAD (ONNX Runtime)
with fast RMS energy fallback. Supports automatic speech endpointing and
silence detection for hands-free continuous conversations.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
    import openwakeword  # type: ignore
    HAS_ORT_VAD = True
except (ImportError, Exception):
    HAS_ORT_VAD = False

from core.logging import get_logger

logger = get_logger("voice.vad")


class VoiceActivityDetector:
    """Detects speech in audio chunks and determines utterance endpoints."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        energy_threshold: float = 0.015,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.energy_threshold = energy_threshold

        self._session: Any | None = None
        self._h: np.ndarray = np.zeros((2, 1, 64), dtype=np.float32)
        self._c: np.ndarray = np.zeros((2, 1, 64), dtype=np.float32)

        if HAS_ORT_VAD:
            self._init_silero()

    def _init_silero(self) -> None:
        try:
            vad_path = os.path.join(
                os.path.dirname(openwakeword.__file__),
                "resources",
                "models",
                "silero_vad.onnx",
            )
            if os.path.exists(vad_path):
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self._session = ort.InferenceSession(vad_path, sess_options=opts, providers=["CPUExecutionProvider"])
                logger.info("silero_vad_initialized_onnx")
        except Exception as exc:
            logger.warn("silero_vad_init_failed_falling_back_to_energy", error=str(exc))
            self._session = None

    def reset_state(self) -> None:
        """Reset recurrent state between utterances."""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def is_speech(self, audio_chunk: np.ndarray) -> tuple[bool, float]:
        """Evaluate if an audio chunk contains human speech.

        Args:
            audio_chunk: 1D numpy array of audio samples (float32, normalized [-1.0, 1.0]).

        Returns:
            (is_speech, confidence_score)
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return False, 0.0

        # Normalization check
        if audio_chunk.dtype == np.int16:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        elif audio_chunk.dtype != np.float32:
            audio_float = audio_chunk.astype(np.float32)
        else:
            audio_float = audio_chunk

        if audio_float.ndim > 1:
            audio_float = audio_float.mean(axis=1)

        # Noise floor check
        rms = float(np.sqrt(np.mean(audio_float**2)))
        if rms < 0.003:
            return False, 0.0

        # 1. Silero VAD (ONNX)
        if self._session is not None and len(audio_float) in (512, 1024, 1536):
            try:
                x = np.expand_dims(audio_float, axis=0)  # Shape [1, N]
                sr = np.array(self.sample_rate, dtype=np.int64)
                inputs = {
                    "input": x,
                    "sr": sr,
                    "h": self._h,
                    "c": self._c,
                }
                out, self._h, self._c = self._session.run(None, inputs)
                prob = float(out[0][0])
                if prob >= self.threshold:
                    return True, prob
            except Exception:
                pass

        # 2. RMS Energy Fallback
        confidence = min(1.0, rms / (self.energy_threshold * 2))
        return rms >= self.energy_threshold, confidence


# Global VAD singleton
vad = VoiceActivityDetector()
