"""JARVIS Lite — Speech-To-Text (faster-whisper).

Local speech recognition running on CPU (int8 quantization) to preserve
the 6 GB GPU VRAM entirely for the LLM.
"""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from config.settings import settings
from core.logging import get_logger

logger = get_logger("voice.stt")

_whisper_instance: WhisperModel | None = None


def get_stt_model() -> WhisperModel:
    """Lazy-load the Faster-Whisper model into memory on CPU (int8)."""
    global _whisper_instance
    if _whisper_instance is None:
        cfg = settings.voice.stt
        logger.info("loading_whisper_model", model=cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        _whisper_instance = WhisperModel(
            cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
            cpu_threads=4,
        )
        logger.info("whisper_model_loaded")
    return _whisper_instance


def transcribe_audio_array(audio_array: np.ndarray, sample_rate: int = 16000) -> str:
    """Transcribe a numpy float32/int16 audio array directly using Faster-Whisper."""
    if audio_array is None or len(audio_array) == 0:
        return ""

    # Ensure float32 normalized between -1.0 and 1.0
    if audio_array.dtype == np.int16:
        audio_data = audio_array.astype(np.float32) / 32768.0
    elif audio_array.dtype != np.float32:
        audio_data = audio_array.astype(np.float32)
    else:
        audio_data = audio_array

    # If multi-channel, downmix to mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    model = get_stt_model()
    segments, info = model.transcribe(
        audio_data,
        beam_size=1,  # Fast greedy search for low latency
        language="en",
        vad_filter=True,
    )

    text_parts = [segment.text.strip() for segment in segments]
    full_text = " ".join(text_parts).strip()
    logger.info("audio_transcribed", text=full_text, language=info.language, duration=info.duration)
    return full_text
