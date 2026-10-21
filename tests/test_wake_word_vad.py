"""JARVIS Lite — Wake Word and VAD Verification Tests."""

import time

import numpy as np

from voice.vad import VoiceActivityDetector
from voice.wake_word import WakeWordDetector


def test_wake_word_inference_and_low_cpu() -> None:
    detector = WakeWordDetector()
    loaded = detector._init_model()
    assert loaded is True

    # 80ms chunk at 16kHz
    chunk = np.zeros(1280, dtype=np.int16)

    t0 = time.perf_counter()
    score = detector.predict_chunk(chunk)
    elapsed = time.perf_counter() - t0

    assert 0.0 <= score <= 1.0
    # On laptop CPU, a single 80ms inference takes less than 20ms (typically 1-3ms)
    assert elapsed < 0.05


def test_voice_activity_detector_silence_vs_energy() -> None:
    vad = VoiceActivityDetector()

    # Silence
    silent_chunk = np.zeros(1024, dtype=np.float32)
    is_speech_silent, score_silent = vad.is_speech(silent_chunk)
    assert is_speech_silent is False
    assert score_silent < 0.2

    # High energy audio simulating speech signal
    t = np.linspace(0, 1024 / 16000, 1024)
    speech_like = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    is_speech_loud, score_loud = vad.is_speech(speech_like)
    assert is_speech_loud is True
    assert score_loud > 0.3
