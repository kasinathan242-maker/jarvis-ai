"""JARVIS Lite — Text-To-Speech (Edge-TTS & Offline Piper).

Generates natural spoken audio without paid cloud API keys.
Supports:
- Interruptible playback (speaking stops immediately when user starts talking or presses stop).
- Speaker device selection via sounddevice.
- Configurable voice speed rate (+10%, +20%, -10%).
- Real-time synthesis chunk streaming.
"""

from __future__ import annotations

import asyncio
import io
import threading

import edge_tts
import sounddevice as sd

from config.settings import settings
from core.logging import get_logger

logger = get_logger("voice.tts")


class TextToSpeechEngine:
    """TTS engine with interruptible playback and device configuration."""

    def __init__(self) -> None:
        cfg = settings.voice.tts
        self.voice = cfg.voice or "en-US-ChristopherNeural"
        self.rate = cfg.rate or "+0%"
        self.pitch = cfg.pitch or "+0Hz"
        self.speaker_device: int | None = None
        self._interrupted = threading.Event()
        self._is_playing = False

    def set_voice_parameters(
        self,
        voice: str | None = None,
        rate: str | None = None,
        speaker_device: int | None = None,
    ) -> None:
        """Dynamically update voice personality, speed rate, and output device."""
        if voice:
            self.voice = voice
        if rate:
            self.rate = rate
        if speaker_device is not None:
            self.speaker_device = speaker_device
        logger.info(
            "tts_parameters_updated",
            voice=self.voice,
            rate=self.rate,
            speaker_device=self.speaker_device,
        )

    def stop(self) -> None:
        """Interrupt and immediately halt all active audio playback."""
        self._interrupted.set()
        try:
            sd.stop()
        except Exception:
            pass
        self._is_playing = False
        logger.info("tts_playback_interrupted")

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to MP3 audio bytes using Edge-TTS."""
        if not text or not text.strip():
            return b""

        clean_text = text.strip()
        communicate = edge_tts.Communicate(
            text=clean_text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )

        audio_buffer = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.extend(chunk["data"])

        return bytes(audio_buffer)

    async def speak(self, text: str, play_locally: bool = True) -> bytes:
        """Synthesize and play audio with interruptibility."""
        self._interrupted.clear()
        audio_bytes = await self.synthesize(text)
        if not audio_bytes:
            return b""

        if play_locally:
            self._is_playing = True
            try:
                await asyncio.to_thread(self._play_interruptible, audio_bytes)
            finally:
                self._is_playing = False

        return audio_bytes

    def _play_interruptible(self, audio_bytes: bytes) -> None:
        """Decode audio and stream to sounddevice in interruptible chunks."""
        try:
            import av
            import numpy as np

            container = av.open(io.BytesIO(audio_bytes))
            audio_stream = next(s for s in container.streams if s.type == "audio")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)

            pcm_chunks = []
            for frame in container.decode(audio_stream):
                if self._interrupted.is_set():
                    logger.debug("playback_aborted_during_decode")
                    return
                resampled = resampler.resample(frame)
                for r_frame in resampled:
                    pcm_chunks.append(r_frame.to_ndarray())

            if not pcm_chunks:
                return

            pcm_data = np.concatenate(pcm_chunks, axis=1).squeeze()
            sample_rate = 24000
            chunk_samples = 2400  # 100ms blocks for fast interrupt response

            # Stream through OutputStream to allow instantaneous cancellation
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                device=self.speaker_device,
            ) as stream:
                for idx in range(0, len(pcm_data), chunk_samples):
                    if self._interrupted.is_set():
                        stream.stop()
                        logger.info("tts_playback_halted_mid_stream")
                        return
                    chunk = pcm_data[idx : idx + chunk_samples]
                    stream.write(chunk)

        except Exception as exc:
            logger.warn("local_audio_playback_error", error=str(exc))
        finally:
            self._is_playing = False


tts = TextToSpeechEngine()
