import os
import tempfile
import numpy as np
from config.settings import settings

_model = None

def _get_model():
    global _model
    if _model is None:
        try:
            import whisper
            print(f'[STT] Loading Whisper model: {settings.WHISPER_MODEL}...')
            _model = whisper.load_model(settings.WHISPER_MODEL)
            print('[STT] Whisper model loaded successfully.')
        except Exception as e:
            print(f'[STT Warning] Whisper model load failed: {e}')
            _model = False
    return _model

def record_audio(duration: int = 5, sample_rate: int = 16000):
    try:
        import sounddevice as sd
        print(f'[STT] Listening for {duration} seconds...')
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        return audio, sample_rate
    except Exception as e:
        print(f'[STT Warning] Audio recording error: {e}')
        return None, sample_rate

def transcribe(duration: int = 5) -> str:
    audio, sr = record_audio(duration)
    if audio is None:
        print('[STT] Audio device unavailable. Enter text manually if needed.')
        return ""

    model = _get_model()
    if not model:
        print('[STT] Whisper model unavailable.')
        return ""

    try:
        import scipy.io.wavfile as wav
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            wav.write(tmp.name, sr, audio)
            tmp_path = tmp.name

        try:
            result = model.transcribe(tmp_path, language='en', fp16=False)
            text = result.get('text', '').strip()
            print(f'[STT] Transcribed: "{text}"')
            return text
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        print(f'[STT] Transcription error: {e}')
        return ""
