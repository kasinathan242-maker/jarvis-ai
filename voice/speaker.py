import httpx
import io
from config.settings import settings

async def speak(text: str) -> None:
    if not text:
        return

    print(f'[TTS Response]: "{text}"')

    if not settings.ELEVENLABS_API_KEY:
        print('[TTS Info] ELEVENLABS_API_KEY not configured in .env. Outputting text response.')
        return

    ELEVENLABS_URL = f'https://api.elevenlabs.io/v1/text-to-speech/{settings.ELEVENLABS_VOICE_ID}'
    headers = {
        'xi-api-key': settings.ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
    }
    payload = {
        'text': text,
        'model_id': 'eleven_monolingual_v1',
        'voice_settings': {'stability': 0.5, 'similarity_boost': 0.75},
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(ELEVENLABS_URL, json=payload, headers=headers)
            response.raise_for_status()

            try:
                import sounddevice as sd
                import scipy.io.wavfile as wav
                audio_data = io.BytesIO(response.content)
                sr, audio_array = wav.read(audio_data)
                sd.play(audio_array, samplerate=sr)
                sd.wait()
                print('[TTS] Speech playback complete.')
            except Exception as play_err:
                print(f'[TTS Warning] Audio playback skipped: {play_err}')
    except Exception as e:
        print(f'[TTS Warning] ElevenLabs TTS error: {e}')
