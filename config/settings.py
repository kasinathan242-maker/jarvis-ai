import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    ELEVENLABS_API_KEY: str = os.getenv('ELEVENLABS_API_KEY', '')
    ELEVENLABS_VOICE_ID: str = os.getenv('ELEVENLABS_VOICE_ID', '21m00Tcm4TlvDq8ikWAM')
    OPENWEATHER_API_KEY: str = os.getenv('OPENWEATHER_API_KEY', '')
    NEWS_API_KEY: str = os.getenv('NEWS_API_KEY', '')
    EMAIL_ADDRESS: str = os.getenv('EMAIL_ADDRESS', '')
    EMAIL_APP_PASSWORD: str = os.getenv('EMAIL_APP_PASSWORD', '')
    GOOGLE_CREDENTIALS_PATH: str = os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials/google_credentials.json')
    WHISPER_MODEL: str = os.getenv('WHISPER_MODEL', 'base')
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    DB_PATH: str = os.getenv('DB_PATH', 'jarvis_memory.db')

settings = Settings()
