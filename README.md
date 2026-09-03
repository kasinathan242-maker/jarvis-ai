# JARVIS AI — Voice-Driven Autonomous Desktop AI Agent Stack

JARVIS AI is an autonomous, voice-driven desktop agent built with Python, AsyncIO, SQLite, Whisper STT, ElevenLabs TTS, and FastAPI.

## Key Technical Features

- **Voice Pipeline**: Real-time microphone audio capture -> OpenAI Whisper STT -> NLP Intent Router -> Async Handlers -> ElevenLabs TTS response.
- **Intent Routing Engine**: Match natural language commands to weather, news, music, email, calendar, and tool execution modules.
- **SQLite Memory**: Persistent multi-table store for conversation context and key-value memory recall.
- **FastAPI Control Center**: Interactive dark glassmorphic control dashboard at `http://localhost:8000/` with web voice recording, instant execution, and memory viewer.

## How to Run

1. **Start Dashboard & Server**:
   ```bash
   .\venv\Scripts\python.exe main.py
   ```
   Open `http://localhost:8000` in your web browser.

2. **Start Continuous Voice Loop Mode**:
   ```bash
   .\venv\Scripts\python.exe main.py --voice
   ```

3. **Explore API Documentation**:
   Navigate to `http://localhost:8000/docs` for Swagger UI endpoints.
