import asyncio
import sys
import uvicorn
from core.jarvis import Jarvis
from voice.listener import transcribe
from config.settings import settings

async def run_voice_mode():
    jarvis = Jarvis()
    await jarvis.run_loop(transcribe_fn=lambda: transcribe(duration=6))

def run_server_mode():
    uvicorn.run("api.server:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--voice":
        asyncio.run(run_voice_mode())
    else:
        print(f"[JARVIS AI] Launching Control Center & API Server on http://localhost:{settings.API_PORT}...")
        run_server_mode()
