"""JARVIS Lite — FastAPI Backend & Local Control Server.

Enforces Rule 9: Binds to 127.0.0.1 ONLY. Requires Bearer Token authentication
generated on first run and stored in Windows Credential Manager.
Exposes REST and WebSocket endpoints for Chat, Tools, Approvals, Telemetry, and Emergency Stop.
Mounts HUD frontend at '/' and '/hud'.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.remote import remote_router
from api.ws import handle_chat_websocket
from config.settings import settings
from core.emergency_stop import emergency_stop
from core.lifecycle import lifecycle
from core.logging import get_logger
from memory.conversation_store import get_recent_history
from memory.long_term_memory import delete_memory, list_all_memories
from security.approval import approval_manager
from security.audit_log import get_recent_audit_records
from security.secrets import verify_api_token
from system.health import system_health
from tools.registry import registry
from voice.devices import audio_devices
from voice.pipeline import pipeline
from voice.text_to_speech import tts
from voice.wake_word import wake_word

logger = get_logger("api.server")
security_bearer = HTTPBearer(auto_error=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = PROJECT_ROOT / "ui"


def require_auth(credentials: HTTPAuthorizationCredentials | None = Security(security_bearer)) -> str:
    """Validate incoming HTTP Bearer token against Windows Credential Manager."""
    if not credentials or not verify_api_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI startup and shutdown lifecycle hooks."""
    await lifecycle.startup()
    await system_health.start_periodic_monitoring(interval_seconds=3.0)
    # Register global hotkey
    emergency_stop.initialize_hotkey()
    yield
    system_health.stop_monitoring()
    emergency_stop.cleanup()
    wake_word.stop()
    await lifecycle.shutdown()


app = FastAPI(
    title="JARVIS Lite API",
    version=settings.system.version,
    description="Local-first Personal AI Operating System for Windows 11.",
    lifespan=lifespan,
)

# CORS restricted strictly to localhost and Tauri webview
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Schemas
class CommandRequest(BaseModel):
    text: str = Field(min_length=1, description="Command or query string")
    play_speech: bool = Field(default=False, description="Speak response aloud via TTS")
    user_approved: bool | None = Field(default=None, description="Pre-authorization decision for sensitive tools")


class CommandResponse(BaseModel):
    query: str
    response: str
    timestamp: str


class ApprovalDecisionRequest(BaseModel):
    approved: bool = Field(description="Decision: True to authorize execution, False to reject")
    source: str = Field(default="ui", description="Origin: ui, voice, or api")


class AudioDeviceConfigRequest(BaseModel):
    input_device_index: int | None = Field(default=None)
    output_device_index: int | None = Field(default=None)


class VoiceSettingsRequest(BaseModel):
    voice: str | None = Field(default=None)
    rate: str | None = Field(default=None)
    speaker_device: int | None = Field(default=None)


# Public / Health endpoints
@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Public unauthenticated health ping endpoint."""
    return {
        "status": "healthy" if lifecycle.is_running else "starting",
        "service": "JARVIS Lite",
        "version": settings.system.version,
        "env": settings.system.env,
    }


# Authenticated System & Command endpoints
@app.get(f"{settings.server.api_prefix}/status")
async def system_status(token: str = Depends(require_auth)) -> dict[str, Any]:
    """Authenticated full system telemetry status."""
    return await system_health.get_latest_health()


@app.post(f"{settings.server.api_prefix}/command", response_model=CommandResponse)
async def execute_command(
    req: CommandRequest,
    token: str = Depends(require_auth),
) -> CommandResponse:
    """Execute a command through verified tool & reasoning pipeline."""
    from datetime import datetime

    resp = await pipeline.process_command(
        req.text,
        play_speech=req.play_speech,
        user_approved=req.user_approved,
    )
    return CommandResponse(
        query=req.text,
        response=resp,
        timestamp=datetime.now(UTC).isoformat(),
    )


# Emergency Stop
@app.post(f"{settings.server.api_prefix}/emergency_stop")
async def trigger_emergency_stop(token: str = Depends(require_auth)) -> dict[str, Any]:
    """Instantly cancel all running tasks and pending approvals in < 1 second."""
    return await emergency_stop.trigger_halt(source="api")


# Security Approvals
@app.get(f"{settings.server.api_prefix}/security/approvals")
async def list_pending_approvals(token: str = Depends(require_auth)) -> list[dict[str, Any]]:
    """List pending authorization tickets awaiting user approval."""
    return approval_manager.get_pending()


@app.post(f"{settings.server.api_prefix}/security/approvals/{{request_id}}/decision")
async def resolve_approval(
    request_id: str,
    req: ApprovalDecisionRequest,
    token: str = Depends(require_auth),
) -> dict[str, Any]:
    """Approve or reject a pending tool execution ticket."""
    success = approval_manager.resolve_ticket(request_id, approved=req.approved, source=req.source)
    if not success:
        raise HTTPException(status_code=404, detail=f"Approval ticket '{request_id}' not found or already resolved.")
    return {"request_id": request_id, "resolved": True, "approved": req.approved}


# Telemetry & Audio Devices
@app.get(f"{settings.server.api_prefix}/telemetry/health")
async def get_telemetry_health(token: str = Depends(require_auth)) -> dict[str, Any]:
    """Get latest system and network telemetry metrics."""
    return await system_health.get_latest_health()


@app.get(f"{settings.server.api_prefix}/telemetry/devices")
async def get_audio_devices(token: str = Depends(require_auth)) -> dict[str, Any]:
    """List available audio input microphones and output speakers."""
    return {
        "inputs": [d.to_dict() for d in audio_devices.list_input_devices()],
        "outputs": [d.to_dict() for d in audio_devices.list_output_devices()],
    }


@app.post(f"{settings.server.api_prefix}/telemetry/devices")
async def configure_audio_devices(
    req: AudioDeviceConfigRequest,
    token: str = Depends(require_auth),
) -> dict[str, Any]:
    """Configure default audio input/output devices."""
    audio_devices.set_default_devices(req.input_device_index, req.output_device_index)
    return {"configured": True, "input": req.input_device_index, "output": req.output_device_index}


# Voice & Wake Word
@app.get(f"{settings.server.api_prefix}/voice/wake_word/status")
async def wake_word_status(token: str = Depends(require_auth)) -> dict[str, Any]:
    return {"is_listening": wake_word.is_listening}


@app.post(f"{settings.server.api_prefix}/voice/wake_word/toggle")
async def toggle_wake_word(token: str = Depends(require_auth)) -> dict[str, Any]:
    if wake_word.is_listening:
        wake_word.stop()
        return {"is_listening": False, "action": "stopped"}
    else:
        started = wake_word.start()
        return {"is_listening": started, "action": "started" if started else "failed"}


@app.post(f"{settings.server.api_prefix}/voice/settings")
async def update_voice_settings(
    req: VoiceSettingsRequest,
    token: str = Depends(require_auth),
) -> dict[str, Any]:
    tts.set_voice_parameters(voice=req.voice, rate=req.rate, speaker_device=req.speaker_device)
    return {"updated": True, "voice": tts.voice, "rate": tts.rate, "speaker_device": tts.speaker_device}


# Tools, Audit & Memory
@app.get(f"{settings.server.api_prefix}/tools")
async def list_tools(token: str = Depends(require_auth)) -> list[dict[str, Any]]:
    """List all registered tools, risk levels, and parameter schemas."""
    return registry.list_tools()


@app.get(f"{settings.server.api_prefix}/audit")
async def get_audit_logs(limit: int = 50, token: str = Depends(require_auth)) -> list[dict[str, Any]]:
    """Retrieve immutable tool execution audit records."""
    return get_recent_audit_records(limit=limit)


@app.get(f"{settings.server.api_prefix}/history")
async def get_history(limit: int = 20, token: str = Depends(require_auth)) -> list[dict[str, Any]]:
    """Retrieve recent conversation history from SQLite."""
    return get_recent_history(limit=limit)


@app.get(f"{settings.server.api_prefix}/memories")
async def get_memories(token: str = Depends(require_auth)) -> list[dict[str, Any]]:
    """List all long-term vector memories stored in ChromaDB."""
    return list_all_memories()


@app.delete(f"{settings.server.api_prefix}/memories/{{memory_id}}")
async def remove_memory(memory_id: str, token: str = Depends(require_auth)) -> dict[str, Any]:
    """Delete a memory entry from ChromaDB."""
    success = delete_memory(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")
    return {"deleted": True, "id": memory_id}


# Mount streaming WebSocket
app.add_api_websocket_route(f"{settings.server.api_prefix}/ws/chat", handle_chat_websocket)

# Mount remote companion router
app.include_router(remote_router)

# Mount HUD static directory if present
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

    @app.get("/")
    @app.get("/hud")
    async def serve_hud() -> FileResponse:
        return FileResponse(str(UI_DIR / "index.html"))

    @app.get("/remote")
    async def serve_remote_companion() -> FileResponse:
        remote_html = UI_DIR / "remote.html"
        if not remote_html.exists():
            return FileResponse(str(UI_DIR / "index.html"))
        return FileResponse(str(remote_html))
