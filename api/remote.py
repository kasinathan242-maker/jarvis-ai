"""JARVIS Lite — Remote Companion & Mobile Control API.

Enforces strict Bearer Token authentication via Windows Credential Manager.
Provides endpoints for remote telemetry, approval resolution, voice/text command dispatch,
emergency stop, and personalization management.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from core.emergency_stop import emergency_stop
from core.logging import get_logger
from maintenance.backup import backup_manager
from personalization.manager import personalization_manager
from proactive.engine import proactive_engine
from security.approval import approval_manager
from security.secrets import verify_api_token
from system.health import system_health
from system.resource_monitor import resource_monitor
from voice.pipeline import pipeline

logger = get_logger("api.remote")
security_bearer = HTTPBearer(auto_error=False)

remote_router = APIRouter(prefix="/api/remote", tags=["Remote Dashboard"])


def require_remote_token(credentials: HTTPAuthorizationCredentials | None = Security(security_bearer)) -> str:
    """Validate Bearer token. Rejects unauthenticated requests with HTTP 401."""
    if not credentials or not verify_api_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Valid API Bearer token is required to access remote dashboard.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


class RemoteCommandRequest(BaseModel):
    command: str = Field(min_length=1, max_length=500, description="Natural language command to execute")


class ResolveApprovalRequest(BaseModel):
    approved: bool = Field(description="True to approve execution, False to deny")


@remote_router.get("/status")
async def get_remote_status(token: str = Depends(require_remote_token)) -> dict[str, Any]:
    """Return live system telemetry, quiet hours, and alerts for remote companion."""
    snap = resource_monitor.get_snapshot()
    alerts = system_health.get_active_alerts()
    pending_approvals = approval_manager.list_pending_tickets()
    is_quiet = proactive_engine.is_quiet_hours()

    return {
        "status": "online",
        "cpu_percent": snap.cpu_percent,
        "ram_percent": snap.ram_percent,
        "ram_used_mb": snap.ram_used_mb,
        "ram_total_mb": snap.ram_total_mb,
        "gpu": asdict(snap.gpu) if snap.gpu else None,
        "battery": asdict(snap.battery) if snap.battery else None,
        "disk_percent": snap.disk_percent,
        "alerts": alerts,
        "pending_approvals_count": len(pending_approvals),
        "is_quiet_hours": is_quiet,
    }


@remote_router.get("/approvals")
async def get_remote_approvals(token: str = Depends(require_remote_token)) -> list[dict[str, Any]]:
    """List pending approval tickets requiring user action."""
    tickets = approval_manager.list_pending_tickets()
    return [t.model_dump() for t in tickets]


@remote_router.post("/approvals/{ticket_id}/resolve")
async def resolve_remote_approval(
    ticket_id: str,
    req: ResolveApprovalRequest,
    token: str = Depends(require_remote_token),
) -> dict[str, Any]:
    """Approve or deny an action ticket remotely."""
    success = approval_manager.resolve_ticket(request_id=ticket_id, approved=req.approved)
    if not success:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found or already resolved.")
    return {"ticket_id": ticket_id, "resolved": True, "approved": req.approved}


@remote_router.post("/command")
async def post_remote_command(
    req: RemoteCommandRequest,
    token: str = Depends(require_remote_token),
) -> dict[str, Any]:
    """Execute natural language or voice command remotely and return response."""
    response = await pipeline.process_command(req.command, play_speech=False)
    return {"command": req.command, "response": response}


@remote_router.post("/emergency-stop")
async def trigger_remote_emergency_stop(token: str = Depends(require_remote_token)) -> dict[str, Any]:
    """Trigger system-wide emergency halt remotely in under 1 second."""
    res = emergency_stop.trigger(reason="Remote Dashboard Emergency Halt")
    return res


@remote_router.get("/personalization")
async def get_remote_personalization(token: str = Depends(require_remote_token)) -> list[dict[str, Any]]:
    """List all explicitly learned user preferences."""
    return personalization_manager.list_preferences()


@remote_router.post("/personalization/reset")
async def reset_remote_personalization(token: str = Depends(require_remote_token)) -> dict[str, Any]:
    """Reset all learned personalization preferences to defaults."""
    cleared = personalization_manager.reset_personalization()
    return {"reset": True, "cleared_count": cleared}


@remote_router.post("/backup")
async def trigger_remote_backup(token: str = Depends(require_remote_token)) -> dict[str, Any]:
    """Create a complete disaster recovery backup archive."""
    archive_path = backup_manager.create_backup()
    return {"backup_created": True, "archive_name": archive_path.name, "path": str(archive_path)}
