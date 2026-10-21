"""JARVIS Lite — Interactive Security Approval Manager.

Enforces Rule 6 & Rule 8:
Manages pending user approvals for CONFIRM and DANGEROUS actions
(destructive file operations, non-allowlisted shell commands).
Supports asynchronous resolution via UI dialog, REST API, WebSocket, or Voice confirmation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from core.errors import SecurityViolationError
from core.event_bus import bus
from core.logging import get_logger
from security.permissions import RiskLevel

logger = get_logger("security.approval")

EVENT_APPROVAL_REQUESTED = "security.approval_requested"
EVENT_APPROVAL_RESOLVED = "security.approval_resolved"


@dataclass
class ApprovalTicket:
    request_id: str
    tool_name: str
    risk_level: str
    parameters: dict[str, Any]
    description: str
    created_at: float
    timeout_seconds: float
    resolved: bool = False
    approved: bool = False
    resolved_by: str | None = None  # "ui", "voice", "timeout", "emergency_stop"

    @property
    def ticket_id(self) -> str:
        return self.request_id

    @property
    def action_type(self) -> str:
        return self.tool_name

    @property
    def target_resource(self) -> str:
        return str(self.parameters.get("target", self.description))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ticket_id"] = self.request_id
        d["action_type"] = self.tool_name
        d["target_resource"] = self.target_resource
        return d

    def model_dump(self) -> dict[str, Any]:
        return self.to_dict()


class ApprovalManager:
    """Manages pending authorization requests for sensitive actions."""

    def __init__(self) -> None:
        self._pending_tickets: dict[str, ApprovalTicket] = {}
        self._futures: dict[str, asyncio.Future[bool]] = {}

    def get_pending(self) -> list[dict[str, Any]]:
        """List active pending approval tickets."""
        return [t.to_dict() for t in self._pending_tickets.values() if not t.resolved]

    def list_pending_tickets(self) -> list[ApprovalTicket]:
        """Return list of active pending approval tickets."""
        return [t for t in self._pending_tickets.values() if not t.resolved]

    def create_ticket(
        self,
        action_type: str,
        target_resource: str = "",
        details: dict[str, Any] | None = None,
        risk_level: RiskLevel = RiskLevel.CONFIRM,
        timeout_seconds: float = 60.0,
    ) -> ApprovalTicket:
        """Create and register an approval ticket synchronously."""
        request_id = str(uuid.uuid4())
        ticket = ApprovalTicket(
            request_id=request_id,
            tool_name=action_type,
            risk_level=risk_level.value,
            parameters={"target": target_resource, **(details or {})},
            description=f"Action '{action_type}' on '{target_resource}'",
            created_at=time.time(),
            timeout_seconds=timeout_seconds,
        )
        self._pending_tickets[request_id] = ticket
        return ticket

    def get_ticket(self, request_id: str) -> ApprovalTicket | None:
        return self._pending_tickets.get(request_id)

    async def request_approval(
        self,
        tool_name: str,
        risk_level: RiskLevel,
        parameters: dict[str, Any],
        description: str,
        timeout_seconds: float = 30.0,
    ) -> bool:
        """Create a pending approval ticket and wait for user authorization.

        Returns:
            True if user explicitly approved.
            False if rejected, timed out, or cancelled.
        """
        if risk_level == RiskLevel.FORBIDDEN:
            logger.warn("approval_denied_forbidden", tool_name=tool_name, parameters=parameters)
            raise SecurityViolationError(f"Action '{tool_name}' is FORBIDDEN and cannot be approved.")

        request_id = str(uuid.uuid4())[:8]
        ticket = ApprovalTicket(
            request_id=request_id,
            tool_name=tool_name,
            risk_level=risk_level.value,
            parameters=parameters,
            description=description,
            created_at=time.time(),
            timeout_seconds=timeout_seconds,
        )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()

        self._pending_tickets[request_id] = ticket
        self._futures[request_id] = future

        logger.info(
            "approval_requested",
            request_id=request_id,
            tool_name=tool_name,
            risk=risk_level.value,
            parameters=parameters,
        )

        # Broadcast approval request to UI and voice
        await bus.publish(EVENT_APPROVAL_REQUESTED, ticket.to_dict())

        try:
            # Await user decision with timeout
            approved = await asyncio.wait_for(future, timeout=timeout_seconds)
            ticket.resolved = True
            ticket.approved = approved
            return approved
        except TimeoutError:
            ticket.resolved = True
            ticket.approved = False
            ticket.resolved_by = "timeout"
            logger.warn("approval_timed_out", request_id=request_id, tool_name=tool_name)
            await bus.publish(EVENT_APPROVAL_RESOLVED, ticket.to_dict())
            return False
        except asyncio.CancelledError:
            ticket.resolved = True
            ticket.approved = False
            ticket.resolved_by = "emergency_stop"
            await bus.publish(EVENT_APPROVAL_RESOLVED, ticket.to_dict())
            raise
        finally:
            self._futures.pop(request_id, None)

    def resolve_ticket(self, request_id: str, approved: bool, source: str = "ui") -> bool:
        """Resolve a pending ticket with an approval or rejection decision."""
        ticket = self._pending_tickets.get(request_id)
        future = self._futures.get(request_id)

        if not ticket or not future or future.done():
            logger.warn("ticket_not_found_or_already_resolved", request_id=request_id)
            return False

        ticket.resolved = True
        ticket.approved = approved
        ticket.resolved_by = source

        logger.info(
            "approval_ticket_resolved",
            request_id=request_id,
            approved=approved,
            source=source,
            tool=ticket.tool_name,
        )

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(bus.publish(EVENT_APPROVAL_RESOLVED, ticket.to_dict()))
        except RuntimeError:
            pass

        future.set_result(approved)
        return True

    def resolve_latest_by_voice(self, approved: bool) -> ApprovalTicket | None:
        """Resolve the most recent pending ticket via voice confirmation."""
        active = [t for t in self._pending_tickets.values() if not t.resolved]
        if not active:
            return None
        # Sort by creation time descending (most recent first)
        active.sort(key=lambda t: t.created_at, reverse=True)
        latest = active[0]
        self.resolve_ticket(latest.request_id, approved=approved, source="voice")
        return latest

    def cancel_all_pending(self, reason: str = "emergency_stop") -> int:
        """Cancel all pending approval tickets immediately."""
        cancelled = 0
        for req_id, future in list(self._futures.items()):
            if not future.done():
                ticket = self._pending_tickets.get(req_id)
                if ticket:
                    ticket.resolved = True
                    ticket.approved = False
                    ticket.resolved_by = reason
                future.set_result(False)
                cancelled += 1
        self._pending_tickets.clear()
        self._futures.clear()
        return cancelled


# Global approval manager singleton
approval_manager = ApprovalManager()
