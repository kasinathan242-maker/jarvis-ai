"""JARVIS Lite — Circuit Breaker Pattern for External Model Services.

Prevents the application from freezing or endlessly retrying against dead
or unresponsive local model servers.
Transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.
"""

from __future__ import annotations

import enum
import time

from core.errors import CircuitBreakerOpenError
from core.logging import get_logger

logger = get_logger("ai.circuit_breaker")


class CircuitState(enum.StrEnum):
    CLOSED = "CLOSED"      # Healthy, calls pass through
    OPEN = "OPEN"          # Unhealthy, calls fail immediately
    HALF_OPEN = "HALF_OPEN"  # Testing recovery with single probe


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_time_seconds: float = 30.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_time_seconds = recovery_time_seconds

        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.last_state_change: float = time.monotonic()

    def record_success(self) -> None:
        """Called when an external call succeeds. Resets failure count and closes circuit."""
        if self.state != CircuitState.CLOSED:
            logger.info("circuit_breaker_recovered", circuit=self.name, previous_state=self.state)
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None

    def record_failure(self, error: Exception) -> None:
        """Called when an external call fails. Increments failure count and trips circuit if threshold reached."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.last_state_change = time.monotonic()
            logger.error(
                "circuit_breaker_tripped_open",
                circuit=self.name,
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
                error=str(error),
            )
        elif self.state == CircuitState.HALF_OPEN:
            # Re-trip immediately
            self.state = CircuitState.OPEN
            self.last_state_change = time.monotonic()
            logger.warn("circuit_breaker_probe_failed", circuit=self.name, error=str(error))

    def can_execute(self) -> bool:
        """Check if call can proceed.

        Returns True if CLOSED or HALF_OPEN. Raises CircuitBreakerOpenError if OPEN.
        """
        now = time.monotonic()

        if self.state == CircuitState.OPEN:
            elapsed = now - self.last_state_change
            if elapsed >= self.recovery_time_seconds:
                # Transition to HALF_OPEN to attempt recovery
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = now
                logger.info("circuit_breaker_half_open_probe", circuit=self.name)
                return True
            else:
                remaining = round(self.recovery_time_seconds - elapsed, 1)
                msg = (
                    f"Model server '{self.name}' circuit is OPEN "
                    f"({remaining}s until probe). Service is currently offline."
                )
                raise CircuitBreakerOpenError(msg)

        return True
