"""Tests for Circuit Breaker and Model Server Failure Resilience."""

import time

import pytest

from ai.circuit_breaker import CircuitBreaker, CircuitState
from ai.ollama_client import OllamaClient
from core.errors import CircuitBreakerOpenError, ModelUnavailableError


def _check_state(cb: CircuitBreaker, expected: CircuitState) -> None:
    assert cb.state == expected


def test_circuit_breaker_trips_and_recovers() -> None:
    cb = CircuitBreaker("test_breaker", failure_threshold=3, recovery_time_seconds=0.2)
    _check_state(cb, CircuitState.CLOSED)
    assert cb.can_execute() is True

    # 1st failure
    cb.record_failure(Exception("Failure 1"))
    _check_state(cb, CircuitState.CLOSED)
    assert cb.can_execute() is True

    # 2nd failure
    cb.record_failure(Exception("Failure 2"))
    _check_state(cb, CircuitState.CLOSED)
    assert cb.can_execute() is True

    # 3rd failure trips the breaker OPEN
    cb.record_failure(Exception("Failure 3"))
    _check_state(cb, CircuitState.OPEN)

    # While OPEN, can_execute() raises CircuitBreakerOpenError immediately
    with pytest.raises(CircuitBreakerOpenError):
        cb.can_execute()

    # Wait for recovery window (0.2s)
    time.sleep(0.25)

    # After timeout, enters HALF_OPEN
    assert cb.can_execute() is True
    _check_state(cb, CircuitState.HALF_OPEN)

    # Successful probe restores to CLOSED
    cb.record_success()
    _check_state(cb, CircuitState.CLOSED)
    assert cb.failure_count == 0


@pytest.mark.asyncio
async def test_ollama_client_offline_does_not_freeze() -> None:
    """Verify Ollama client fails fast with ModelUnavailableError when server is unreachable."""
    client = OllamaClient(
        base_url="http://127.0.0.1:59999",  # Non-existent port
        connect_timeout=0.2,
        read_timeout=0.5,
        max_retries=2,
    )

    start = time.monotonic()
    with pytest.raises(ModelUnavailableError) as exc_info:
        await client.chat([{"role": "user", "content": "ping"}])

    duration = time.monotonic() - start
    # Must fail fast without hanging
    assert duration < 3.0
    assert "unavailable" in str(exc_info.value).lower()
