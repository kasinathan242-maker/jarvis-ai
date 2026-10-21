"""Tests for End-to-End Pipeline Execution and Latency Guarantees."""

import time

import pytest

from voice.pipeline import pipeline


@pytest.mark.asyncio
async def test_what_time_is_it_latency_under_4_seconds() -> None:
    """Acceptance Test: 'what time is it' produces correct answer in under 4 seconds end-to-end."""
    start = time.monotonic()
    response = await pipeline.process_command("what time is it", play_speech=False)
    elapsed = time.monotonic() - start

    assert elapsed < 4.0, f"Latency {elapsed:.2f}s exceeded 4.0s acceptance limit"
    assert "time is" in response.lower()
    print(f"\n[LATENCY BENCHMARK] 'what time is it' processed in {elapsed:.3f} seconds.")


@pytest.mark.asyncio
async def test_emergency_stop_command() -> None:
    response = await pipeline.process_command("jarvis stop everything", play_speech=False)
    assert "emergency stop executed" in response.lower()


@pytest.mark.asyncio
async def test_greeting_command() -> None:
    response = await pipeline.process_command("hello jarvis", play_speech=False)
    assert "jarvis lite" in response.lower()
