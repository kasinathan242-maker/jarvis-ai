"""Tests for Phase 4 Vision Subsystem and Screen Understanding."""

from __future__ import annotations

import pytest

from intent.classifier import classifier
from intent.schemas import IntentType
from tools.registry import registry
from vision.screen_reader import screen_reader
from voice.pipeline import pipeline


@pytest.mark.asyncio
async def test_screen_reader_capture_and_reclaim_vram() -> None:
    """Test screen capture, analysis, and VRAM baseline tracking."""
    result = await screen_reader.analyze_screen(
        question="Describe the active applications on screen.",
        force_unload=True,
    )

    assert "description" in result
    assert len(result["description"]) > 10
    assert "baseline_vram_mb" in result
    assert "post_vram_mb" in result
    assert "model_used" in result
    assert result["reclaimed"] is True


@pytest.mark.asyncio
async def test_analyze_screen_tool_registered_and_dispatched() -> None:
    """Test that analyze_screen tool is registered and dispatches properly."""
    assert "analyze_screen" in registry

    res = await registry.dispatch("analyze_screen", {"question": "What's on screen?"}, user_approved=True)
    assert res.success is True
    assert res.data is not None


@pytest.mark.asyncio
async def test_whats_on_my_screen_voice_pipeline() -> None:
    """Test 'what's on my screen' voice query routing and response."""
    intent = await classifier.classify("What's on my screen right now?")
    assert intent.type == IntentType.ANALYZE_SCREEN

    response = await pipeline.process_command("What's on my screen right now?", play_speech=False)
    assert "screen:" in response.lower()
