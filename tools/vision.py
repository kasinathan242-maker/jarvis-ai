"""JARVIS Lite — Vision & Screen Analysis Tool.

Wraps ScreenReader to provide on-demand screen understanding to the agent pipeline.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from security.permissions import RiskLevel
from tools.base import BaseTool
from vision.screen_reader import screen_reader


class AnalyzeScreenInput(BaseModel):
    question: str = Field(
        default="What is currently displayed on my screen?",
        description="Question or focal point to analyze from the current screen display.",
    )


class AnalyzeScreenOutput(BaseModel):
    description: str
    model_used: str
    image_path: str
    baseline_vram_mb: float
    post_vram_mb: float
    vram_delta_mb: float
    reclaimed: bool
    verified: bool = False


class AnalyzeScreenTool(BaseTool[AnalyzeScreenInput, AnalyzeScreenOutput]):
    """Captures the active screen, analyzes visual contents with on-demand vision model, and reclaims VRAM."""

    name = "analyze_screen"
    description = (
        "Capture and visually analyze the user's current desktop screen or window. "
        "Answers questions about open documents, websites, code editors, and visual UI controls."
    )
    risk_level = RiskLevel.SAFE
    timeout_seconds = 50.0
    input_schema = AnalyzeScreenInput
    output_schema = AnalyzeScreenOutput

    async def _execute(self, params: AnalyzeScreenInput) -> AnalyzeScreenOutput:
        result = await screen_reader.analyze_screen(question=params.question, force_unload=True)
        return AnalyzeScreenOutput(
            description=result["description"],
            model_used=result["model_used"],
            image_path=result["image_path"],
            baseline_vram_mb=result["baseline_vram_mb"],
            post_vram_mb=result["post_vram_mb"],
            vram_delta_mb=result["vram_delta_mb"],
            reclaimed=result["reclaimed"],
            verified=True,
        )

    async def verify(self, params: AnalyzeScreenInput, result: AnalyzeScreenOutput) -> bool:
        # Verify image file was created and description is non-empty
        img_exists = Path(result.image_path).exists()
        has_description = len(result.description.strip()) > 0
        verified = img_exists and has_description
        result.verified = verified
        return verified


__all__ = ["AnalyzeScreenTool", "AnalyzeScreenInput", "AnalyzeScreenOutput"]
