"""JARVIS Lite — Browser and Web Research Tools.

Exposes web browsing, page content extraction with prompt injection defense,
and autonomous multi-source research capabilities to the agent tool registry.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from browser.controller import browser_controller
from browser.researcher import research_agent
from security.permissions import RiskLevel
from tools.base import BaseTool


class WebBrowseInput(BaseModel):
    url: str = Field(description="HTTP/HTTPS web page URL to inspect")
    timeout_seconds: float = Field(default=30.0, ge=5.0, le=60.0, description="Page load timeout")


class WebBrowseOutput(BaseModel):
    url: str
    title: str
    text_content: str
    link_count: int
    has_injection_risk: bool
    quarantined: bool
    verified: bool


class WebBrowseTool(BaseTool[WebBrowseInput, WebBrowseOutput]):
    name = "browse_web"
    description = "Navigate to a web page and extract clean text with automatic prompt-injection screening."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 35.0
    input_schema = WebBrowseInput
    output_schema = WebBrowseOutput

    async def _execute(self, params: WebBrowseInput) -> WebBrowseOutput:
        result = await browser_controller.extract_page(
            url=params.url,
            timeout_seconds=params.timeout_seconds,
        )
        return WebBrowseOutput(
            url=result.url,
            title=result.title,
            text_content=result.text_content,
            link_count=len(result.links),
            has_injection_risk=result.has_injection_risk,
            quarantined=result.quarantined,
            verified=True,
        )

    async def verify(self, params: WebBrowseInput, result: WebBrowseOutput) -> bool:
        verified = bool(len(result.text_content) > 0 or result.quarantined)
        result.verified = verified
        return verified


class ResearchInput(BaseModel):
    topic: str = Field(description="Research query or topic to investigate across multiple web sources")
    sources: list[str] | None = Field(default=None, description="Optional custom source URLs to consult")


class ResearchOutput(BaseModel):
    topic: str
    report_path: str
    report_exists: bool
    file_size_bytes: int
    verified: bool


class WebSearchResearchTool(BaseTool[ResearchInput, ResearchOutput]):
    name = "research_topic"
    description = "Conduct comprehensive multi-source web research, synthesize comparison tables, and save report."
    risk_level = RiskLevel.SAFE
    timeout_seconds = 180.0
    input_schema = ResearchInput
    output_schema = ResearchOutput

    async def _execute(self, params: ResearchInput) -> ResearchOutput:
        saved_path = await research_agent.conduct_research(
            topic=params.topic,
            sources=params.sources,
        )
        size = saved_path.stat().st_size if saved_path.exists() else 0
        return ResearchOutput(
            topic=params.topic,
            report_path=str(saved_path),
            report_exists=saved_path.exists(),
            file_size_bytes=size,
            verified=saved_path.exists() and size > 0,
        )

    async def verify(self, params: ResearchInput, result: ResearchOutput) -> bool:
        path = Path(result.report_path)
        verified = path.exists() and path.stat().st_size > 100
        result.verified = verified
        return verified
