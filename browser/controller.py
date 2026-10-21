"""JARVIS Lite — Playwright Headless Browser Controller.

Provides secure, sandboxed web page navigation, DOM text extraction, link discovery,
and screenshot capture. Enforces prompt injection scanning on every visited page.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import Browser, Playwright, async_playwright

from core.errors import ToolExecutionError
from core.logging import get_logger
from security.audit_log import record_audit_event
from security.injection_guard import is_adversarial, sanitize_for_prompt

logger = get_logger("browser.controller")


@dataclass
class PageExtractionResult:
    url: str
    title: str
    text_content: str
    links: list[dict[str, str]]
    has_injection_risk: bool
    quarantined: bool


class BrowserController:
    """Async Playwright browser manager with security sanitization."""

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()

    async def _ensure_browser(self) -> Browser:
        running_loop = asyncio.get_running_loop()
        if self._loop is not running_loop:
            self._playwright = None
            self._browser = None
            self._loop = running_loop
            self._lock = asyncio.Lock()

        async with self._lock:
            if self._playwright is None:
                self._playwright = await async_playwright().start()
            if self._browser is None or not self._browser.is_connected():
                self._browser = await self._playwright.chromium.launch(
                    headless=self.headless,
                    args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
                )
            return self._browser

    async def extract_page(
        self,
        url: str,
        timeout_seconds: float = 30.0,
        max_chars: int = 15000,
    ) -> PageExtractionResult:
        """Navigate to a URL, extract title and text, and run prompt-injection detection."""
        browser = await self._ensure_browser()
        try:
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 JARVIS-Lite/1.0"
            )
        except Exception:
            # Stale context or disconnected browser; force reconnect
            self._browser = None
            self._playwright = None
            browser = await self._ensure_browser()
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 JARVIS-Lite/1.0"
            )
        page = await context.new_page()

        try:
            logger.info("browser_navigating", url=url)
            await page.goto(url, timeout=int(timeout_seconds * 1000), wait_until="domcontentloaded")
            title = await page.title()
            content_html = await page.content()

            # Parse with BeautifulSoup to strip scripts and styles
            soup = BeautifulSoup(content_html, "html.parser")
            for script_or_style in soup(["script", "style", "noscript", "svg"]):
                script_or_style.decompose()

            raw_text = soup.get_text(separator="\n", strip=True)

            # Discover links
            links: list[dict[str, str]] = []
            for a_tag in soup.find_all("a", href=True):
                href = str(a_tag["href"]).strip()
                text = str(a_tag.get_text()).strip()
                if href.startswith("http") and text:
                    links.append({"text": text[:60], "url": href})
                    if len(links) >= 30:
                        break

            # Security Guardrail: Check for adversarial prompt injections
            has_injection = is_adversarial(raw_text)
            quarantined = False

            if has_injection:
                logger.warn("prompt_injection_detected_on_webpage", url=url)
                # Immutable security audit log entry
                record_audit_event(
                    tool_name="browser_extract",
                    parameters={"url": url, "injection_flagged": True},
                    execution_status="BLOCKED",
                    error_message="Prompt injection detected on web page. Content quarantined.",
                )
                quarantined = True
                safe_text = (
                    f"[SECURITY WARNING: Malicious prompt injection detected on page '{url}'. "
                    f"Content has been quarantined to protect your system.]\n\n"
                    + sanitize_for_prompt(raw_text[:max_chars])
                )
            else:
                safe_text = sanitize_for_prompt(raw_text[:max_chars])

            return PageExtractionResult(
                url=url,
                title=title,
                text_content=safe_text,
                links=links,
                has_injection_risk=has_injection,
                quarantined=quarantined,
            )

        except Exception as exc:
            logger.error("browser_extraction_failed", url=url, error=str(exc))
            raise ToolExecutionError(f"Failed to extract web content from {url}: {exc}") from exc
        finally:
            await page.close()
            await context.close()

    async def capture_screenshot(
        self,
        url: str,
        output_path: Path,
        timeout_seconds: float = 30.0,
    ) -> Path:
        """Navigate to a URL and capture a full-page screenshot."""
        browser = await self._ensure_browser()
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        try:
            await page.goto(url, timeout=int(timeout_seconds * 1000), wait_until="networkidle")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(output_path), full_page=False)
            return output_path
        except Exception as exc:
            raise ToolExecutionError(f"Failed to capture web screenshot of {url}: {exc}") from exc
        finally:
            await page.close()
            await context.close()

    async def close(self) -> None:
        """Close browser instance and clean up Playwright."""
        async with self._lock:
            if self._browser is not None:
                await self._browser.close()
                self._browser = None
            if self._playwright is not None:
                await self._playwright.stop()
                self._playwright = None


browser_controller = BrowserController(headless=True)
