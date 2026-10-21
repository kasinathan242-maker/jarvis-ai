"""JARVIS Lite — Resilient LM Studio (OpenAI-Compatible) Client.

Provides secondary local LLM support when LM Studio is running on localhost:1234.
Includes timeout, retry backoff, and circuit breaking.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from ai.circuit_breaker import CircuitBreaker
from config.settings import settings
from core.errors import ModelUnavailableError
from core.logging import get_logger

logger = get_logger("ai.lmstudio_client")


class LMStudioClient:
    def __init__(
        self,
        base_url: str | None = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        cfg = settings.models.lmstudio
        self.base_url = (base_url or cfg.base_url).rstrip("/")
        self.connect_timeout = connect_timeout or cfg.connect_timeout
        self.read_timeout = read_timeout or cfg.read_timeout
        self.max_retries = max_retries or cfg.max_retries

        self.circuit_breaker = CircuitBreaker(
            name="lmstudio",
            failure_threshold=cfg.circuit_breaker_failure_threshold,
            recovery_time_seconds=cfg.circuit_breaker_recovery_time,
        )

        self._limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        self._timeout = httpx.Timeout(
            timeout=self.read_timeout,
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=10.0,
            pool=5.0,
        )

    async def is_healthy(self) -> bool:
        """Check if LM Studio OpenAI-compatible endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/models")
                return resp.status_code == 200
        except Exception:
            return False

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
    ) -> str:
        """Non-streaming chat completion via OpenAI-compatible API."""
        self.circuit_breaker.can_execute()

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout, limits=self._limits) as client:
                    resp = await client.post(f"{self.base_url}/chat/completions", json=payload)
                    if resp.status_code == 200:
                        self.circuit_breaker.record_success()
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            return str(content) if content is not None else ""
                        return ""
                    else:
                        raise ModelUnavailableError(
                            f"LM Studio returned HTTP {resp.status_code}: {resp.text}"
                        )
            except (httpx.ConnectError, httpx.TimeoutException, ModelUnavailableError) as exc:
                last_error = exc
                logger.warn(
                    "lmstudio_chat_attempt_failed",
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error=str(exc),
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

        self.circuit_breaker.record_failure(last_error or Exception("Unknown LM Studio failure"))
        raise ModelUnavailableError(
            f"LM Studio server is unavailable at {self.base_url}. Details: {last_error}"
        )

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens via SSE format from OpenAI-compatible endpoint."""
        self.circuit_breaker.can_execute()

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout, limits=self._limits) as client:
                async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as resp:
                    if resp.status_code != 200:
                        raise ModelUnavailableError(f"LM Studio stream HTTP error: {resp.status_code}")

                    self.circuit_breaker.record_success()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        raw_data = line[len("data:"):].strip()
                        if raw_data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(raw_data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                yield delta
                        except json.JSONDecodeError:
                            continue
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            self.circuit_breaker.record_failure(exc)
            raise ModelUnavailableError(f"LM Studio connection dropped during streaming: {exc}") from exc
