"""JARVIS Lite — Resilient Ollama Client.

Features:
- Streaming token generator and non-streaming completions.
- Strict connection timeout (5s) and read timeout (120s).
- 3 bounded retries with exponential backoff.
- Integrated Circuit Breaker pattern to prevent freezing.
- Embeddings generation for ChromaDB integration.
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

logger = get_logger("ai.ollama_client")


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        cfg = settings.models.ollama
        self.base_url = (base_url or cfg.base_url).rstrip("/")
        self.connect_timeout = connect_timeout or cfg.connect_timeout
        self.read_timeout = read_timeout or cfg.read_timeout
        self.max_retries = max_retries or cfg.max_retries

        self.circuit_breaker = CircuitBreaker(
            name="ollama",
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
        """Probe Ollama tags endpoint to determine service health."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """Return list of models currently installed in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    return [m.get("name", "") for m in data.get("models", [])]
                return []
        except Exception as exc:
            logger.warn("ollama_list_models_failed", error=str(exc))
            return []

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        format_json: bool = False,
    ) -> str:
        """Execute a non-streaming chat completion with automatic retries and circuit breaking."""
        self.circuit_breaker.can_execute()
        target_model = model or settings.models.routing.get("chat", "qwen2.5:7b-instruct-q4_K_M")

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if format_json:
            payload["format"] = "json"

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout, limits=self._limits) as client:
                    resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                    if resp.status_code == 200:
                        self.circuit_breaker.record_success()
                        data = resp.json()
                        content = data.get("message", {}).get("content", "")
                        return str(content) if content is not None else ""
                    else:
                        raise ModelUnavailableError(
                            f"Ollama returned HTTP {resp.status_code}: {resp.text}"
                        )
            except (httpx.ConnectError, httpx.TimeoutException, ModelUnavailableError) as exc:
                last_error = exc
                logger.warn(
                    "ollama_chat_attempt_failed",
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error=str(exc),
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))  # Exponential backoff: 0.5s, 1.0s, 2.0s

        # If all retries exhausted:
        self.circuit_breaker.record_failure(last_error or Exception("Unknown Ollama failure"))
        raise ModelUnavailableError(
            f"Ollama server is unavailable at {self.base_url} after {self.max_retries} attempts. Details: {last_error}"
        )

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream chat tokens in real-time. Yields chunks as they arrive from Ollama."""
        self.circuit_breaker.can_execute()
        target_model = model or settings.models.routing.get("chat", "qwen2.5:7b-instruct-q4_K_M")

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout, limits=self._limits) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        raise ModelUnavailableError(f"Ollama stream failed with HTTP {resp.status_code}")

                    self.circuit_breaker.record_success()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            self.circuit_breaker.record_failure(exc)
            raise ModelUnavailableError(
                f"Ollama connection dropped during token streaming: {exc}"
            ) from exc

    async def generate_embedding(self, text: str, model: str | None = None) -> list[float]:
        """Generate text embedding vector for ChromaDB storage/retrieval."""
        self.circuit_breaker.can_execute()
        target_model = model or settings.models.routing.get("embed", "nomic-embed-text")

        payload = {
            "model": target_model,
            "prompt": text,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout, limits=self._limits) as client:
                resp = await client.post(f"{self.base_url}/api/embeddings", json=payload)
                if resp.status_code == 200:
                    self.circuit_breaker.record_success()
                    data = resp.json()
                    raw_emb = data.get("embedding", [])
                    return [float(x) for x in raw_emb]
                raise ModelUnavailableError(f"Failed to generate embedding: HTTP {resp.status_code}")
        except Exception as exc:
            self.circuit_breaker.record_failure(exc)
            raise ModelUnavailableError(f"Ollama embedding generation error: {exc}") from exc


# Global singleton client
ollama_client = OllamaClient()
