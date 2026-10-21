"""JARVIS Lite — Model Router.

Routes tasks (chat, route, plan, code, summarize, embed) to appropriate models
and healthy backends (Ollama primary, LM Studio fallback).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from ai.lmstudio_client import LMStudioClient
from ai.ollama_client import OllamaClient
from config.settings import settings
from core.errors import ModelUnavailableError
from core.logging import get_logger

logger = get_logger("ai.model_router")


class ModelRouter:
    def __init__(self) -> None:
        self.ollama = OllamaClient()
        self.lmstudio = LMStudioClient()

    def get_model_for_task(self, task_type: str) -> str:
        """Look up the configured model identifier for a given task type."""
        routing = settings.models.routing
        return routing.get(task_type, routing.get("chat", "qwen2.5:7b-instruct-q4_K_M"))

    async def get_active_backend(self) -> str:
        """Determine which backend is currently online and healthy."""
        if await self.ollama.is_healthy():
            return "ollama"
        if await self.lmstudio.is_healthy():
            return "lmstudio"
        return "none"

    async def chat(
        self,
        messages: list[dict[str, str]],
        task_type: str = "chat",
        temperature: float = 0.7,
        format_json: bool = False,
    ) -> str:
        """Route non-streaming chat request with graceful fallback."""
        model = self.get_model_for_task(task_type)

        primary_error: Exception | None = None
        # 1. Try Ollama (Primary)
        try:
            return await self.ollama.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                format_json=format_json,
            )
        except ModelUnavailableError as ollama_err:
            primary_error = ollama_err
            logger.warn("primary_backend_failed_trying_fallback", error=str(ollama_err))

        # 2. Try LM Studio (Secondary fallback)
        try:
            return await self.lmstudio.chat(
                messages=messages,
                model="default",
                temperature=temperature,
            )
        except ModelUnavailableError as lm_err:
            logger.error(
                "all_model_backends_unavailable",
                ollama_error=str(primary_error),
                lmstudio_error=str(lm_err),
            )
            raise ModelUnavailableError(
                "All local AI model servers (Ollama, LM Studio) are currently unavailable. "
                "Please verify that 'ollama serve' is running."
            ) from lm_err

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        task_type: str = "chat",
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Route streaming tokens with graceful fallback."""
        model = self.get_model_for_task(task_type)

        try:
            async for token in self.ollama.stream_chat(messages=messages, model=model, temperature=temperature):
                yield token
            return
        except ModelUnavailableError as ollama_err:
            logger.warn("ollama_stream_failed_trying_fallback", error=str(ollama_err))

        try:
            async for token in self.lmstudio.stream_chat(messages=messages, model="default", temperature=temperature):
                yield token
            return
        except ModelUnavailableError as lm_err:
            raise ModelUnavailableError(
                "Model server is currently unavailable. Tokens cannot be generated."
            ) from lm_err


router = ModelRouter()
