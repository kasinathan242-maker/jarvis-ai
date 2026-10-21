"""JARVIS Lite — Main Application Entry Point.

Usage:
    python main.py          # Launches FastAPI backend on 127.0.0.1:8001
    python main.py --voice  # Launches interactive push-to-talk voice mode
"""

from __future__ import annotations

import asyncio
import sys

import uvicorn

from bootstrap import run_bootstrap
from config.settings import settings
from core.lifecycle import lifecycle
from core.logging import get_logger, setup_logging
from security.secrets import get_or_create_api_token
from voice.pipeline import pipeline

logger = get_logger("main")


async def run_voice_cli() -> None:
    """Interactive CLI mode demonstrating push-to-talk voice/command processing."""
    setup_logging(log_level=settings.system.log_level)
    print("\n[JARVIS LITE] Starting Interactive Voice & Command Shell...")
    print("Type a command (e.g. 'what time is it', 'open notepad', 'system stats', 'exit') or press Enter for sample:")

    await lifecycle.startup()

    try:
        while True:
            try:
                user_input = await asyncio.to_thread(input, "\n[YOU] > ")
                user_text = user_input.strip()
                if not user_text:
                    continue

                if user_text.lower() in ("exit", "quit", "q"):
                    print("[JARVIS] Shutting down.")
                    break

                response = await pipeline.process_command(user_text, play_speech=True)
                print(f"[JARVIS] > {response}")

            except (KeyboardInterrupt, EOFError):
                break
    finally:
        await lifecycle.shutdown()


def run_server() -> None:
    """Launch production Uvicorn ASGI server bound strictly to 127.0.0.1."""
    setup_logging(log_level=settings.system.log_level)

    # Perform pre-flight bootstrap validation
    passed = run_bootstrap()
    if not passed:
        print("\n[ABORT] Critical pre-flight bootstrap checks failed. Correct errors before running.")
        sys.exit(1)

    token = get_or_create_api_token()
    print(f"\n[SECURITY] Local API Bearer Token: {token}")
    print(f"[JARVIS LITE] Binding server to http://{settings.server.host}:{settings.server.port}...")

    uvicorn.run(
        "api.server:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
        log_config=None,  # Handled by structlog
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--voice":
        asyncio.run(run_voice_cli())
    else:
        run_server()
