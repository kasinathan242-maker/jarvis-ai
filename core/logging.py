"""JARVIS Lite — Structured JSON Logging with Secret Redaction.

Configures structlog to output machine-readable JSON logs to file and
human-readable colored logs to the terminal, redacting API keys,
bearer tokens, passwords, and sensitive credentials automatically.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOGS_DIR / "jarvis.jsonl"

# Regex patterns for secret redaction
SENSITIVE_PATTERNS = [
    (
        re.compile(r"(?i)(api[_-]?key|secret|token|password|auth|bearer)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?"),
        r"\1=[REDACTED]",
    ),
    (re.compile(r"Bearer\s+([a-zA-Z0-9_\-\.]{12,})"), r"Bearer [REDACTED]"),
    (re.compile(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"), r"[REDACTED_EMAIL]"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), r"sk-[REDACTED]"),
]


def redact_sensitive_data(
    logger: logging.Logger, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Structlog processor that recursively masks tokens, credentials, and PII."""

    def _sanitize(value: Any) -> Any:
        if isinstance(value, str):
            sanitized = value
            for pattern, repl in SENSITIVE_PATTERNS:
                sanitized = pattern.sub(repl, sanitized)
            return sanitized
        elif isinstance(value, dict):
            return {k: _sanitize(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_sanitize(v) for v in value]
        return value

    for key, value in list(event_dict.items()):
        event_dict[key] = _sanitize(value)
    return event_dict


def setup_logging(log_level: str = "INFO") -> None:
    """Initialize standard library logging and structlog pipeline."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Handlers: File handler for JSON, Stream handler for console
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [file_handler, console_handler]


    # Shared processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        redact_sensitive_data,
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # JSON formatter for the file handler
    file_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    file_handler.setFormatter(file_formatter)

    # Console formatter
    console_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ],
    )
    console_handler.setFormatter(console_formatter)


def get_logger(name: str = "jarvis") -> structlog.stdlib.BoundLogger:
    """Return a pre-configured bound structlog logger."""
    from typing import cast
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
