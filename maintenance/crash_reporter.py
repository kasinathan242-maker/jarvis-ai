"""JARVIS Lite — Secret-Scrubbed Crash Reporter.

Captures system errors, stack traces, and environment contexts, automatically stripping
sensitive tokens, passwords, API keys, and private path credentials.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path
from typing import Any

from core.logging import get_logger

logger = get_logger("maintenance.crash_reporter")

CRASH_REPORTS_DIR = Path("data/crash_reports")

# Patterns matching sensitive credentials and secrets
SECRET_PATTERNS = [
    (re.compile(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{15,}", re.IGNORECASE), r"\1[REDACTED_TOKEN]"),
    (re.compile(r"(api[_-]?key\s*[:=]\s*['\"])[^'\"]+(['\"])", re.IGNORECASE), r"\1[REDACTED_API_KEY]\2"),
    (re.compile(r"(password\s*[:=]\s*['\"])[^'\"]+(['\"])", re.IGNORECASE), r"\1[REDACTED_PASSWORD]\2"),
    (re.compile(r"(secret\s*[:=]\s*['\"])[^'\"]+(['\"])", re.IGNORECASE), r"\1[REDACTED_SECRET]\2"),
    (re.compile(r"(ghp_[a-zA-Z0-9]{36})", re.IGNORECASE), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"(AIza[0-9A-Za-z-_]{35})", re.IGNORECASE), "[REDACTED_GOOGLE_API_KEY]"),
    (re.compile(r"([A-Za-z]:\\Users\\)[^\\]+(\\)", re.IGNORECASE), r"\1<USER>\2"),
]


class CrashReporter:
    """Sanitizes and records system crash diagnostics."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = (output_dir or CRASH_REPORTS_DIR).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scrub_secrets(self, text: str) -> str:
        """Strip all known secrets, keys, and credentials from a string."""
        sanitized = text
        for pattern, replacement in SECRET_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized

    def generate_report(
        self,
        exc: Exception,
        context: dict[str, Any] | None = None,
    ) -> Path:
        """Generate and persist a secret-stripped crash report file.

        Returns:
            Path to the saved crash report JSON.
        """
        raw_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        sanitized_tb = self.scrub_secrets(raw_traceback)
        sanitized_msg = self.scrub_secrets(str(exc))

        sanitized_context: dict[str, Any] = {}
        if context:
            for k, v in context.items():
                if isinstance(v, str):
                    sanitized_context[k] = self.scrub_secrets(v)
                else:
                    sanitized_context[k] = v

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_data = {
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "exception_type": type(exc).__name__,
            "message": sanitized_msg,
            "traceback": sanitized_tb,
            "context": sanitized_context,
            "secrets_stripped": True,
        }

        report_file = self.output_dir / f"crash_{timestamp}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        logger.info(
            "crash_report_generated",
            file=str(report_file.name),
            exception=type(exc).__name__,
        )
        return report_file


crash_reporter = CrashReporter()
