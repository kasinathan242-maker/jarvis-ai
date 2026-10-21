"""JARVIS Lite — Permission and Security Enforcement Layer.

Implements risk levels (SAFE, CONFIRM, DANGEROUS, FORBIDDEN),
shell command allowlists and blocklists, path sandboxing, and
approval decision hooks.
"""

from __future__ import annotations

import enum
import re
from pathlib import Path

from config.settings import settings
from core.errors import SecurityViolationError


class RiskLevel(enum.StrEnum):
    SAFE = "SAFE"
    CONFIRM = "CONFIRM"
    DANGEROUS = "DANGEROUS"
    FORBIDDEN = "FORBIDDEN"


FORBIDDEN_PATTERNS = [
    re.compile(r"\bformat\b", re.IGNORECASE),
    re.compile(r"\bdiskpart\b", re.IGNORECASE),
    re.compile(r"\breg\s+delete\b", re.IGNORECASE),
    re.compile(r"\bbcdedit\b", re.IGNORECASE),
    re.compile(r"\bdel\s+/[sS]\b", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"[cC]:\\windows", re.IGNORECASE),
    re.compile(r"\bRemove-Item\b.*-Recurse", re.IGNORECASE),
]


def classify_shell_command(cmd_str: str) -> tuple[RiskLevel, str]:
    """Inspect a shell command string against forbidden blocklists and allowlists.

    Returns:
        (RiskLevel, reason_description)
    """
    clean_cmd = cmd_str.strip()

    # 1. Check FORBIDDEN blocklist first — cannot be approved under any circumstances
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(clean_cmd):
            return RiskLevel.FORBIDDEN, f"Command contains forbidden pattern: {pattern.pattern}"

    # Check for hard blocklist entries configured in settings
    for blocked in settings.security.block_shell_commands:
        if blocked.lower() in clean_cmd.lower():
            return RiskLevel.FORBIDDEN, f"Command matches blocked keyword: '{blocked}'"

    # Extract base command executable
    tokens = clean_cmd.split()
    if not tokens:
        return RiskLevel.SAFE, "Empty command"

    base_binary = tokens[0].lower().replace(".exe", "")

    # 2. Check SAFE allowlist
    allowlist = [a.lower() for a in settings.security.allow_shell_commands]
    if base_binary in allowlist:
        return RiskLevel.SAFE, f"Command '{base_binary}' is in the safe allowlist"

    # 3. Everything else defaults to CONFIRM
    return RiskLevel.CONFIRM, f"Command '{base_binary}' is outside the allowlist and requires confirmation"


def validate_path_sandbox(target_path: Path, allow_read_only: bool = True) -> bool:
    """Validate that path operations stay within permissible bounds."""
    try:
        resolved = target_path.resolve()
        # System directories are strictly forbidden for write operations
        sys_root = Path("C:/Windows").resolve()
        if sys_root in resolved.parents or resolved == sys_root:
            raise SecurityViolationError(f"Access to Windows system path '{resolved}' is forbidden.")
        return True
    except SecurityViolationError:
        raise
    except Exception as exc:
        raise SecurityViolationError(f"Failed to validate path sandbox for '{target_path}': {exc}") from exc
