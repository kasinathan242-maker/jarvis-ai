"""JARVIS Lite — Secret Storage via Windows Credential Manager (Keyring).

Stores bearer tokens, credentials, and API secrets securely in the Windows
Credential Manager via Python's keyring library. Never writes raw keys to disk.
"""

from __future__ import annotations

import secrets

import keyring

SERVICE_NAME = "JARVIS_Lite"
BEARER_TOKEN_KEY = "api_bearer_token"

_in_memory_token_cache: str | None = None


def get_secret(key: str) -> str | None:
    """Retrieve a secret from Windows Credential Manager."""
    try:
        secret = keyring.get_password(SERVICE_NAME, key)
        if secret:
            return secret
    except Exception:
        pass
    if key == BEARER_TOKEN_KEY and _in_memory_token_cache:
        return _in_memory_token_cache
    return None


def set_secret(key: str, value: str) -> None:
    """Store a secret in Windows Credential Manager."""
    global _in_memory_token_cache
    try:
        keyring.set_password(SERVICE_NAME, key, value)
    except Exception:
        # Fallback to process-memory cache if Windows keyring is blocked
        if key == BEARER_TOKEN_KEY:
            _in_memory_token_cache = value


def delete_secret(key: str) -> bool:
    """Delete a secret from Windows Credential Manager."""
    global _in_memory_token_cache
    try:
        keyring.delete_password(SERVICE_NAME, key)
        if key == BEARER_TOKEN_KEY:
            _in_memory_token_cache = None
        return True
    except Exception:
        if key == BEARER_TOKEN_KEY:
            _in_memory_token_cache = None
            return True
        return False


def get_or_create_api_token() -> str:
    """Retrieve the API bearer token from keyring, or generate a cryptographically strong token on first run."""
    token = get_secret(BEARER_TOKEN_KEY)
    if not token:
        token = secrets.token_hex(32)
        set_secret(BEARER_TOKEN_KEY, token)
    return token


def verify_api_token(candidate_token: str) -> bool:
    """Constant-time verification of an incoming bearer token."""
    expected_token = get_or_create_api_token()
    if not candidate_token:
        return False
    return secrets.compare_digest(candidate_token.strip(), expected_token.strip())
