"""JARVIS Lite — Core Exception Hierarchy.

Defines typed, explicit domain errors across AI model interactions,
circuit breaking, tool execution, security policies, and lifecycle.
"""

from __future__ import annotations

from typing import Any


class JarvisError(Exception):
    """Base exception for all JARVIS Lite domain errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(JarvisError):
    """Raised when configuration validation fails."""


class ModelUnavailableError(JarvisError):
    """Raised when an AI model server (Ollama, LM Studio) cannot be contacted."""


class CircuitBreakerOpenError(ModelUnavailableError):
    """Raised when the circuit breaker is tripped OPEN to prevent hammering a dead service."""


class ToolExecutionError(JarvisError):
    """Raised when a tool fails during execution or fails verify()."""


class ToolNotFoundError(JarvisError):
    """Raised when an unregistered tool is requested."""


class SecurityViolationError(JarvisError):
    """Raised when an action violates security policies (forbidden command, sandbox escape)."""


class ApprovalRejectedError(JarvisError):
    """Raised when user denies approval for a confirm/dangerous action."""


class InjectionDetectedError(SecurityViolationError):
    """Raised when prompt injection payload is detected in untrusted content."""
