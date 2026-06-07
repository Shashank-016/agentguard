"""Secret and PII redaction for event payloads.

AgentGuard observes raw LLM messages, tool arguments, and responses — which
routinely contain API keys, access tokens, private keys, JWTs, or email
addresses that shouldn't be persisted verbatim into the audit trail or shown
on the dashboard. :func:`redact` walks a payload and replaces recognizable
secret/PII substrings with a ``«REDACTED:<kind>»`` placeholder, preserving
enough structure to remain useful for debugging without leaking the secret
itself.

Toggle
------
Redaction is **on by default**. Disable it via the ``AGENTGUARD_REDACT``
environment variable (``"0"``/``"false"``/``"no"``/``"off"``) or
programmatically via :func:`set_redaction_enabled` — e.g. for tests that need
to assert on raw payload contents.
"""

from __future__ import annotations

import os
import re
from typing import Any

# Each entry is (kind label, compiled pattern). Order matters where patterns
# could overlap — more specific patterns are listed first.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private_key",
        re.compile(
            r"-----BEGIN(?: [A-Z0-9 ]+)? PRIVATE KEY-----"
            r"[\s\S]*?-----END(?: [A-Z0-9 ]+)? PRIVATE KEY-----"
        ),
    ),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b")),
    ("github_token", re.compile(r"\bgh[opsu]_[A-Za-z0-9]{30,}\b")),
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]

_ENV_VAR = "AGENTGUARD_REDACT"
_FALSY = frozenset({"0", "false", "no", "off"})

# Programmatic override — set via set_redaction_enabled(). Takes precedence
# over AGENTGUARD_REDACT when not None. Primarily for tests and embedders that
# need to force a setting regardless of the process environment.
_override: bool | None = None


def set_redaction_enabled(enabled: bool | None) -> None:
    """Force redaction on/off process-wide, overriding ``AGENTGUARD_REDACT``.

    Pass ``None`` to clear the override and fall back to the environment
    variable (the default behavior).
    """
    global _override
    _override = enabled


def is_redaction_enabled() -> bool:
    """Return whether redaction is currently active.

    Resolution order: explicit :func:`set_redaction_enabled` override, then
    the ``AGENTGUARD_REDACT`` environment variable, then the default (on).
    """
    if _override is not None:
        return _override
    raw = os.getenv(_ENV_VAR)
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSY


def _placeholder(kind: str) -> str:
    return f"«REDACTED:{kind}»"


def redact_text(text: str) -> str:
    """Replace recognizable secret/PII substrings in ``text`` with placeholders."""
    for kind, pattern in _PATTERNS:
        text = pattern.sub(_placeholder(kind), text)
    return text


def redact(value: Any) -> Any:
    """Recursively redact secrets/PII from strings nested in ``value``.

    Dicts and lists are walked structurally (keys are left intact — only
    string values are scanned); other types pass through unchanged. Returns
    ``value`` unmodified when redaction is disabled.
    """
    if not is_redaction_enabled():
        return value
    return _redact(value)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact(v) for v in value)
    return value
