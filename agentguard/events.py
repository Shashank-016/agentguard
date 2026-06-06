"""Core SecurityEvent model — the lingua franca of AgentGuard."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

EventType = Literal[
    "llm_call",
    "tool_call",
    "node_traversal",
    "injection_detected",
    "policy_violation",
    "trust_flag",
    "session_start",
    "session_end",
]

Severity = Literal["info", "warning", "critical"]
Source = Literal["sdk", "langgraph"]


class SecurityEvent(BaseModel):
    """A single auditable event emitted by an instrumented agent.

    Events are the atomic unit of AgentGuard's observability model. Every LLM
    call, tool invocation, injection detection, and policy check produces one
    event. Events may be chained causally via ``parent_event_id``.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: Source
    event_type: EventType
    severity: Severity = "info"
    # Raw request/response data, truncated to 4000 chars for storage efficiency.
    payload: dict[str, Any] = {}
    # Triggered rule identifiers, e.g. "injection:jailbreak_attempt".
    flags: list[str] = []
    # Links this event to the upstream event that caused it.
    parent_event_id: Optional[str] = None
    metadata: dict[str, Any] = {}

    model_config = {
        "json_schema_extra": {},
    }


def _truncate(value: Any, max_chars: int = 4000) -> Any:
    """Truncate string values in a payload dict to ``max_chars`` characters."""
    if isinstance(value, str):
        return value[:max_chars] + ("…" if len(value) > max_chars else "")
    if isinstance(value, dict):
        return {k: _truncate(v, max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v, max_chars) for v in value]
    return value


def make_payload(**kwargs: Any) -> dict[str, Any]:
    """Build a storage-safe payload dict with truncated string values."""
    return _truncate(kwargs)  # type: ignore[return-value]
