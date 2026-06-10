"""Trust scoring -- tracks information provenance across agent hops.

The core insight: when an agent processes external content and passes the
result downstream, the downstream agent inherits the trust level of that
content. An agent acting on low-trust data and calling a sensitive tool is a
trust violation -- even if the agent itself is fully trusted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ._state import DEFAULT_MAX_ENTRIES, DEFAULT_TTL_SECONDS, BoundedStateStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trust levels
# ---------------------------------------------------------------------------

TRUSTED = 1.0  # Instructions from the human user or system prompt.
INTERNAL = 0.7  # Output from another verified, trusted agent.
EXTERNAL = 0.3  # Content from the web, files, or user-uploaded documents.
UNTRUSTED = 0.0  # Flagged or unknown provenance.

_SENSITIVE_TOOL_KEYWORDS = frozenset(
    {"write", "exec", "delete", "send", "shell", "post", "rm", "overwrite"}
)


@dataclass
class TrustEvent:
    """A record of a provenance-changing event in a session."""

    event_type: str  # "external_content" | "agent_handoff" | "injection_flag"
    description: str
    trust_delta: float  # How much trust changed (negative = degradation).
    resulting_score: float


@dataclass
class SessionTrustState:
    """Mutable trust state for a single session."""

    score: float = TRUSTED
    history: list[TrustEvent] = field(default_factory=list)
    flagged: bool = False


class TrustScorer:
    """Tracks and scores trust degradation across agent sessions.

    Trust degrades multiplicatively when external content is processed --
    compounding degradations reflect compounding uncertainty about data
    provenance. A researcher reading a web page (score → 0.3) that passes
    output to a writer (score → 0.3 × 0.7 ≈ 0.21) models the real risk
    correctly: the further from the original source, the lower the trust.

    Parameters
    ----------
    sensitive_threshold:
        Trust score below which a sensitive tool call triggers a flag.
    max_sessions:
        Maximum number of sessions to retain. The least-recently-touched
        session is evicted once this limit is exceeded — bounds memory for
        long-running proxies that see many distinct sessions.
    ttl_seconds:
        Sessions untouched for longer than this are evicted lazily.
    """

    def __init__(
        self,
        sensitive_threshold: float = 0.5,
        max_sessions: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._threshold = sensitive_threshold
        self._sessions: BoundedStateStore[str, SessionTrustState] = BoundedStateStore(
            max_entries=max_sessions, ttl_seconds=ttl_seconds
        )

    def _get_or_create(self, session_id: str) -> SessionTrustState:
        return self._sessions.get_or_create(session_id, SessionTrustState)

    # ------------------------------------------------------------------
    # Trust-modifying events
    # ------------------------------------------------------------------

    def record_external_content(self, session_id: str, source_type: str = "unknown") -> None:
        """Signal that this session has processed external, untrusted content.

        Examples: reading a file from disk, fetching a web page, loading a
        user-uploaded document. Each call multiplicatively degrades trust.
        """
        state = self._get_or_create(session_id)
        previous = state.score
        state.score = round(state.score * EXTERNAL, 4)
        event = TrustEvent(
            event_type="external_content",
            description=f"Processed external content from source_type='{source_type}'",
            trust_delta=round(state.score - previous, 4),
            resulting_score=state.score,
        )
        state.history.append(event)
        logger.debug(
            "[TrustScorer] session=%s external_content source=%s score: %.2f → %.2f",
            session_id,
            source_type,
            previous,
            state.score,
        )

    def record_agent_handoff(self, session_id: str, from_agent: str, to_agent: str) -> None:
        """Signal that session data is being passed between agents.

        Cross-agent handoffs apply an INTERNAL-level multiplier to reflect
        that the receiving agent inherits whatever trust state the sending
        agent accumulated.
        """
        state = self._get_or_create(session_id)
        previous = state.score
        # Handoff only degrades trust if the session is already below INTERNAL.
        if state.score < INTERNAL:
            state.score = round(state.score * INTERNAL, 4)
        event = TrustEvent(
            event_type="agent_handoff",
            description=f"Data passed from '{from_agent}' to '{to_agent}'",
            trust_delta=round(state.score - previous, 4),
            resulting_score=state.score,
        )
        state.history.append(event)
        logger.debug(
            "[TrustScorer] session=%s handoff %s→%s score: %.2f → %.2f",
            session_id,
            from_agent,
            to_agent,
            previous,
            state.score,
        )

    def record_injection_flag(self, session_id: str) -> None:
        """Immediately set session trust to UNTRUSTED when injection is detected."""
        state = self._get_or_create(session_id)
        previous = state.score
        state.score = UNTRUSTED
        state.flagged = True
        state.history.append(
            TrustEvent(
                event_type="injection_flag",
                description="Injection detected -- session marked UNTRUSTED",
                trust_delta=round(UNTRUSTED - previous, 4),
                resulting_score=UNTRUSTED,
            )
        )
        logger.warning(
            "[TrustScorer] session=%s INJECTION FLAGGED -- trust zeroed (was %.2f)",
            session_id,
            previous,
        )

    # ------------------------------------------------------------------
    # Scoring and flagging
    # ------------------------------------------------------------------

    def score(self, session_id: str) -> float:
        """Return the current trust score for a session (1.0 = fully trusted)."""
        return self._get_or_create(session_id).score

    def is_flagged(self, session_id: str) -> bool:
        """Return True if the session was explicitly flagged for injection."""
        return self._get_or_create(session_id).flagged

    def should_flag(self, session_id: str, tool_name: str) -> bool:
        """Return True if the trust score is too low for the requested tool.

        Flags when:
        - Trust is below ``sensitive_threshold`` AND
        - The tool name contains a sensitive keyword (write, execute, send…)
        """
        score = self.score(session_id)
        if score >= self._threshold:
            return False
        tool_lower = tool_name.lower()
        return any(kw in tool_lower for kw in _SENSITIVE_TOOL_KEYWORDS)

    def trust_label(self, session_id: str) -> str:
        """Return a human-readable trust label for display purposes."""
        score = self.score(session_id)
        if score >= 0.9:
            return "TRUSTED"
        if score >= 0.6:
            return "INTERNAL"
        if score >= 0.2:
            return "EXTERNAL"
        return "UNTRUSTED"

    def history(self, session_id: str) -> list[TrustEvent]:
        """Return the list of trust-modifying events for a session."""
        return self._get_or_create(session_id).history

    def summary(self, session_id: str) -> dict:
        """Return a dict summary of the session's trust state."""
        state = self._get_or_create(session_id)
        return {
            "session_id": session_id,
            "score": state.score,
            "label": self.trust_label(session_id),
            "flagged": state.flagged,
            "event_count": len(state.history),
            "history": [
                {
                    "event_type": e.event_type,
                    "description": e.description,
                    "trust_delta": e.trust_delta,
                    "resulting_score": e.resulting_score,
                }
                for e in state.history
            ],
        }
