"""AgentGuardCallback — LangGraph/LangChain callback handler for security observability."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from .audit import AuditLogger
from .bus import EventBus
from .engine.injection import InjectionDetector
from .engine.policy import ToolPolicyEngine
from .engine.trust import TrustScorer
from .events import SecurityEvent, make_payload

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import BaseCallbackHandler

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    # Provide a stub so the module is importable without langchain installed.
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass

    _LANGCHAIN_AVAILABLE = False


class AgentGuardCallback(BaseCallbackHandler):
    """LangGraph/LangChain callback handler that emits SecurityEvents.

    Attach to any LangChain or LangGraph runnable to get automatic
    observability over node traversals, LLM calls, and tool invocations.

    Parameters
    ----------
    session_id:
        Identifier grouping all events in this run. Auto-generated if omitted.
    agent_id:
        Logical name of the agent (used in policy lookups and event records).
    bus:
        Shared :class:`~agentguard.bus.EventBus`. Created fresh if not provided.
    policy_path:
        Path to a YAML tool policy file.
    trust_scorer:
        Shared :class:`~agentguard.engine.trust.TrustScorer`.
    mode:
        ``"observe"`` (default) or ``"enforce"``.
    audit_log:
        Path to a JSONL audit file. Every event is appended durably and
        synchronously — survives process restarts. Pass ``None`` (default) to
        keep events in-memory only.

    Example
    -------
    ::

        from agentguard import AgentGuardCallback

        callback = AgentGuardCallback(
            session_id="run-001",
            agent_id="researcher",
            audit_log="logs/audit.jsonl",
        )
        graph.invoke({"input": doc}, config={"callbacks": [callback]})
    """

    def __init__(
        self,
        session_id: str | None = None,
        agent_id: str = "langgraph",
        bus: EventBus | None = None,
        policy_path: str | None = None,
        trust_scorer: TrustScorer | None = None,
        mode: str = "observe",
        audit_log: str | None = None,
    ) -> None:
        super().__init__()
        self.session_id = session_id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.mode = mode

        self._bus = bus if bus is not None else EventBus()
        self._injection_detector = InjectionDetector()
        self._policy_engine = ToolPolicyEngine(policy_path=policy_path)
        self._trust_scorer = trust_scorer if trust_scorer is not None else TrustScorer()

        # Wire up durable audit logging if requested.
        self._audit_logger: AuditLogger | None = None
        if audit_log is not None:
            self._audit_logger = AuditLogger(path=audit_log)
            self._bus.subscribe(self._audit_logger)
            logger.info("[AgentGuard] Audit log -> %s", audit_log)

        # Tracks (chain_run_id -> event_id) for causal linking.
        self._active_chains: dict[str, str] = {}
        # Current agent_id may change on node traversal.
        self._current_node: str | None = None

        self._bus.emit(
            SecurityEvent(
                session_id=self.session_id,
                agent_id=self.agent_id,
                source="langgraph",
                event_type="session_start",
                severity="info",
                payload=make_payload(mode=mode),
            )
        )
        logger.info(
            "[AgentGuard] Session %s started (agent=%s, mode=%s)",
            self.session_id,
            agent_id,
            mode,
        )

    # ------------------------------------------------------------------
    # Chain / node events
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a LangGraph node or chain begins execution."""
        node_name = _extract_name(serialized) or self.agent_id
        self._current_node = node_name
        event_id = str(uuid.uuid4())
        self._active_chains[str(run_id)] = event_id

        self._bus.emit(
            SecurityEvent(
                event_id=event_id,
                session_id=self.session_id,
                agent_id=node_name,
                source="langgraph",
                event_type="node_traversal",
                severity="info",
                payload=make_payload(
                    node=node_name,
                    input_keys=list(inputs.keys()) if isinstance(inputs, dict) else [],
                ),
            )
        )
        logger.info("[AgentGuard] node_traversal: %s -> INFO", node_name)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a LangGraph node or chain finishes."""
        self._active_chains.pop(str(run_id), None)

    # ------------------------------------------------------------------
    # LLM / chat model events
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Called before a chat model is invoked — runs injection detection."""
        node_name = self._current_node or self.agent_id
        llm_event_id = str(uuid.uuid4())
        self._active_chains[f"llm:{run_id}"] = llm_event_id

        # Flatten all message content for scanning.
        flat_messages = []
        for batch in messages:
            for msg in batch:
                content = getattr(msg, "content", "") or ""
                flat_messages.append({"role": getattr(msg, "type", "unknown"), "content": content})

        injection_matches = self._injection_detector.scan_messages(flat_messages)

        self._bus.emit(
            SecurityEvent(
                event_id=llm_event_id,
                session_id=self.session_id,
                agent_id=node_name,
                source="langgraph",
                event_type="llm_call",
                severity="info",
                payload=make_payload(message_count=len(flat_messages)),
            )
        )
        logger.info("[AgentGuard] llm_call: %s -> INFO", node_name)

        if injection_matches:
            flags = [m.flag for m in injection_matches]
            severity = (
                "critical"
                if any(m.severity == "critical" for m in injection_matches)
                else "warning"
            )
            self._trust_scorer.record_injection_flag(self.session_id)
            trust_score = self._trust_scorer.score(self.session_id)

            self._bus.emit(
                SecurityEvent(
                    session_id=self.session_id,
                    agent_id=node_name,
                    source="langgraph",
                    event_type="injection_detected",
                    severity=severity,
                    payload=make_payload(
                        matches=[
                            {
                                "pattern": m.pattern_name,
                                "category": m.category,
                                "matched_text": m.matched_text,
                            }
                            for m in injection_matches
                        ]
                    ),
                    flags=flags,
                    parent_event_id=llm_event_id,
                    metadata={"trust_score": trust_score},
                )
            )
            logger.warning(
                "[AgentGuard] injection_detected: %s -> %s\n  flags: %s\n  trust_score: %.2f "
                "(degraded — external content processed)",
                node_name,
                severity.upper(),
                flags,
                trust_score,
            )

    # ------------------------------------------------------------------
    # Tool events
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Called before a tool is executed — checks tool policy and trust."""
        tool_name = _extract_name(serialized) or "unknown"
        node_name = self._current_node or self.agent_id
        parent_id = self._active_chains.get(str(run_id))

        # --- Trust check ---------------------------------------------------
        if self._trust_scorer.should_flag(self.session_id, tool_name):
            trust_score = self._trust_scorer.score(self.session_id)
            self._bus.emit(
                SecurityEvent(
                    session_id=self.session_id,
                    agent_id=node_name,
                    source="langgraph",
                    event_type="trust_flag",
                    severity="warning",
                    payload=make_payload(
                        tool_name=tool_name,
                        trust_score=trust_score,
                        reason=(
                            f"Agent received output from low-trust session "
                            f"(score={trust_score:.2f}), attempting {tool_name}"
                        ),
                    ),
                    flags=["trust:low_score_sensitive_tool"],
                    parent_event_id=parent_id,
                    metadata={"trust_score": trust_score},
                )
            )
            logger.warning(
                "[AgentGuard] trust_flag: %s -> WARNING\n  reason: low-trust session "
                "(score=%.2f), attempting %s",
                node_name,
                trust_score,
                tool_name,
            )

        # --- Policy check --------------------------------------------------
        policy_result = self._policy_engine.check(node_name, tool_name)
        if not policy_result.allowed:
            self._bus.emit(
                SecurityEvent(
                    session_id=self.session_id,
                    agent_id=node_name,
                    source="langgraph",
                    event_type="policy_violation",
                    severity="critical",
                    payload=make_payload(
                        tool_name=tool_name,
                        input=input_str,
                        reason=policy_result.reason,
                    ),
                    flags=[policy_result.rule_name],
                    parent_event_id=parent_id,
                )
            )
            logger.error(
                "[AgentGuard] policy_violation: %s.%s -> CRITICAL\n  reason: %s",
                node_name,
                tool_name,
                policy_result.reason,
            )
        else:
            self._bus.emit(
                SecurityEvent(
                    session_id=self.session_id,
                    agent_id=node_name,
                    source="langgraph",
                    event_type="tool_call",
                    severity="info",
                    payload=make_payload(tool_name=tool_name, input=input_str),
                    parent_event_id=parent_id,
                )
            )
            logger.info("[AgentGuard] tool_call: %s.%s -> INFO", node_name, tool_name)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Called after a tool completes — scans the output for injection."""
        output_str = str(output) if output else ""
        if not output_str:
            return

        matches = self._injection_detector.scan(output_str)
        if matches:
            flags = [m.flag for m in matches]
            node_name = self._current_node or self.agent_id
            self._trust_scorer.record_external_content(self.session_id, source_type="tool_output")

            self._bus.emit(
                SecurityEvent(
                    session_id=self.session_id,
                    agent_id=node_name,
                    source="langgraph",
                    event_type="injection_detected",
                    severity="warning",
                    payload=make_payload(
                        context="tool_output",
                        output_preview=output_str[:500],
                        match_count=len(matches),
                    ),
                    flags=flags,
                )
            )
            logger.warning(
                "[AgentGuard] injection_detected in tool output: %s -> WARNING  flags=%s",
                node_name,
                flags,
            )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def record_external_content(self, source_type: str = "file") -> None:
        """Signal that external content was processed — degrades trust score."""
        self._trust_scorer.record_external_content(self.session_id, source_type)

    def record_agent_handoff(self, from_agent: str, to_agent: str) -> None:
        """Signal a data handoff between agents in a multi-agent graph."""
        self._trust_scorer.record_agent_handoff(self.session_id, from_agent, to_agent)
        logger.info(
            "[AgentGuard] agent_handoff: %s -> %s (trust=%.2f)",
            from_agent,
            to_agent,
            self._trust_scorer.score(self.session_id),
        )

    def trust_summary(self) -> dict:
        """Return a dict summary of the current trust state."""
        return self._trust_scorer.summary(self.session_id)

    def session_report(self) -> dict:
        """Return a structured report of all events in this session."""
        events = self._bus.get_session_events(self.session_id)
        critical = [e for e in events if e.severity == "critical"]
        warnings = [e for e in events if e.severity == "warning"]
        return {
            "session_id": self.session_id,
            "total_events": len(events),
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "trust_score": self._trust_scorer.score(self.session_id),
            "trust_label": self._trust_scorer.trust_label(self.session_id),
            "flagged": self._trust_scorer.is_flagged(self.session_id),
            "flags_seen": sorted({f for e in events for f in e.flags}),
        }


def _extract_name(serialized: dict[str, Any]) -> str | None:
    """Extract a human-readable name from a LangChain serialized component dict."""
    if not isinstance(serialized, dict):
        return None
    return serialized.get("name") or serialized.get("id", [None])[-1] or None
