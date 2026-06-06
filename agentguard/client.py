"""GuardedClient — drop-in wrapper around anthropic.Anthropic with security instrumentation."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal, Optional

from .audit import AuditLogger
from .bus import EventBus
from .engine.injection import InjectionDetector
from .engine.policy import ToolPolicyEngine
from .engine.trust import TrustScorer
from .events import SecurityEvent, make_payload

logger = logging.getLogger(__name__)


class AgentGuardException(Exception):
    """Raised in enforce mode when a security violation blocks execution."""


class GuardedMessages:
    """Proxy for ``client.messages`` that intercepts ``.create()`` calls.

    Performs pre-call security checks (injection scan, policy check) and
    post-call event emission (tool call tracking) without altering the
    underlying response object.
    """

    def __init__(self, guard: "GuardedClient") -> None:
        self._guard = guard
        self._underlying = guard._client.messages

    def create(self, **kwargs: Any) -> Any:
        """Intercept a messages.create call to perform security checks."""
        g = self._guard
        messages: list[dict] = kwargs.get("messages", [])
        tools: list[dict] = kwargs.get("tools", [])
        system: str = kwargs.get("system", "")
        model: str = kwargs.get("model", "unknown")

        # Build a scan corpus that includes the system prompt.
        scan_corpus = messages.copy()
        if system:
            scan_corpus = [{"role": "system", "content": system}] + scan_corpus

        llm_event_id = str(uuid.uuid4())

        # --- 1. Injection scan --------------------------------------------
        injection_matches = g._injection_detector.scan_messages(scan_corpus)

        # --- 2. Tool policy check -----------------------------------------
        policy_violations: list[str] = []
        for tool_def in tools:
            tool_name = tool_def.get("name", "unknown")
            result = g._policy_engine.check(g.agent_id, tool_name)
            if not result.allowed:
                policy_violations.append(
                    f"{tool_name}: {result.reason} [{result.rule_name}]"
                )

        # --- 3. Emit llm_call event ----------------------------------------
        g._bus.emit(
            SecurityEvent(
                event_id=llm_event_id,
                session_id=g.session_id,
                agent_id=g.agent_id,
                source="sdk",
                event_type="llm_call",
                severity="info",
                payload=make_payload(
                    model=model,
                    message_count=len(messages),
                    tool_count=len(tools),
                    has_system_prompt=bool(system),
                ),
                metadata={"tool_names": [t.get("name") for t in tools]},
            )
        )
        logger.info(
            "[AgentGuard] llm_call: %s -> INFO (%d tool definition(s) checked)",
            g.agent_id,
            len(tools),
        )

        # --- 4. Injection events -------------------------------------------
        if injection_matches:
            flags = [m.flag for m in injection_matches]
            max_severity = "critical" if any(
                m.severity == "critical" for m in injection_matches
            ) else "warning"

            # Update trust state.
            g._trust_scorer.record_injection_flag(g.session_id)
            trust_score = g._trust_scorer.score(g.session_id)

            g._bus.emit(
                SecurityEvent(
                    session_id=g.session_id,
                    agent_id=g.agent_id,
                    source="sdk",
                    event_type="injection_detected",
                    severity=max_severity,
                    payload=make_payload(
                        matches=[
                            {
                                "pattern": m.pattern_name,
                                "category": m.category,
                                "matched_text": m.matched_text,
                                "source": m.source,
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
                "[AgentGuard] injection_detected: %s -> %s\n  flags: %s\n  trust_score: %.2f",
                g.agent_id,
                max_severity.upper(),
                flags,
                trust_score,
            )

            if g.mode == "enforce":
                raise AgentGuardException(
                    f"Injection detected in enforce mode — call blocked. Flags: {flags}"
                )

        # --- 5. Policy violation events -----------------------------------
        if policy_violations:
            g._bus.emit(
                SecurityEvent(
                    session_id=g.session_id,
                    agent_id=g.agent_id,
                    source="sdk",
                    event_type="policy_violation",
                    severity="critical",
                    payload=make_payload(violations=policy_violations),
                    flags=["policy:tool_not_allowed"],
                    parent_event_id=llm_event_id,
                )
            )
            logger.error(
                "[AgentGuard] policy_violation: %s -> CRITICAL\n  %s",
                g.agent_id,
                "\n  ".join(policy_violations),
            )
            if g.mode == "enforce":
                raise AgentGuardException(
                    f"Policy violation in enforce mode — call blocked. "
                    f"Violations: {policy_violations}"
                )

        # --- 6. Actual API call -------------------------------------------
        response = self._underlying.create(**kwargs)

        # --- 7. Emit tool_call events for each tool use block in response ---
        if hasattr(response, "content"):
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    tool_name = getattr(block, "name", "unknown")
                    tool_input = getattr(block, "input", {})

                    # Trust check before tool execution.
                    if g._trust_scorer.should_flag(g.session_id, tool_name):
                        trust_score = g._trust_scorer.score(g.session_id)
                        g._bus.emit(
                            SecurityEvent(
                                session_id=g.session_id,
                                agent_id=g.agent_id,
                                source="sdk",
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
                                parent_event_id=llm_event_id,
                                metadata={"trust_score": trust_score},
                            )
                        )
                        logger.warning(
                            "[AgentGuard] trust_flag: %s -> WARNING\n  reason: low-trust "
                            "session (score=%.2f) attempting %s",
                            g.agent_id,
                            trust_score,
                            tool_name,
                        )

                    # Policy check for the actual tool being called.
                    policy_result = g._policy_engine.check(g.agent_id, tool_name)
                    tool_severity = "info" if policy_result.allowed else "critical"
                    tool_flags = [] if policy_result.allowed else [policy_result.rule_name]

                    g._bus.emit(
                        SecurityEvent(
                            session_id=g.session_id,
                            agent_id=g.agent_id,
                            source="sdk",
                            event_type="tool_call" if policy_result.allowed else "policy_violation",
                            severity=tool_severity,
                            payload=make_payload(
                                tool_name=tool_name,
                                tool_input=tool_input,
                                policy_result={
                                    "allowed": policy_result.allowed,
                                    "reason": policy_result.reason,
                                    "rule_name": policy_result.rule_name,
                                },
                            ),
                            flags=tool_flags,
                            parent_event_id=llm_event_id,
                        )
                    )
                    log_msg = (
                        "[AgentGuard] tool_call: %s.%s -> %s"
                        if policy_result.allowed
                        else "[AgentGuard] policy_violation: %s.%s -> CRITICAL\n  reason: %s"
                    )
                    if policy_result.allowed:
                        logger.info(log_msg, g.agent_id, tool_name, "INFO")
                    else:
                        logger.error(log_msg, g.agent_id, tool_name, policy_result.reason)

        return response

    def __getattr__(self, name: str) -> Any:
        """Proxy any other attribute (e.g., .stream, .with_raw_response) unchanged."""
        return getattr(self._underlying, name)


class GuardedClient:
    """Drop-in replacement for ``anthropic.Anthropic`` with security observability.

    Wraps the official Anthropic client, intercepting ``messages.create``
    calls to perform injection detection, tool policy enforcement, and trust
    scoring before and after each LLM call.

    Parameters
    ----------
    client:
        An instantiated ``anthropic.Anthropic`` (or ``AsyncAnthropic``) client.
    session_id:
        Identifier grouping all events from a single agent run. Auto-generated
        if not provided.
    agent_id:
        Logical name of the agent (e.g. ``"researcher"``). Used in policy
        lookups and event records.
    policy_path:
        Path to a YAML policy file. If ``None``, all tools are permitted.
    bus:
        Shared :class:`~agentguard.bus.EventBus`. If ``None``, a new bus is
        created with no persistent store.
    mode:
        ``"observe"`` (default) — detect and log, but never block.
        ``"enforce"`` — raise :class:`AgentGuardException` on any violation.
    trust_scorer:
        Shared :class:`~agentguard.engine.trust.TrustScorer`. Useful when
        multiple ``GuardedClient`` instances share a session.
    use_embeddings:
        Pass ``True`` to enable the embedding-based injection detection pass.
    audit_log:
        Path to a JSONL audit file. Every event is appended to this file
        durably and synchronously — survives process restarts. Defaults to
        ``"agentguard_audit.jsonl"`` in the current directory when set to the
        sentinel ``True``, or pass an explicit path string. Set to ``None``
        (default) to disable file auditing and use the in-memory bus only.

    Example
    -------
    ::

        import anthropic
        from agentguard import GuardedClient

        raw_client = anthropic.Anthropic()
        client = GuardedClient(
            raw_client,
            agent_id="researcher",
            policy_path="policy.yaml",
            audit_log="logs/audit.jsonl",  # durable, survives restarts
        )

        # Use exactly like the real client:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Summarize this document..."}],
        )
    """

    def __init__(
        self,
        client: Any,
        session_id: Optional[str] = None,
        agent_id: str = "default",
        policy_path: Optional[str] = None,
        bus: Optional[EventBus] = None,
        mode: Literal["observe", "enforce"] = "observe",
        trust_scorer: Optional[TrustScorer] = None,
        use_embeddings: bool = False,
        audit_log: Optional[str] = None,
    ) -> None:
        self._client = client
        self.session_id = session_id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.mode = mode

        self._bus = bus if bus is not None else EventBus()
        self._injection_detector = InjectionDetector(use_embeddings=use_embeddings)
        self._policy_engine = ToolPolicyEngine(policy_path=policy_path)
        self._trust_scorer = trust_scorer if trust_scorer is not None else TrustScorer()

        # Wire up durable audit logging if requested.
        self._audit_logger: Optional[AuditLogger] = None
        if audit_log is not None:
            self._audit_logger = AuditLogger(path=audit_log)
            self._bus.subscribe(self._audit_logger)
            logger.info("[AgentGuard] Audit log -> %s", audit_log)

        # Emit session_start.
        self._bus.emit(
            SecurityEvent(
                session_id=self.session_id,
                agent_id=self.agent_id,
                source="sdk",
                event_type="session_start",
                severity="info",
                payload=make_payload(mode=mode, policy_path=policy_path or "none"),
            )
        )
        logger.info("[AgentGuard] Session %s started (agent=%s, mode=%s)", self.session_id, agent_id, mode)

    @property
    def messages(self) -> GuardedMessages:
        """Intercepted messages namespace."""
        return GuardedMessages(self)

    def record_external_content(self, source_type: str = "file") -> None:
        """Signal that this session processed external content.

        Call this before making an LLM call that processes untrusted data
        (web scrape, file read, user upload) to trigger trust score degradation.
        """
        self._trust_scorer.record_external_content(self.session_id, source_type)

    def trust_score(self) -> float:
        """Return the current trust score for this session."""
        return self._trust_scorer.score(self.session_id)

    def end_session(self) -> None:
        """Emit a session_end event and log a summary."""
        self._bus.emit(
            SecurityEvent(
                session_id=self.session_id,
                agent_id=self.agent_id,
                source="sdk",
                event_type="session_end",
                severity="info",
                payload=make_payload(trust_score=self.trust_score()),
            )
        )

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes directly to the underlying client."""
        return getattr(self._client, name)
