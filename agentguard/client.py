"""GuardedClient — drop-in wrapper around anthropic.Anthropic with security instrumentation."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Generator
from typing import Any, Literal

from .audit import AuditLogger
from .bus import EventBus
from .control import ApprovalGate, ApprovalRequest, KillSwitch, get_default_kill_switch
from .engine.injection import InjectionDetector
from .engine.policy import ToolPolicyEngine
from .engine.trust import TrustScorer
from .events import SecurityEvent, make_payload

logger = logging.getLogger(__name__)


class AgentGuardException(Exception):
    """Raised in enforce mode when a security violation blocks execution."""


class AgentGuardKilled(AgentGuardException):
    """Raised when a session (or the global switch) has been halted via KillSwitch."""


def _raise_if_killed(guard: Any, source: str) -> None:
    """Raise :class:`AgentGuardKilled` if this session or the global switch has been tripped.

    ``guard`` is any guarded-client-like object exposing ``session_id``,
    ``agent_id``, ``_bus``, and ``_kill_switch`` — shared by
    :class:`GuardedClient` and :class:`~agentguard.openai_client.GuardedOpenAI`.

    Emits a critical ``session_end`` event before raising, so the kill is
    visible in the audit trail even though no further work happens.
    """
    if not guard._kill_switch.is_killed(guard.session_id):
        return
    guard._bus.emit(
        SecurityEvent(
            session_id=guard.session_id,
            agent_id=guard.agent_id,
            source=source,
            event_type="session_end",
            severity="critical",
            payload=make_payload(reason="kill_switch_tripped"),
            flags=["kill:tripped"],
        )
    )
    logger.critical("[AgentGuard] Session %s halted by kill switch", guard.session_id)
    raise AgentGuardKilled(
        f"Session '{guard.session_id}' has been killed via KillSwitch — call blocked."
    )


def _handle_engine_error(
    guard: Any,
    *,
    source: str,
    phase: str,
    exc: BaseException,
    parent_event_id: str | None = None,
) -> None:
    """Handle an internal failure of the detection/policy engines.

    ``guard`` is any guarded-client-like object exposing ``mode``,
    ``session_id``, ``agent_id``, and ``_bus`` — shared by
    :class:`GuardedClient` and :class:`~agentguard.openai_client.GuardedOpenAI`.

    Detection logic (``InjectionDetector.scan``, ``ToolPolicyEngine.check_arguments``,
    etc.) is not allowed to silently break or bypass security. An internal error
    here always emits a critical ``engine_error`` event, and:

    * ``enforce``/``interactive`` — **fail closed**: raise
      :class:`AgentGuardException`, blocking the call. An attacker who can
      crash the detector must not be able to use that crash to bypass checks.
    * ``observe`` — **fail open but loud**: the call proceeds (this function
      simply returns), but the failure is recorded just as critically.
    """
    logger.exception("[AgentGuard] Internal security engine error during %s", phase)
    guard._bus.emit(
        SecurityEvent(
            session_id=guard.session_id,
            agent_id=guard.agent_id,
            source=source,
            event_type="engine_error",
            severity="critical",
            payload=make_payload(phase=phase, error=str(exc), error_type=type(exc).__name__),
            flags=["engine:internal_error"],
            parent_event_id=parent_event_id,
        )
    )
    if guard.mode in ("enforce", "interactive"):
        raise AgentGuardException(
            f"Internal security engine error during '{phase}' — failing closed "
            f"(mode={guard.mode}). {exc}"
        )


def _evaluate_tool_use_sync(
    g: GuardedClient, llm_event_id: str, tool_name: str, tool_input: dict
) -> None:
    """Run the trust/policy/argument-constraint checks for one ``tool_use`` block.

    Factored out of :meth:`GuardedMessages.create` so the whole
    security-evaluation section for a single tool call can be wrapped in one
    try/except at the call site — an internal failure here (a bad regex, a
    malformed constraint) must not silently let the action through unchecked.
    See :func:`_handle_engine_error`.
    """
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
        # Trust flags are warning-only — never hard-block in enforce mode —
        # but interactive mode still routes them through the approval gate,
        # giving humans finer-grained control than a blanket policy.
        _dispatch_violation(
            g,
            parent_event_id=llm_event_id,
            label="Trust flag",
            action=f"tool_call:{tool_name}",
            reason=(
                f"Agent received output from low-trust session "
                f"(score={trust_score:.2f}), attempting {tool_name}"
            ),
            payload={"tool_name": tool_name, "trust_score": trust_score},
            block_in_enforce=False,
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

    # Argument-level constraint check (path traversal, SSRF, etc.)
    violations = g._policy_engine.check_arguments(g.agent_id, tool_name, tool_input)
    if violations:
        violation_flags = [v.flag for v in violations]
        g._bus.emit(
            SecurityEvent(
                session_id=g.session_id,
                agent_id=g.agent_id,
                source="sdk",
                event_type="policy_violation",
                severity="critical",
                payload=make_payload(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    violations=[
                        {
                            "constraint": v.constraint,
                            "argument": v.argument,
                            "value": v.value,
                            "detail": v.detail,
                        }
                        for v in violations
                    ],
                ),
                flags=violation_flags,
                parent_event_id=llm_event_id,
            )
        )
        logger.error(
            "[AgentGuard] policy_violation: %s.%s -> CRITICAL\n  argument constraints: %s",
            g.agent_id,
            tool_name,
            [v.detail for v in violations],
        )
        _dispatch_violation(
            g,
            parent_event_id=llm_event_id,
            label="Argument constraint violation",
            action=f"tool_call:{tool_name}",
            reason=f"Flags: {violation_flags}",
            payload={
                "tool_name": tool_name,
                "tool_input": tool_input,
                "flags": violation_flags,
            },
        )


def _dispatch_violation(
    guard: Any,
    *,
    parent_event_id: str,
    label: str,
    action: str,
    reason: str,
    payload: dict,
    block_in_enforce: bool = True,
) -> None:
    """Apply mode-aware handling for an already-detected and already-logged violation.

    ``guard`` is any guarded-client-like object exposing ``mode``, ``session_id``,
    ``agent_id``, ``_bus``, and ``_approval_gate`` — shared by :class:`GuardedClient`
    and :class:`~agentguard.openai_client.GuardedOpenAI`.

    * ``observe`` — no-op; the violation event has already been emitted.
    * ``enforce`` — raise :class:`AgentGuardException` immediately, unless
      ``block_in_enforce`` is False (used for trust flags, which are
      warning-only and never hard-block by themselves in enforce mode).
    * ``interactive`` — emit ``approval_required``, route the decision through
      ``guard._approval_gate``, emit ``approval_granted``/``approval_denied``,
      and raise :class:`AgentGuardException` if the approver denies. Unlike
      enforce mode, this applies even when ``block_in_enforce`` is False — a
      human's explicit "deny" always blocks, giving interactive mode finer
      control than a blanket policy.
    """
    if guard.mode == "observe":
        return

    if guard.mode == "enforce":
        if block_in_enforce:
            raise AgentGuardException(f"{label} in enforce mode — call blocked. {reason}")
        return

    # interactive
    guard._bus.emit(
        SecurityEvent(
            session_id=guard.session_id,
            agent_id=guard.agent_id,
            source="sdk",
            event_type="approval_required",
            severity="warning",
            payload=make_payload(action=action, reason=reason, context=payload),
            parent_event_id=parent_event_id,
        )
    )
    decision = guard._approval_gate.request(
        ApprovalRequest(
            session_id=guard.session_id,
            agent_id=guard.agent_id,
            action=action,
            reason=reason,
            payload=payload,
        )
    )
    guard._bus.emit(
        SecurityEvent(
            session_id=guard.session_id,
            agent_id=guard.agent_id,
            source="sdk",
            event_type="approval_granted" if decision == "approve" else "approval_denied",
            severity="info" if decision == "approve" else "critical",
            payload=make_payload(action=action, decision=decision),
            parent_event_id=parent_event_id,
        )
    )
    if decision == "deny":
        raise AgentGuardException(
            f"{label} denied by approver in interactive mode — call blocked. {reason}"
        )


class _SyncAccumulatingTextStream:
    """Wraps a sync text stream and accumulates yielded chunks into a buffer."""

    def __init__(self, real_text_stream: Any, buffer: list[str]) -> None:
        self._real = real_text_stream
        self._buffer = buffer

    def __iter__(self) -> Generator[str, None, None]:
        for text in self._real:
            self._buffer.append(text)
            yield text


class _SyncStreamProxy:
    """Proxy around the real MessageStream that accumulates text chunks."""

    def __init__(self, real_stream: Any, buffer: list[str]) -> None:
        self._real = real_stream
        self._buffer = buffer

    @property
    def text_stream(self) -> _SyncAccumulatingTextStream:
        return _SyncAccumulatingTextStream(self._real.text_stream, self._buffer)

    def get_final_message(self) -> Any:
        return self._real.get_final_message()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class GuardedStream:
    """Sync context manager that wraps ``client.messages.stream()`` with security instrumentation.

    Performs injection scanning before the stream opens, emits a ``llm_call`` event on
    open, accumulates streamed text, and emits a ``llm_call_complete`` event on close.
    """

    def __init__(self, guard: GuardedClient, **kwargs: Any) -> None:
        self._guard = guard
        self._kwargs = kwargs
        self._cm: Any = None
        self._stream_proxy: _SyncStreamProxy | None = None
        self._text_buffer: list[str] = []
        self._llm_event_id: str = ""

    def __enter__(self) -> _SyncStreamProxy:
        g = self._guard
        _raise_if_killed(g, source="sdk")
        messages: list[dict] = self._kwargs.get("messages", [])
        tools: list[dict] = self._kwargs.get("tools", [])
        system: str = self._kwargs.get("system", "")
        model: str = self._kwargs.get("model", "unknown")

        scan_corpus = messages.copy()
        if system:
            scan_corpus = [{"role": "system", "content": system}] + scan_corpus

        self._llm_event_id = str(uuid.uuid4())
        injection_matches = g._injection_detector.scan_messages(scan_corpus)

        g._bus.emit(
            SecurityEvent(
                event_id=self._llm_event_id,
                session_id=g.session_id,
                agent_id=g.agent_id,
                source="sdk",
                event_type="llm_call",
                severity="info",
                payload=make_payload(
                    model=model,
                    message_count=len(messages),
                    tool_count=len(tools),
                    streaming=True,
                ),
            )
        )

        if injection_matches:
            flags = [m.flag for m in injection_matches]
            max_severity = (
                "critical"
                if any(m.severity == "critical" for m in injection_matches)
                else "warning"
            )
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
                            }
                            for m in injection_matches
                        ]
                    ),
                    flags=flags,
                    parent_event_id=self._llm_event_id,
                    metadata={"trust_score": trust_score},
                )
            )
            _dispatch_violation(
                g,
                parent_event_id=self._llm_event_id,
                label="Injection detected",
                action=f"stream:{model}",
                reason=f"Flags: {flags}",
                payload={"flags": flags, "model": model},
            )

        self._cm = g._client.messages.stream(**self._kwargs)
        real_stream = self._cm.__enter__()
        self._stream_proxy = _SyncStreamProxy(real_stream, self._text_buffer)
        return self._stream_proxy

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        result = self._cm.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is None and self._stream_proxy is not None:
            g = self._guard
            accumulated_text = "".join(self._text_buffer)
            output_matches = g._injection_detector.scan(accumulated_text)

            tool_calls: list[dict] = []
            try:
                final_msg = self._stream_proxy.get_final_message()
                if hasattr(final_msg, "content"):
                    tool_calls = [
                        {
                            "name": getattr(b, "name", ""),
                            "input": getattr(b, "input", {}),
                        }
                        for b in final_msg.content
                        if getattr(b, "type", None) == "tool_use"
                    ]
            except Exception:
                logger.debug("Could not retrieve final message after stream close")

            g._bus.emit(
                SecurityEvent(
                    session_id=g.session_id,
                    agent_id=g.agent_id,
                    source="sdk",
                    event_type="llm_call_complete",
                    severity="critical" if output_matches else "info",
                    payload=make_payload(
                        text=accumulated_text,
                        tool_calls=tool_calls,
                        output_injection_detected=bool(output_matches),
                    ),
                    flags=[m.flag for m in output_matches],
                    parent_event_id=self._llm_event_id,
                )
            )
        return result


class GuardedMessages:
    """Proxy for ``client.messages`` that intercepts ``.create()`` calls.

    Performs pre-call security checks (injection scan, policy check) and
    post-call event emission (tool call tracking) without altering the
    underlying response object.
    """

    def __init__(self, guard: GuardedClient) -> None:
        self._guard = guard
        self._underlying = guard._client.messages

    def create(self, **kwargs: Any) -> Any:
        """Intercept a messages.create call to perform security checks."""
        g = self._guard
        _raise_if_killed(g, source="sdk")
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
        try:
            injection_matches = g._injection_detector.scan_messages(scan_corpus)
        except Exception as exc:
            injection_matches = []
            _handle_engine_error(
                g, source="sdk", phase="injection_scan", parent_event_id=llm_event_id, exc=exc
            )

        # --- 2. Tool policy check -----------------------------------------
        policy_violations: list[str] = []
        try:
            for tool_def in tools:
                tool_name = tool_def.get("name", "unknown")
                result = g._policy_engine.check(g.agent_id, tool_name)
                if not result.allowed:
                    policy_violations.append(f"{tool_name}: {result.reason} [{result.rule_name}]")
        except Exception as exc:
            policy_violations = []
            _handle_engine_error(
                g,
                source="sdk",
                phase="tool_definition_policy_check",
                parent_event_id=llm_event_id,
                exc=exc,
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
            max_severity = (
                "critical"
                if any(m.severity == "critical" for m in injection_matches)
                else "warning"
            )

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

            _dispatch_violation(
                g,
                parent_event_id=llm_event_id,
                label="Injection detected",
                action=f"llm_call:{model}",
                reason=f"Flags: {flags}",
                payload={"flags": flags, "model": model},
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
            _dispatch_violation(
                g,
                parent_event_id=llm_event_id,
                label="Policy violation",
                action=f"llm_call:{model}",
                reason=f"Violations: {policy_violations}",
                payload={"violations": policy_violations},
            )

        # --- 6. Actual API call -------------------------------------------
        response = self._underlying.create(**kwargs)

        # --- 7. Emit tool_call events for each tool use block in response ---
        if hasattr(response, "content"):
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    tool_name = getattr(block, "name", "unknown")
                    tool_input = getattr(block, "input", {})

                    try:
                        _evaluate_tool_use_sync(g, llm_event_id, tool_name, tool_input)
                    except AgentGuardException:
                        raise
                    except Exception as exc:
                        _handle_engine_error(
                            g,
                            source="sdk",
                            phase="post_call_tool_evaluation",
                            parent_event_id=llm_event_id,
                            exc=exc,
                        )

        return response

    def stream(self, **kwargs: Any) -> GuardedStream:
        """Return a context manager that intercepts the streaming response."""
        return GuardedStream(self._guard, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Proxy any other attribute (e.g., .with_raw_response) unchanged."""
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
        ``"interactive"`` — route violations through ``approval_gate`` for a
        human (or programmatic) decision; raises :class:`AgentGuardException`
        if denied. See :class:`~agentguard.control.ApprovalGate`.
    trust_scorer:
        Shared :class:`~agentguard.engine.trust.TrustScorer`. Useful when
        multiple ``GuardedClient`` instances share a session.
    use_embeddings:
        Pass ``True`` to enable the embedding-based injection detection pass.
    approval_gate:
        :class:`~agentguard.control.ApprovalGate` used in ``mode="interactive"``
        to route violations to a human or programmatic approver. If ``None``,
        a default gate (CLI y/N prompt) is created when ``mode="interactive"``.
    kill_switch:
        Shared :class:`~agentguard.control.KillSwitch`. If ``None``, the
        process-wide default from :func:`~agentguard.control.get_default_kill_switch`
        is used, so any code holding that singleton can halt this session.
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
        session_id: str | None = None,
        agent_id: str = "default",
        policy_path: str | None = None,
        bus: EventBus | None = None,
        mode: Literal["observe", "enforce", "interactive"] = "observe",
        trust_scorer: TrustScorer | None = None,
        use_embeddings: bool = False,
        approval_gate: ApprovalGate | None = None,
        kill_switch: KillSwitch | None = None,
        audit_log: str | None = None,
    ) -> None:
        self._client = client
        self.session_id = session_id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.mode = mode

        self._bus = bus if bus is not None else EventBus()
        self._injection_detector = InjectionDetector(use_embeddings=use_embeddings)
        self._policy_engine = ToolPolicyEngine(policy_path=policy_path)
        self._trust_scorer = trust_scorer if trust_scorer is not None else TrustScorer()
        self._approval_gate = approval_gate or (ApprovalGate() if mode == "interactive" else None)
        self._kill_switch = kill_switch if kill_switch is not None else get_default_kill_switch()

        # Wire up durable audit logging if requested.
        self._audit_logger: AuditLogger | None = None
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
        logger.info(
            "[AgentGuard] Session %s started (agent=%s, mode=%s)", self.session_id, agent_id, mode
        )

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
