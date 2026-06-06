"""AsyncGuardedClient — drop-in wrapper around anthropic.AsyncAnthropic with security instrumentation."""

from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncGenerator, Literal, Optional

from .audit import AuditLogger
from .bus import EventBus
from .client import AgentGuardException, AgentGuardKilled
from .control import ApprovalGate, ApprovalRequest, KillSwitch, get_default_kill_switch
from .engine.injection import InjectionDetector
from .engine.policy import ToolPolicyEngine
from .engine.trust import TrustScorer
from .events import SecurityEvent, make_payload

logger = logging.getLogger(__name__)


async def _raise_if_killed_async(guard: Any, source: str) -> None:
    """Async variant of :func:`agentguard.client._raise_if_killed`.

    ``guard`` is any async guarded-client-like object exposing ``session_id``,
    ``agent_id``, ``_bus`` (with ``emit_async``), and ``_kill_switch`` — shared by
    :class:`AsyncGuardedClient` and :class:`~agentguard.openai_client.AsyncGuardedOpenAI`.
    """
    if not guard._kill_switch.is_killed(guard.session_id):
        return
    await guard._bus.emit_async(
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
    raise AgentGuardKilled(f"Session '{guard.session_id}' has been killed via KillSwitch — call blocked.")


async def _dispatch_violation_async(
    guard: Any,
    *,
    parent_event_id: str,
    label: str,
    action: str,
    reason: str,
    payload: dict,
    block_in_enforce: bool = True,
) -> None:
    """Async variant of :func:`agentguard.client._dispatch_violation`.

    Uses ``await guard._bus.emit_async(...)`` and ``await
    guard._approval_gate.request_async(...)`` so the event loop is never blocked.
    """
    if guard.mode == "observe":
        return

    if guard.mode == "enforce":
        if block_in_enforce:
            raise AgentGuardException(f"{label} in enforce mode — call blocked. {reason}")
        return

    # interactive
    await guard._bus.emit_async(
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
    decision = await guard._approval_gate.request_async(
        ApprovalRequest(
            session_id=guard.session_id,
            agent_id=guard.agent_id,
            action=action,
            reason=reason,
            payload=payload,
        )
    )
    await guard._bus.emit_async(
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


class _AsyncAccumulatingTextStream:
    """Wraps an async text stream and accumulates yielded chunks into a buffer."""

    def __init__(self, real_text_stream: Any, buffer: list[str]) -> None:
        self._real = real_text_stream
        self._buffer = buffer

    def __aiter__(self) -> "_AsyncAccumulatingTextStream":
        return self._gen()  # type: ignore[return-value]

    async def _gen(self) -> AsyncGenerator[str, None]:  # type: ignore[override]
        async for text in self._real:
            self._buffer.append(text)
            yield text


class _AsyncStreamProxy:
    """Proxy around the real AsyncMessageStream that accumulates text chunks."""

    def __init__(self, real_stream: Any, buffer: list[str]) -> None:
        self._real = real_stream
        self._buffer = buffer

    @property
    def text_stream(self) -> _AsyncAccumulatingTextStream:
        return _AsyncAccumulatingTextStream(self._real.text_stream, self._buffer)

    async def get_final_message(self) -> Any:
        """Await the final message from the underlying stream."""
        return await self._real.get_final_message()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class AsyncGuardedStream:
    """Async context manager that wraps ``client.messages.stream()`` with security instrumentation.

    Performs injection scanning before the stream opens, emits a ``llm_call`` event on
    open, accumulates streamed text, and emits a ``llm_call_complete`` event on close
    (including a scan of the complete model output for output-side injection).
    """

    def __init__(self, guard: "AsyncGuardedClient", **kwargs: Any) -> None:
        self._guard = guard
        self._kwargs = kwargs
        self._cm: Any = None
        self._stream_proxy: Optional[_AsyncStreamProxy] = None
        self._text_buffer: list[str] = []
        self._llm_event_id: str = ""

    async def __aenter__(self) -> _AsyncStreamProxy:
        g = self._guard
        await _raise_if_killed_async(g, source="sdk")
        messages: list[dict] = self._kwargs.get("messages", [])
        tools: list[dict] = self._kwargs.get("tools", [])
        system: str = self._kwargs.get("system", "")
        model: str = self._kwargs.get("model", "unknown")

        scan_corpus = messages.copy()
        if system:
            scan_corpus = [{"role": "system", "content": system}] + scan_corpus

        self._llm_event_id = str(uuid.uuid4())
        injection_matches = g._injection_detector.scan_messages(scan_corpus)

        await g._bus.emit_async(
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
            await g._bus.emit_async(
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
            await _dispatch_violation_async(
                g,
                parent_event_id=self._llm_event_id,
                label="Injection detected",
                action=f"stream:{model}",
                reason=f"Flags: {flags}",
                payload={"flags": flags, "model": model},
            )

        self._cm = g._client.messages.stream(**self._kwargs)
        real_stream = await self._cm.__aenter__()
        self._stream_proxy = _AsyncStreamProxy(real_stream, self._text_buffer)
        return self._stream_proxy

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        result = await self._cm.__aexit__(exc_type, exc_val, exc_tb)
        if exc_type is None and self._stream_proxy is not None:
            g = self._guard
            accumulated_text = "".join(self._text_buffer)

            output_matches = g._injection_detector.scan(accumulated_text)

            tool_calls: list[dict] = []
            try:
                final_msg = await self._stream_proxy.get_final_message()
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

            complete_severity = "critical" if output_matches else "info"
            await g._bus.emit_async(
                SecurityEvent(
                    session_id=g.session_id,
                    agent_id=g.agent_id,
                    source="sdk",
                    event_type="llm_call_complete",
                    severity=complete_severity,
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


class AsyncGuardedMessages:
    """Proxy for ``client.messages`` that intercepts async ``.create()`` and ``.stream()`` calls."""

    def __init__(self, guard: "AsyncGuardedClient") -> None:
        self._guard = guard
        self._underlying = guard._client.messages

    async def create(self, **kwargs: Any) -> Any:
        """Intercept an async messages.create call to perform security checks."""
        g = self._guard
        await _raise_if_killed_async(g, source="sdk")
        messages: list[dict] = kwargs.get("messages", [])
        tools: list[dict] = kwargs.get("tools", [])
        system: str = kwargs.get("system", "")
        model: str = kwargs.get("model", "unknown")

        scan_corpus = messages.copy()
        if system:
            scan_corpus = [{"role": "system", "content": system}] + scan_corpus

        llm_event_id = str(uuid.uuid4())

        # --- 1. Injection scan
        injection_matches = g._injection_detector.scan_messages(scan_corpus)

        # --- 2. Tool policy check
        policy_violations: list[str] = []
        for tool_def in tools:
            tool_name = tool_def.get("name", "unknown")
            result = g._policy_engine.check(g.agent_id, tool_name)
            if not result.allowed:
                policy_violations.append(
                    f"{tool_name}: {result.reason} [{result.rule_name}]"
                )

        # --- 3. Emit llm_call event
        await g._bus.emit_async(
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

        # --- 4. Injection events
        if injection_matches:
            flags = [m.flag for m in injection_matches]
            max_severity = (
                "critical"
                if any(m.severity == "critical" for m in injection_matches)
                else "warning"
            )
            g._trust_scorer.record_injection_flag(g.session_id)
            trust_score = g._trust_scorer.score(g.session_id)

            await g._bus.emit_async(
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

            await _dispatch_violation_async(
                g,
                parent_event_id=llm_event_id,
                label="Injection detected",
                action=f"llm_call:{model}",
                reason=f"Flags: {flags}",
                payload={"flags": flags, "model": model},
            )

        # --- 5. Policy violations
        if policy_violations:
            await g._bus.emit_async(
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
            await _dispatch_violation_async(
                g,
                parent_event_id=llm_event_id,
                label="Policy violation",
                action=f"llm_call:{model}",
                reason=f"Violations: {policy_violations}",
                payload={"violations": policy_violations},
            )

        # --- 6. Actual API call
        response = await self._underlying.create(**kwargs)

        # --- 7. Emit tool_call events for each tool use block in response
        if hasattr(response, "content"):
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    tool_name = getattr(block, "name", "unknown")
                    tool_input = getattr(block, "input", {})

                    if g._trust_scorer.should_flag(g.session_id, tool_name):
                        trust_score = g._trust_scorer.score(g.session_id)
                        await g._bus.emit_async(
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
                        # Trust flags are warning-only — never hard-block in enforce mode —
                        # but interactive mode still routes them through the approval gate.
                        await _dispatch_violation_async(
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

                    policy_result = g._policy_engine.check(g.agent_id, tool_name)
                    tool_severity = "info" if policy_result.allowed else "critical"
                    tool_flags = [] if policy_result.allowed else [policy_result.rule_name]

                    await g._bus.emit_async(
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

                    # Argument-level constraint check (path traversal, SSRF, etc.)
                    violations = g._policy_engine.check_arguments(g.agent_id, tool_name, tool_input)
                    if violations:
                        violation_flags = [v.flag for v in violations]
                        await g._bus.emit_async(
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
                        await _dispatch_violation_async(
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

        return response

    def stream(self, **kwargs: Any) -> AsyncGuardedStream:
        """Return an async context manager that intercepts the streaming response."""
        return AsyncGuardedStream(self._guard, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._underlying, name)


class AsyncGuardedClient:
    """Drop-in replacement for ``anthropic.AsyncAnthropic`` with security observability.

    Wraps the official async Anthropic client, intercepting ``messages.create``
    and ``messages.stream`` calls to perform injection detection, tool policy
    enforcement, and trust scoring in a fully async context.

    Parameters
    ----------
    client:
        An instantiated ``anthropic.AsyncAnthropic`` client.
    session_id:
        Identifier grouping all events from a single agent run. Auto-generated
        if not provided.
    agent_id:
        Logical name of the agent. Used in policy lookups and event records.
    policy_path:
        Path to a YAML policy file. If ``None``, all tools are permitted.
    bus:
        Shared :class:`~agentguard.bus.EventBus`. If ``None``, a new bus is
        created with no persistent store.
    mode:
        ``"observe"`` (default) — detect and log, but never block.
        ``"enforce"`` — raise :class:`~agentguard.client.AgentGuardException` on any violation.
        ``"interactive"`` — route violations through ``approval_gate`` for a
        human (or programmatic) decision; raises
        :class:`~agentguard.client.AgentGuardException` if denied.
    trust_scorer:
        Shared :class:`~agentguard.engine.trust.TrustScorer`.
    use_embeddings:
        Pass ``True`` to enable the embedding-based injection detection pass.
    approval_gate:
        :class:`~agentguard.control.ApprovalGate` used in ``mode="interactive"``.
        If ``None``, a default gate (CLI y/N prompt) is created when needed.
    kill_switch:
        Shared :class:`~agentguard.control.KillSwitch`. Defaults to the
        process-wide singleton from :func:`~agentguard.control.get_default_kill_switch`.
    audit_log:
        Path to a JSONL audit file for durable event persistence.

    Example
    -------
    ::

        import anthropic
        from agentguard import AsyncGuardedClient

        client = AsyncGuardedClient(
            anthropic.AsyncAnthropic(),
            agent_id="researcher",
            mode="observe",
        )

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": user_input}],
        )
    """

    def __init__(
        self,
        client: Any,
        session_id: Optional[str] = None,
        agent_id: str = "default",
        policy_path: Optional[str] = None,
        bus: Optional[EventBus] = None,
        mode: Literal["observe", "enforce", "interactive"] = "observe",
        trust_scorer: Optional[TrustScorer] = None,
        use_embeddings: bool = False,
        approval_gate: Optional[ApprovalGate] = None,
        kill_switch: Optional[KillSwitch] = None,
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
        self._approval_gate = approval_gate or (ApprovalGate() if mode == "interactive" else None)
        self._kill_switch = kill_switch if kill_switch is not None else get_default_kill_switch()

        self._audit_logger: Optional[AuditLogger] = None
        if audit_log is not None:
            self._audit_logger = AuditLogger(path=audit_log)
            self._bus.subscribe(self._audit_logger)
            logger.info("[AgentGuard] Audit log -> %s", audit_log)

        # Emit session_start synchronously so it's visible even before the first await.
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
            "[AgentGuard] Async session %s started (agent=%s, mode=%s)",
            self.session_id,
            agent_id,
            mode,
        )

    @property
    def messages(self) -> AsyncGuardedMessages:
        """Intercepted async messages namespace."""
        return AsyncGuardedMessages(self)

    def record_external_content(self, source_type: str = "file") -> None:
        """Signal that this session processed external, untrusted content."""
        self._trust_scorer.record_external_content(self.session_id, source_type)

    def trust_score(self) -> float:
        """Return the current trust score for this session."""
        return self._trust_scorer.score(self.session_id)

    async def end_session(self) -> None:
        """Emit a session_end event."""
        await self._bus.emit_async(
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
        """Proxy all other attributes directly to the underlying async client."""
        return getattr(self._client, name)
