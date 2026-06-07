"""GuardedOpenAI / AsyncGuardedOpenAI — drop-in wrappers around the OpenAI SDK.

Mirrors :mod:`agentguard.client` / :mod:`agentguard.async_client` for the OpenAI
Chat Completions API: the same injection scan → policy check → trust scoring →
argument-constraint pipeline, the same event types, and the same ``mode``
semantics (``observe`` / ``enforce`` / ``interactive``), adapted to OpenAI's
request/response shapes.

Key shape differences from Anthropic that this module bridges:

* Tool *definitions* are nested — ``{"type": "function", "function": {"name": ...}}``
  — rather than Anthropic's flat ``{"name": ..., "input_schema": ...}``.
* Tool *calls* surface on ``response.choices[i].message.tool_calls``, each with
  ``function.name`` and ``function.arguments`` — the latter a **JSON string**
  that must be parsed before constraint checking, rather than Anthropic's
  already-structured ``input`` dict.

Streaming is not yet intercepted — ``client.chat.completions`` proxies
``.create()``; any other attribute (including streaming helpers) is forwarded
transparently to the underlying OpenAI client via ``__getattr__``.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Literal

from .async_client import (
    _dispatch_violation_async,
    _handle_engine_error_async,
    _raise_if_killed_async,
)
from .audit import AuditLogger
from .bus import EventBus
from .client import AgentGuardException, _dispatch_violation, _handle_engine_error, _raise_if_killed
from .control import ApprovalGate, KillSwitch, get_default_kill_switch
from .engine.injection import InjectionDetector
from .engine.policy import ToolPolicyEngine
from .engine.trust import TrustScorer
from .events import SecurityEvent, make_payload

logger = logging.getLogger(__name__)


def _tool_def_name(tool_def: dict) -> str:
    """Extract a tool's name from an OpenAI-format
    ``{"type": "function", "function": {...}}`` definition."""
    return tool_def.get("function", {}).get("name", "unknown")


def _parse_tool_arguments(raw: Any) -> dict:
    """Parse a tool call's ``function.arguments`` — a JSON-encoded string in the OpenAI format.

    Falls back to wrapping unparseable values rather than raising, so a malformed
    or truncated argument string degrades to a constraint-check miss instead of
    crashing the intercepted call.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": parsed}


def _violation_payloads(violations: Any) -> list[dict]:
    """Serialize a list of :class:`~agentguard.engine.constraints.ConstraintViolation`
    for event payloads."""
    return [
        {"constraint": v.constraint, "argument": v.argument, "value": v.value, "detail": v.detail}
        for v in violations
    ]


def _evaluate_openai_tool_call_sync(
    g: GuardedOpenAI, llm_event_id: str, tool_name: str, tool_input: dict
) -> None:
    """Run the trust/policy/argument-constraint checks for one OpenAI function call.

    Mirrors :func:`agentguard.client._evaluate_tool_use_sync` for the OpenAI
    response shape. Factored out so the whole security-evaluation section for
    a single tool call can be wrapped in one try/except at the call site —
    see :func:`agentguard.client._handle_engine_error`.
    """
    # Trust check before tool execution.
    if g._trust_scorer.should_flag(g.session_id, tool_name):
        trust_score = g._trust_scorer.score(g.session_id)
        g._bus.emit(
            SecurityEvent(
                session_id=g.session_id,
                agent_id=g.agent_id,
                source="openai",
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
            source="openai",
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
    if policy_result.allowed:
        logger.info("[AgentGuard] tool_call: %s.%s -> INFO", g.agent_id, tool_name)
    else:
        logger.error(
            "[AgentGuard] policy_violation: %s.%s -> CRITICAL\n  reason: %s",
            g.agent_id,
            tool_name,
            policy_result.reason,
        )

    # Argument-level constraint check (path traversal, SSRF, etc.)
    violations = g._policy_engine.check_arguments(g.agent_id, tool_name, tool_input)
    if violations:
        violation_flags = [v.flag for v in violations]
        g._bus.emit(
            SecurityEvent(
                session_id=g.session_id,
                agent_id=g.agent_id,
                source="openai",
                event_type="policy_violation",
                severity="critical",
                payload=make_payload(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    violations=_violation_payloads(violations),
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


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


class GuardedChatCompletions:
    """Proxy for ``client.chat.completions`` that intercepts ``.create()`` calls.

    Mirrors :class:`agentguard.client.GuardedMessages`: pre-call injection scan
    and tool-definition policy checks, then post-call tool-call tracking (trust
    flags, policy checks, argument-constraint checks) for every function call
    the model returns.
    """

    def __init__(self, guard: GuardedOpenAI) -> None:
        self._guard = guard
        self._underlying = guard._client.chat.completions

    def create(self, **kwargs: Any) -> Any:
        """Intercept a chat.completions.create call to perform security checks."""
        g = self._guard
        _raise_if_killed(g, source="openai")
        messages: list[dict] = kwargs.get("messages", [])
        tools: list[dict] = kwargs.get("tools", [])
        model: str = kwargs.get("model", "unknown")

        llm_event_id = str(uuid.uuid4())

        # --- 1. Injection scan ---------------------------------------------
        try:
            injection_matches = g._injection_detector.scan_messages(messages)
        except Exception as exc:
            injection_matches = []
            _handle_engine_error(
                g, source="openai", phase="injection_scan", parent_event_id=llm_event_id, exc=exc
            )

        # --- 2. Tool policy check (tool-call name lives at tool["function"]["name"])
        policy_violations: list[str] = []
        try:
            for tool_def in tools:
                tool_name = _tool_def_name(tool_def)
                result = g._policy_engine.check(g.agent_id, tool_name)
                if not result.allowed:
                    policy_violations.append(f"{tool_name}: {result.reason} [{result.rule_name}]")
        except Exception as exc:
            policy_violations = []
            _handle_engine_error(
                g,
                source="openai",
                phase="tool_definition_policy_check",
                parent_event_id=llm_event_id,
                exc=exc,
            )

        # --- 3. Emit llm_call event -----------------------------------------
        g._bus.emit(
            SecurityEvent(
                event_id=llm_event_id,
                session_id=g.session_id,
                agent_id=g.agent_id,
                source="openai",
                event_type="llm_call",
                severity="info",
                payload=make_payload(
                    model=model,
                    message_count=len(messages),
                    tool_count=len(tools),
                ),
                metadata={"tool_names": [_tool_def_name(t) for t in tools]},
            )
        )
        logger.info(
            "[AgentGuard] llm_call: %s -> INFO (%d tool definition(s) checked)",
            g.agent_id,
            len(tools),
        )

        # --- 4. Injection events ---------------------------------------------
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
                    source="openai",
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

        # --- 5. Policy violation events ---------------------------------------
        if policy_violations:
            g._bus.emit(
                SecurityEvent(
                    session_id=g.session_id,
                    agent_id=g.agent_id,
                    source="openai",
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

        # --- 6. Actual API call ------------------------------------------------
        response = self._underlying.create(**kwargs)

        # --- 7. Emit tool_call events for each function call in the response ---
        for choice in getattr(response, "choices", None) or []:
            message = getattr(choice, "message", None)
            for tool_call in getattr(message, "tool_calls", None) or []:
                fn = getattr(tool_call, "function", None)
                tool_name = getattr(fn, "name", "unknown")
                tool_input = _parse_tool_arguments(getattr(fn, "arguments", "{}"))

                try:
                    _evaluate_openai_tool_call_sync(g, llm_event_id, tool_name, tool_input)
                except AgentGuardException:
                    raise
                except Exception as exc:
                    _handle_engine_error(
                        g,
                        source="openai",
                        phase="post_call_tool_evaluation",
                        parent_event_id=llm_event_id,
                        exc=exc,
                    )

        return response

    def __getattr__(self, name: str) -> Any:
        """Proxy any other attribute (e.g., .with_raw_response) unchanged."""
        return getattr(self._underlying, name)


class _GuardedChatNamespace:
    """Proxy for ``client.chat`` that exposes the intercepted ``completions`` namespace."""

    def __init__(self, guard: GuardedOpenAI) -> None:
        self._guard = guard
        self._underlying = guard._client.chat

    @property
    def completions(self) -> GuardedChatCompletions:
        """Intercepted chat-completions namespace."""
        return GuardedChatCompletions(self._guard)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._underlying, name)


class GuardedOpenAI:
    """Drop-in replacement for ``openai.OpenAI`` with security observability.

    Wraps the official OpenAI client, intercepting ``chat.completions.create``
    calls to perform injection detection, tool policy enforcement, trust
    scoring, and argument-constraint checking — the same pipeline as
    :class:`~agentguard.client.GuardedClient`, adapted to the OpenAI Chat
    Completions request/response shapes.

    Parameters
    ----------
    client:
        An instantiated ``openai.OpenAI`` client.
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

        import openai
        from agentguard import GuardedOpenAI

        client = GuardedOpenAI(
            openai.OpenAI(),
            agent_id="researcher",
            policy_path="policy.yaml",
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Summarize this document..."}],
            tools=[{"type": "function", "function": {"name": "read_file", "parameters": {...}}}],
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

        self._audit_logger: AuditLogger | None = None
        if audit_log is not None:
            self._audit_logger = AuditLogger(path=audit_log)
            self._bus.subscribe(self._audit_logger)
            logger.info("[AgentGuard] Audit log -> %s", audit_log)

        self._bus.emit(
            SecurityEvent(
                session_id=self.session_id,
                agent_id=self.agent_id,
                source="openai",
                event_type="session_start",
                severity="info",
                payload=make_payload(mode=mode, policy_path=policy_path or "none"),
            )
        )
        logger.info(
            "[AgentGuard] Session %s started (agent=%s, mode=%s)", self.session_id, agent_id, mode
        )

    @property
    def chat(self) -> _GuardedChatNamespace:
        """Intercepted chat namespace — ``chat.completions.create`` is checked."""
        return _GuardedChatNamespace(self)

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
                source="openai",
                event_type="session_end",
                severity="info",
                payload=make_payload(trust_score=self.trust_score()),
            )
        )

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes directly to the underlying client."""
        return getattr(self._client, name)


async def _evaluate_openai_tool_call_async(
    g: AsyncGuardedOpenAI, llm_event_id: str, tool_name: str, tool_input: dict
) -> None:
    """Run the trust/policy/argument-constraint checks for one OpenAI function call.

    Async mirror of :func:`_evaluate_openai_tool_call_sync` — see its docstring
    for the rationale behind factoring this out.
    """
    # Trust check before tool execution.
    if g._trust_scorer.should_flag(g.session_id, tool_name):
        trust_score = g._trust_scorer.score(g.session_id)
        await g._bus.emit_async(
            SecurityEvent(
                session_id=g.session_id,
                agent_id=g.agent_id,
                source="openai",
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

    # Policy check for the actual tool being called.
    policy_result = g._policy_engine.check(g.agent_id, tool_name)
    tool_severity = "info" if policy_result.allowed else "critical"
    tool_flags = [] if policy_result.allowed else [policy_result.rule_name]

    await g._bus.emit_async(
        SecurityEvent(
            session_id=g.session_id,
            agent_id=g.agent_id,
            source="openai",
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
    if policy_result.allowed:
        logger.info("[AgentGuard] tool_call: %s.%s -> INFO", g.agent_id, tool_name)
    else:
        logger.error(
            "[AgentGuard] policy_violation: %s.%s -> CRITICAL\n  reason: %s",
            g.agent_id,
            tool_name,
            policy_result.reason,
        )

    # Argument-level constraint check (path traversal, SSRF, etc.)
    violations = g._policy_engine.check_arguments(g.agent_id, tool_name, tool_input)
    if violations:
        violation_flags = [v.flag for v in violations]
        await g._bus.emit_async(
            SecurityEvent(
                session_id=g.session_id,
                agent_id=g.agent_id,
                source="openai",
                event_type="policy_violation",
                severity="critical",
                payload=make_payload(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    violations=_violation_payloads(violations),
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


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------


class AsyncGuardedChatCompletions:
    """Proxy for ``client.chat.completions`` that intercepts async ``.create()`` calls.

    Async mirror of :class:`GuardedChatCompletions` — see that class for the
    full pipeline description.
    """

    def __init__(self, guard: AsyncGuardedOpenAI) -> None:
        self._guard = guard
        self._underlying = guard._client.chat.completions

    async def create(self, **kwargs: Any) -> Any:
        """Intercept an async chat.completions.create call to perform security checks."""
        g = self._guard
        await _raise_if_killed_async(g, source="openai")
        messages: list[dict] = kwargs.get("messages", [])
        tools: list[dict] = kwargs.get("tools", [])
        model: str = kwargs.get("model", "unknown")

        llm_event_id = str(uuid.uuid4())

        # --- 1. Injection scan
        try:
            injection_matches = g._injection_detector.scan_messages(messages)
        except Exception as exc:
            injection_matches = []
            await _handle_engine_error_async(
                g, source="openai", phase="injection_scan", parent_event_id=llm_event_id, exc=exc
            )

        # --- 2. Tool policy check
        policy_violations: list[str] = []
        try:
            for tool_def in tools:
                tool_name = _tool_def_name(tool_def)
                result = g._policy_engine.check(g.agent_id, tool_name)
                if not result.allowed:
                    policy_violations.append(f"{tool_name}: {result.reason} [{result.rule_name}]")
        except Exception as exc:
            policy_violations = []
            await _handle_engine_error_async(
                g,
                source="openai",
                phase="tool_definition_policy_check",
                parent_event_id=llm_event_id,
                exc=exc,
            )

        # --- 3. Emit llm_call event
        await g._bus.emit_async(
            SecurityEvent(
                event_id=llm_event_id,
                session_id=g.session_id,
                agent_id=g.agent_id,
                source="openai",
                event_type="llm_call",
                severity="info",
                payload=make_payload(
                    model=model,
                    message_count=len(messages),
                    tool_count=len(tools),
                ),
                metadata={"tool_names": [_tool_def_name(t) for t in tools]},
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
                    source="openai",
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

        # --- 5. Policy violation events
        if policy_violations:
            await g._bus.emit_async(
                SecurityEvent(
                    session_id=g.session_id,
                    agent_id=g.agent_id,
                    source="openai",
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

        # --- 7. Emit tool_call events for each function call in the response
        for choice in getattr(response, "choices", None) or []:
            message = getattr(choice, "message", None)
            for tool_call in getattr(message, "tool_calls", None) or []:
                fn = getattr(tool_call, "function", None)
                tool_name = getattr(fn, "name", "unknown")
                tool_input = _parse_tool_arguments(getattr(fn, "arguments", "{}"))

                try:
                    await _evaluate_openai_tool_call_async(g, llm_event_id, tool_name, tool_input)
                except AgentGuardException:
                    raise
                except Exception as exc:
                    await _handle_engine_error_async(
                        g,
                        source="openai",
                        phase="post_call_tool_evaluation",
                        parent_event_id=llm_event_id,
                        exc=exc,
                    )

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._underlying, name)


class _AsyncGuardedChatNamespace:
    """Proxy for ``client.chat`` that exposes the intercepted async ``completions`` namespace."""

    def __init__(self, guard: AsyncGuardedOpenAI) -> None:
        self._guard = guard
        self._underlying = guard._client.chat

    @property
    def completions(self) -> AsyncGuardedChatCompletions:
        """Intercepted async chat-completions namespace."""
        return AsyncGuardedChatCompletions(self._guard)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._underlying, name)


class AsyncGuardedOpenAI:
    """Drop-in replacement for ``openai.AsyncOpenAI`` with security observability.

    Async mirror of :class:`GuardedOpenAI` — see that class for the full
    parameter and pipeline description. Every check runs without blocking the
    event loop (``await self._bus.emit_async(...)``, ``await
    self._approval_gate.request_async(...)``).

    Example
    -------
    ::

        import openai
        from agentguard import AsyncGuardedOpenAI

        client = AsyncGuardedOpenAI(openai.AsyncOpenAI(), agent_id="researcher")

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_input}],
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

        self._audit_logger: AuditLogger | None = None
        if audit_log is not None:
            self._audit_logger = AuditLogger(path=audit_log)
            self._bus.subscribe(self._audit_logger)
            logger.info("[AgentGuard] Audit log -> %s", audit_log)

        # Emit session_start synchronously so it's visible even before the first await.
        self._bus.emit(
            SecurityEvent(
                session_id=self.session_id,
                agent_id=self.agent_id,
                source="openai",
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
    def chat(self) -> _AsyncGuardedChatNamespace:
        """Intercepted async chat namespace — ``chat.completions.create`` is checked."""
        return _AsyncGuardedChatNamespace(self)

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
                source="openai",
                event_type="session_end",
                severity="info",
                payload=make_payload(trust_score=self.trust_score()),
            )
        )

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes directly to the underlying async client."""
        return getattr(self._client, name)
