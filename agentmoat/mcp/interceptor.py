"""MCPInterceptor — security checks for MCP tool calls."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from ..bus import EventBus
from ..control import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    KillSwitch,
    get_default_kill_switch,
)
from ..engine.injection import InjectionDetector
from ..engine.policy import ToolPolicyEngine
from ..engine.trust import TrustScorer
from ..events import SecurityEvent
from .models import (
    AGENTMOAT_ENGINE_ERROR,
    AGENTMOAT_INJECTION_DETECTED,
    AGENTMOAT_POLICY_VIOLATION,
    AGENTMOAT_SESSION_KILLED,
    AGENTMOAT_TRUST_VIOLATION,
    MCPRequest,
    MCPToolCallParams,
)

logger = logging.getLogger(__name__)


@dataclass
class InterceptResult:
    """Result of an MCP security interception check.

    Attributes
    ----------
    allowed:
        Whether the request should be forwarded to the upstream server.
    events:
        All :class:`~agentmoat.events.SecurityEvent` objects generated during this check.
    block_reason:
        Human-readable reason for blocking (None if allowed).
    block_code:
        JSON-RPC error code to return when blocked (None if allowed).
    """

    allowed: bool
    events: list[SecurityEvent] = field(default_factory=list)
    block_reason: str | None = None
    block_code: int | None = None


_PASSTHROUGH_METHODS = frozenset(
    {
        "initialize",
        "initialized",
        "tools/list",
        "resources/list",
        "resources/read",
        "prompts/list",
        "prompts/get",
        "ping",
    }
)


class MCPInterceptor:
    """Security engine adapter for MCP tool calls.

    Receives an :class:`~agentmoat.mcp.models.MCPRequest`, runs all relevant
    security checks (injection, policy, trust), emits events via the bus, and
    returns an :class:`InterceptResult`.

    Parameters
    ----------
    agent_id:
        Logical name of the agent being proxied.
    session_id:
        Session identifier for grouping events.
    bus:
        :class:`~agentmoat.bus.EventBus` to emit events on.
    policy_path:
        Optional path to a YAML policy file.
    mode:
        ``"observe"`` — log but never block.
        ``"enforce"`` — block on violations.
        ``"interactive"`` — route violations through ``approval_gate`` for a
        human (or programmatic) decision; a "deny" blocks the request.
    approval_gate:
        :class:`~agentmoat.control.ApprovalGate` used in ``mode="interactive"``.
        If ``None``, a default gate (CLI y/N prompt) is created when needed.
    kill_switch:
        Shared :class:`~agentmoat.control.KillSwitch`. Defaults to the
        process-wide singleton from :func:`~agentmoat.control.get_default_kill_switch`.
    """

    def __init__(
        self,
        agent_id: str,
        session_id: str,
        bus: EventBus,
        policy_path: str | None = None,
        mode: Literal["observe", "enforce", "interactive"] = "observe",
        approval_gate: ApprovalGate | None = None,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.bus = bus
        self.mode = mode
        self._injection = InjectionDetector()
        self._policy = ToolPolicyEngine(policy_path)
        self._trust = TrustScorer()
        self._approval_gate = approval_gate or (ApprovalGate() if mode == "interactive" else None)
        self._kill_switch = kill_switch if kill_switch is not None else get_default_kill_switch()

    def _request_approval(
        self,
        *,
        events: list[SecurityEvent],
        action: str,
        reason: str,
        payload: dict,
        parent_event_id: str | None = None,
    ) -> ApprovalDecision:
        """Append ``approval_required``/``approval_granted``/``approval_denied`` events and
        return the decision.

        Mirrors the SDK wrappers' interactive-mode flow: routes the violation
        through ``self._approval_gate`` and records the full round trip. Events
        are appended to ``events`` (emitted by the caller, like every other
        event this method produces) rather than emitted immediately.
        """
        events.append(
            SecurityEvent(
                session_id=self.session_id,
                agent_id=self.agent_id,
                source="mcp",
                event_type="approval_required",
                severity="warning",
                payload={"action": action, "reason": reason, "context": payload},
                parent_event_id=parent_event_id,
            )
        )
        decision = self._approval_gate.request(
            ApprovalRequest(
                session_id=self.session_id,
                agent_id=self.agent_id,
                action=action,
                reason=reason,
                payload=payload,
            )
        )
        events.append(
            SecurityEvent(
                session_id=self.session_id,
                agent_id=self.agent_id,
                source="mcp",
                event_type="approval_granted" if decision == "approve" else "approval_denied",
                severity="info" if decision == "approve" else "critical",
                payload={"action": action, "decision": decision},
                parent_event_id=parent_event_id,
            )
        )
        return decision

    def intercept(self, request: MCPRequest) -> InterceptResult:
        """Run all security checks on an incoming MCP request.

        Only ``tools/call`` requests are deeply inspected. Other MCP lifecycle
        methods are logged at info level and passed through without blocking.

        If this session (or the global switch) has been tripped via
        :class:`~agentmoat.control.KillSwitch`, the request is blocked
        immediately without running any checks or contacting the upstream server.

        Returns
        -------
        InterceptResult
            Contains ``allowed`` flag, emitted events, and optional block details.
        """
        if self._kill_switch.is_killed(self.session_id):
            event = SecurityEvent(
                session_id=self.session_id,
                agent_id=self.agent_id,
                source="mcp",
                event_type="session_end",
                severity="critical",
                payload={"reason": "kill_switch_tripped", "method": request.method},
                flags=["kill:tripped"],
            )
            self.bus.emit(event)
            logger.critical("[AgentMoat/MCP] Session %s halted by kill switch", self.session_id)
            return InterceptResult(
                allowed=False,
                events=[event],
                block_reason=(
                    f"Session '{self.session_id}' has been killed via KillSwitch — "
                    "request blocked."
                ),
                block_code=AGENTMOAT_SESSION_KILLED,
            )

        events: list[SecurityEvent] = []
        allowed = True
        block_reason: str | None = None
        block_code: int | None = None

        if request.method == "tools/call":
            try:
                params = MCPToolCallParams(**(request.params or {}))

                # 1. Injection scan on tool argument values
                arg_text = " ".join(str(v) for v in params.arguments.values())
                matches = self._injection.scan(arg_text)
                if matches:
                    flags = [m.flag for m in matches]
                    event = SecurityEvent(
                        session_id=self.session_id,
                        agent_id=self.agent_id,
                        source="mcp",
                        event_type="injection_detected",
                        severity="critical",
                        payload={
                            "tool": params.name,
                            "arguments": params.arguments,
                            "patterns": [m.pattern_name for m in matches],
                        },
                        flags=flags,
                    )
                    events.append(event)
                    logger.warning(
                        "[AgentMoat/MCP] injection_detected: %s → CRITICAL  tool=%s  flags=%s",
                        self.agent_id,
                        params.name,
                        flags,
                    )
                    if self.mode == "enforce":
                        allowed = False
                        block_reason = (
                            f"Injection detected in tool arguments: "
                            f"{[m.pattern_name for m in matches]}"
                        )
                        block_code = AGENTMOAT_INJECTION_DETECTED
                    elif self.mode == "interactive":
                        decision = self._request_approval(
                            events=events,
                            action=f"tool_call:{params.name}",
                            reason=(
                                "Injection detected in tool arguments: "
                                f"{[m.pattern_name for m in matches]}"
                            ),
                            payload={
                                "tool": params.name,
                                "arguments": params.arguments,
                                "flags": flags,
                            },
                        )
                        if decision == "deny":
                            allowed = False
                            block_reason = (
                                f"Injection detected in tool arguments — denied by approver: "
                                f"{[m.pattern_name for m in matches]}"
                            )
                            block_code = AGENTMOAT_INJECTION_DETECTED

                # 2. Argument-level constraint check (path traversal, SSRF, shell metachars, ...)
                violations = self._policy.check_arguments(
                    self.agent_id, params.name, params.arguments
                )
                if violations:
                    violation_flags = [v.flag for v in violations]
                    event = SecurityEvent(
                        session_id=self.session_id,
                        agent_id=self.agent_id,
                        source="mcp",
                        event_type="policy_violation",
                        severity="critical",
                        payload={
                            "tool": params.name,
                            "arguments": params.arguments,
                            "violations": [
                                {
                                    "constraint": v.constraint,
                                    "argument": v.argument,
                                    "value": v.value,
                                    "detail": v.detail,
                                }
                                for v in violations
                            ],
                        },
                        flags=violation_flags,
                    )
                    events.append(event)
                    logger.warning(
                        "[AgentMoat/MCP] policy_violation: %s → CRITICAL  tool=%s  constraints=%s",
                        self.agent_id,
                        params.name,
                        violation_flags,
                    )
                    if self.mode == "enforce":
                        allowed = False
                        block_reason = (
                            f"Argument constraint violation for tool '{params.name}': "
                            f"{[v.detail for v in violations]}"
                        )
                        block_code = AGENTMOAT_POLICY_VIOLATION
                    elif self.mode == "interactive":
                        decision = self._request_approval(
                            events=events,
                            action=f"tool_call:{params.name}",
                            reason=(
                                f"Argument constraint violation for tool '{params.name}': "
                                f"{[v.detail for v in violations]}"
                            ),
                            payload={
                                "tool": params.name,
                                "arguments": params.arguments,
                                "flags": violation_flags,
                            },
                        )
                        if decision == "deny":
                            allowed = False
                            block_reason = (
                                f"Argument constraint violation for tool '{params.name}' — "
                                f"denied by approver: {[v.detail for v in violations]}"
                            )
                            block_code = AGENTMOAT_POLICY_VIOLATION

                # 3. Policy check
                policy_result = self._policy.check(self.agent_id, params.name)
                if not policy_result.allowed:
                    event = SecurityEvent(
                        session_id=self.session_id,
                        agent_id=self.agent_id,
                        source="mcp",
                        event_type="policy_violation",
                        severity="critical",
                        payload={
                            "tool": params.name,
                            "reason": policy_result.reason,
                        },
                        flags=[f"policy:{policy_result.rule_name}"],
                    )
                    events.append(event)
                    logger.warning(
                        "[AgentMoat/MCP] policy_violation: %s → CRITICAL  tool=%s  reason=%s",
                        self.agent_id,
                        params.name,
                        policy_result.reason,
                    )
                    if self.mode == "enforce":
                        allowed = False
                        block_reason = (
                            f"Tool '{params.name}' denied by policy: {policy_result.reason}"
                        )
                        block_code = AGENTMOAT_POLICY_VIOLATION
                    elif self.mode == "interactive":
                        decision = self._request_approval(
                            events=events,
                            action=f"tool_call:{params.name}",
                            reason=f"Tool '{params.name}' denied by policy: {policy_result.reason}",
                            payload={"tool": params.name, "reason": policy_result.reason},
                        )
                        if decision == "deny":
                            allowed = False
                            block_reason = (
                                f"Tool '{params.name}' denied by policy and approver: "
                                f"{policy_result.reason}"
                            )
                            block_code = AGENTMOAT_POLICY_VIOLATION

                # 4. Trust check (warning only — never hard-blocks in enforce mode;
                #    interactive mode still routes it through the approval gate)
                if self._trust.should_flag(self.session_id, params.name):
                    score = self._trust.score(self.session_id)
                    event = SecurityEvent(
                        session_id=self.session_id,
                        agent_id=self.agent_id,
                        source="mcp",
                        event_type="trust_flag",
                        severity="warning",
                        payload={
                            "tool": params.name,
                            "trust_score": score,
                        },
                        flags=[f"trust:score_{score:.2f}"],
                    )
                    events.append(event)
                    logger.warning(
                        "[AgentMoat/MCP] trust_flag: %s  tool=%s  score=%.2f",
                        self.agent_id,
                        params.name,
                        score,
                    )
                    if self.mode == "interactive":
                        decision = self._request_approval(
                            events=events,
                            action=f"tool_call:{params.name}",
                            reason=(
                                f"Low-trust session (score={score:.2f}) "
                                f"attempting {params.name}"
                            ),
                            payload={"tool": params.name, "trust_score": score},
                        )
                        if decision == "deny":
                            allowed = False
                            block_reason = (
                                f"Tool '{params.name}' denied by approver due to low trust "
                                f"score ({score:.2f})"
                            )
                            block_code = AGENTMOAT_TRUST_VIOLATION

                # 5. Baseline tool_call event (emitted when the call is allowed or in observe mode)
                if allowed or self.mode == "observe":
                    events.append(
                        SecurityEvent(
                            session_id=self.session_id,
                            agent_id=self.agent_id,
                            source="mcp",
                            event_type="tool_call",
                            severity="info",
                            payload={
                                "tool": params.name,
                                "arguments": params.arguments,
                            },
                        )
                    )
            except Exception as exc:
                logger.exception(
                    "[AgentMoat/MCP] Internal security engine error while evaluating tool call"
                )
                events.append(
                    SecurityEvent(
                        session_id=self.session_id,
                        agent_id=self.agent_id,
                        source="mcp",
                        event_type="engine_error",
                        severity="critical",
                        payload={
                            "method": request.method,
                            "phase": "tool_call_security_check",
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                        flags=["engine:internal_error"],
                    )
                )
                if self.mode in ("enforce", "interactive"):
                    if allowed:
                        allowed = False
                        block_reason = (
                            f"Internal security engine error while evaluating tool call "
                            f"— failing closed (mode={self.mode}). {exc}"
                        )
                        block_code = AGENTMOAT_ENGINE_ERROR
                else:
                    # observe mode: fail open, but the engine_error event above
                    # makes the failure loud and visible in the audit trail.
                    allowed = True

        elif request.method in _PASSTHROUGH_METHODS:
            events.append(
                SecurityEvent(
                    session_id=self.session_id,
                    agent_id=self.agent_id,
                    source="mcp",
                    event_type="node_traversal",
                    severity="info",
                    payload={"method": request.method},
                )
            )

        # Emit all events synchronously
        for event in events:
            self.bus.emit(event)

        return InterceptResult(
            allowed=allowed,
            events=events,
            block_reason=block_reason,
            block_code=block_code,
        )
