"""MCPInterceptor — security checks for MCP tool calls."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from ..bus import EventBus
from ..engine.injection import InjectionDetector
from ..engine.policy import ToolPolicyEngine
from ..engine.trust import TrustScorer
from ..events import SecurityEvent
from .models import (
    AGENTGUARD_INJECTION_DETECTED,
    AGENTGUARD_POLICY_VIOLATION,
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
        All :class:`~agentguard.events.SecurityEvent` objects generated during this check.
    block_reason:
        Human-readable reason for blocking (None if allowed).
    block_code:
        JSON-RPC error code to return when blocked (None if allowed).
    """

    allowed: bool
    events: list[SecurityEvent] = field(default_factory=list)
    block_reason: Optional[str] = None
    block_code: Optional[int] = None


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

    Receives an :class:`~agentguard.mcp.models.MCPRequest`, runs all relevant
    security checks (injection, policy, trust), emits events via the bus, and
    returns an :class:`InterceptResult`.

    Parameters
    ----------
    agent_id:
        Logical name of the agent being proxied.
    session_id:
        Session identifier for grouping events.
    bus:
        :class:`~agentguard.bus.EventBus` to emit events on.
    policy_path:
        Optional path to a YAML policy file.
    mode:
        ``"observe"`` — log but never block.
        ``"enforce"`` — block on violations.
    """

    def __init__(
        self,
        agent_id: str,
        session_id: str,
        bus: EventBus,
        policy_path: Optional[str] = None,
        mode: Literal["observe", "enforce"] = "observe",
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.bus = bus
        self.mode = mode
        self._injection = InjectionDetector()
        self._policy = ToolPolicyEngine(policy_path)
        self._trust = TrustScorer()

    def intercept(self, request: MCPRequest) -> InterceptResult:
        """Run all security checks on an incoming MCP request.

        Only ``tools/call`` requests are deeply inspected. Other MCP lifecycle
        methods are logged at info level and passed through without blocking.

        Returns
        -------
        InterceptResult
            Contains ``allowed`` flag, emitted events, and optional block details.
        """
        events: list[SecurityEvent] = []
        allowed = True
        block_reason: Optional[str] = None
        block_code: Optional[int] = None

        if request.method == "tools/call":
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
                    "[AgentGuard/MCP] injection_detected: %s → CRITICAL  tool=%s  flags=%s",
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
                    block_code = AGENTGUARD_INJECTION_DETECTED

            # 2. Policy check
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
                    "[AgentGuard/MCP] policy_violation: %s → CRITICAL  tool=%s  reason=%s",
                    self.agent_id,
                    params.name,
                    policy_result.reason,
                )
                if self.mode == "enforce":
                    allowed = False
                    block_reason = (
                        f"Tool '{params.name}' denied by policy: {policy_result.reason}"
                    )
                    block_code = AGENTGUARD_POLICY_VIOLATION

            # 3. Trust check (warning only — never blocks by itself)
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
                    "[AgentGuard/MCP] trust_flag: %s  tool=%s  score=%.2f",
                    self.agent_id,
                    params.name,
                    score,
                )

            # 4. Baseline tool_call event (emitted when the call is allowed or in observe mode)
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
