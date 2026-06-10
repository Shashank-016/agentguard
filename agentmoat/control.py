"""Human-in-the-loop approval gating and the process-wide kill switch.

These two primitives back ``mode="interactive"``: instead of a binary
observe/enforce choice, risky actions can be routed to a human (or any
programmatic approver) for a real-time decision, and any session — or every
session at once — can be halted immediately via the kill switch.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------


@dataclass
class ApprovalRequest:
    """A request for a human (or programmatic approver) to approve or deny an action.

    Attributes
    ----------
    session_id:
        Identifier of the session the action belongs to.
    agent_id:
        Logical name of the agent attempting the action.
    action:
        Short identifier of the action requiring approval, e.g. ``"tool_call:write_file"``.
    reason:
        Human-readable explanation of why approval is required — typically the
        triggered flags or violation details.
    payload:
        The contextual data of the action (e.g. tool name and arguments).
    """

    session_id: str
    agent_id: str
    action: str
    reason: str
    payload: dict


ApprovalDecision = Literal["approve", "deny"]

# A handler takes an ApprovalRequest and returns a decision.
ApprovalHandler = Callable[[ApprovalRequest], ApprovalDecision]


def _cli_prompt_handler(request: ApprovalRequest) -> ApprovalDecision:
    """Default approval handler — prints the request and reads a y/N decision from stdin.

    Note: when used with the MCP stdio proxy, stdin/stdout are occupied by the
    JSON-RPC transport — register a different handler (Slack, web UI, queue)
    in that deployment rather than relying on this default.
    """
    import click

    click.echo("")
    click.echo(
        f"[AgentMoat] Approval required — session={request.session_id} agent={request.agent_id}"
    )
    click.echo(f"  action: {request.action}")
    click.echo(f"  reason: {request.reason}")
    click.echo(f"  payload: {request.payload}")
    return "approve" if click.confirm("Approve this action?", default=False) else "deny"


def auto_approve_handler(request: ApprovalRequest) -> ApprovalDecision:
    """Approval handler that always approves — for tests and CI."""
    return "approve"


def auto_deny_handler(request: ApprovalRequest) -> ApprovalDecision:
    """Approval handler that always denies — for tests and CI."""
    return "deny"


class ApprovalGate:
    """Routes risky actions to a human (or programmatic) approver.

    The default handler prompts on the CLI. Applications can register their
    own — a Slack interactive message, a web UI, an auto-approve policy for
    CI — by passing any callable matching :data:`ApprovalHandler`.

    Parameters
    ----------
    handler:
        Callable invoked with an :class:`ApprovalRequest`, returning
        ``"approve"`` or ``"deny"``. Defaults to a CLI y/N prompt.
    """

    def __init__(self, handler: ApprovalHandler | None = None) -> None:
        self._handler = handler or _cli_prompt_handler

    def request(self, req: ApprovalRequest) -> ApprovalDecision:
        """Synchronously route ``req`` to the handler and return its decision.

        Handler exceptions and invalid return values are treated as ``"deny"``
        — a misbehaving approver must never fail open.
        """
        try:
            decision = self._handler(req)
        except Exception:
            logger.exception("[AgentMoat] Approval handler raised — defaulting to deny")
            return "deny"
        if decision not in ("approve", "deny"):
            logger.warning(
                "[AgentMoat] Approval handler returned invalid decision %r — defaulting to deny",
                decision,
            )
            return "deny"
        return decision

    async def request_async(self, req: ApprovalRequest) -> ApprovalDecision:
        """Async-safe variant of :meth:`request` — runs the (sync) handler in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.request, req)


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


class KillSwitch:
    """Immediate halt control for agent sessions.

    A tripped session (or the global switch) causes the next intercepted
    action in that session to raise :class:`~agentmoat.client.AgentMoatKilled`
    (or, for the MCP proxy, return a JSON-RPC error) before any API call or
    tool execution takes place.

    Thread-safe — safe to trip from a request handler while a guarded client
    runs on another thread or in another async task.
    """

    def __init__(self) -> None:
        self._killed_sessions: set[str] = set()
        self._global = False
        self._lock = threading.Lock()

    def kill_session(self, session_id: str) -> None:
        """Trip the switch for a single session."""
        with self._lock:
            self._killed_sessions.add(session_id)
        logger.warning("[AgentMoat] Kill switch tripped for session '%s'", session_id)

    def kill_all(self) -> None:
        """Trip the global switch — halts every session in this process."""
        with self._lock:
            self._global = True
        logger.warning("[AgentMoat] Global kill switch tripped — all sessions halted")

    def revive_session(self, session_id: str) -> None:
        """Untrip a single session, restoring normal operation for it."""
        with self._lock:
            self._killed_sessions.discard(session_id)
        logger.info("[AgentMoat] Session '%s' revived", session_id)

    def reset(self) -> None:
        """Clear the global flag and every individually killed session."""
        with self._lock:
            self._killed_sessions.clear()
            self._global = False
        logger.info("[AgentMoat] Kill switch reset")

    def is_killed(self, session_id: str) -> bool:
        """Return True if ``session_id`` is halted — globally or individually."""
        with self._lock:
            return self._global or session_id in self._killed_sessions

    def status(self) -> dict:
        """Return the current kill state as a plain dict (global flag + killed session IDs)."""
        with self._lock:
            return {"global": self._global, "killed_sessions": sorted(self._killed_sessions)}


_default_kill_switch: KillSwitch | None = None
_default_kill_switch_lock = threading.Lock()


def get_default_kill_switch() -> KillSwitch:
    """Return the process-wide default :class:`KillSwitch` singleton.

    Guarded clients and the MCP interceptor share this instance unless an
    explicit ``kill_switch=`` is supplied, so a single ``kill_all()`` call
    halts every in-process session regardless of which client created it.
    """
    global _default_kill_switch
    if _default_kill_switch is None:
        with _default_kill_switch_lock:
            if _default_kill_switch is None:
                _default_kill_switch = KillSwitch()
    return _default_kill_switch
