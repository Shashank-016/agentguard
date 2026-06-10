"""MCPProxy — transparent MCP proxy that intercepts and validates tool calls."""

from __future__ import annotations

import logging
from typing import Any, Literal

from .client import SSEUpstreamClient, StdioUpstreamClient
from .interceptor import MCPInterceptor
from .models import INVALID_REQUEST, MCPRequest, MCPResponse

logger = logging.getLogger(__name__)


class MCPProxy:
    """Transparent MCP proxy: intercepts requests, enforces policy, forwards to upstream.

    In ``observe`` mode, all requests are forwarded and security events are emitted
    without blocking. In ``enforce`` and ``interactive`` modes, blocked requests
    receive a JSON-RPC error response and are never forwarded to the upstream server
    — the interceptor itself is responsible for deciding what gets blocked (hard
    policy in enforce, human/approver decisions in interactive, kill-switch state in
    either), this proxy simply honors ``result.allowed``.

    Parameters
    ----------
    upstream:
        A connected :class:`~agentmoat.mcp.client.StdioUpstreamClient` or
        :class:`~agentmoat.mcp.client.SSEUpstreamClient`.
    interceptor:
        The :class:`~agentmoat.mcp.interceptor.MCPInterceptor` that runs security checks.
    mode:
        ``"observe"``, ``"enforce"``, or ``"interactive"``.
    """

    def __init__(
        self,
        upstream: StdioUpstreamClient | SSEUpstreamClient,
        interceptor: MCPInterceptor,
        mode: Literal["observe", "enforce", "interactive"] = "observe",
    ) -> None:
        self._upstream = upstream
        self._interceptor = interceptor
        self.mode = mode

    async def handle(self, raw_request: dict[str, Any]) -> dict[str, Any]:
        """Process a raw JSON-RPC dict through the security layer and upstream.

        Parameters
        ----------
        raw_request:
            A decoded JSON-RPC 2.0 request dictionary.

        Returns
        -------
        dict
            A JSON-RPC 2.0 response dictionary.
        """
        try:
            request = MCPRequest(**raw_request)
        except Exception as exc:
            logger.warning("[AgentMoat/MCP] Invalid request: %s", exc)
            return MCPResponse(
                id=raw_request.get("id"),
                error={"code": INVALID_REQUEST, "message": str(exc)},
            ).model_dump()

        result = self._interceptor.intercept(request)

        if not result.allowed:
            logger.info(
                "[AgentMoat/MCP] Request blocked: method=%s  reason=%s",
                request.method,
                result.block_reason,
            )
            flagged_lists = [e.flags for e in result.events if e.flags]
            return MCPResponse(
                id=request.id,
                error={
                    "code": result.block_code,
                    "message": result.block_reason,
                    "data": {"flags": flagged_lists},
                },
            ).model_dump()

        upstream_response = await self._upstream.send(request)
        return upstream_response.model_dump()
