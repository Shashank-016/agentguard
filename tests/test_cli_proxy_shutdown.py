"""Regression test: MCP proxy CLI drains the EventBus on shutdown.

``agentguard mcp proxy stdio|sse`` each construct their own ``EventBus`` for
the run's ``MCPInterceptor``. Before this fix, the ``finally`` block only
called ``upstream.stop()`` — the bus's background persistence worker (a
daemon thread, see ``EventBus._ensure_worker``) was never drained or stopped,
so events emitted right before the process exits could be lost. ``close()``
(which flushes pending persistence and joins the worker thread) must run on
every exit path, including exceptions from the server loop.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentguard.cli import _run_sse, _run_stdio


@pytest.mark.asyncio
async def test_run_stdio_closes_bus_on_clean_shutdown():
    fake_bus = MagicMock()
    fake_upstream = MagicMock()
    fake_upstream.start = AsyncMock()
    fake_upstream.stop = AsyncMock()
    fake_server = MagicMock()
    fake_server.run = AsyncMock()

    with (
        patch("agentguard.bus.EventBus", return_value=fake_bus),
        patch("agentguard.mcp.client.StdioUpstreamClient", return_value=fake_upstream),
        patch("agentguard.mcp.interceptor.MCPInterceptor"),
        patch("agentguard.mcp.proxy.MCPProxy"),
        patch("agentguard.mcp.server.StdioProxyServer", return_value=fake_server),
    ):
        await _run_stdio(
            upstream_cmd="echo hi",
            agent_id="test-agent",
            policy_path=None,
            mode="observe",
            session_id="test-session",
        )

    fake_upstream.stop.assert_awaited_once()
    fake_bus.close.assert_called_once()


@pytest.mark.asyncio
async def test_run_stdio_closes_bus_even_when_server_raises():
    fake_bus = MagicMock()
    fake_upstream = MagicMock()
    fake_upstream.start = AsyncMock()
    fake_upstream.stop = AsyncMock()
    fake_server = MagicMock()
    fake_server.run = AsyncMock(side_effect=RuntimeError("boom"))

    with (
        patch("agentguard.bus.EventBus", return_value=fake_bus),
        patch("agentguard.mcp.client.StdioUpstreamClient", return_value=fake_upstream),
        patch("agentguard.mcp.interceptor.MCPInterceptor"),
        patch("agentguard.mcp.proxy.MCPProxy"),
        patch("agentguard.mcp.server.StdioProxyServer", return_value=fake_server),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await _run_stdio(
                upstream_cmd="echo hi",
                agent_id="test-agent",
                policy_path=None,
                mode="observe",
                session_id="test-session",
            )

    fake_upstream.stop.assert_awaited_once()
    fake_bus.close.assert_called_once()


@pytest.mark.asyncio
async def test_run_sse_closes_bus_on_clean_shutdown():
    fake_bus = MagicMock()
    fake_upstream = MagicMock()
    fake_upstream.start = AsyncMock()
    fake_upstream.stop = AsyncMock()
    fake_server = MagicMock()
    fake_server.run = AsyncMock()

    with (
        patch("agentguard.bus.EventBus", return_value=fake_bus),
        patch("agentguard.mcp.client.SSEUpstreamClient", return_value=fake_upstream),
        patch("agentguard.mcp.interceptor.MCPInterceptor"),
        patch("agentguard.mcp.proxy.MCPProxy"),
        patch("agentguard.mcp.server.SSEProxyServer", return_value=fake_server),
    ):
        await _run_sse(
            upstream_url="http://localhost:9999",
            port=8899,
            agent_id="test-agent",
            mode="observe",
            session_id="test-session",
            policy_path=None,
        )

    fake_upstream.stop.assert_awaited_once()
    fake_bus.close.assert_called_once()
