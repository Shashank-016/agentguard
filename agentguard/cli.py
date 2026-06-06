"""AgentGuard command-line interface."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import click

logger = logging.getLogger(__name__)


@click.group()
@click.option("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
def cli(log_level: str) -> None:
    """AgentGuard — security observability for AI agents."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@cli.group()
def mcp() -> None:
    """MCP (Model Context Protocol) proxy commands."""


@mcp.group()
def proxy() -> None:
    """Run AgentGuard as a transparent MCP proxy."""


@proxy.command()
@click.option(
    "--upstream-cmd",
    required=True,
    help="Command to launch the upstream MCP server (e.g. 'npx -y @mcp/server-filesystem /tmp')",
)
@click.option("--agent-id", default="default", help="Logical agent identifier")
@click.option("--policy", "policy_path", default=None, help="Path to YAML policy file")
@click.option(
    "--mode",
    default="observe",
    type=click.Choice(["observe", "enforce"]),
    help="observe (log only) or enforce (block violations)",
)
@click.option("--session-id", default=None, help="Session ID (auto-generated if omitted)")
def stdio(
    upstream_cmd: str,
    agent_id: str,
    policy_path: Optional[str],
    mode: str,
    session_id: Optional[str],
) -> None:
    """Run an MCP proxy over stdio (wrap a local MCP server subprocess).

    Example::

        agentguard mcp proxy stdio \\
            --upstream-cmd "npx -y @modelcontextprotocol/server-filesystem /tmp" \\
            --agent-id researcher \\
            --policy policy.yaml \\
            --mode enforce
    """
    asyncio.run(_run_stdio(upstream_cmd, agent_id, policy_path, mode, session_id))


@proxy.command()
@click.option("--upstream-url", required=True, help="Base URL of the remote MCP server")
@click.option("--port", default=8899, type=int, help="Port for the local HTTP proxy server")
@click.option("--agent-id", default="default", help="Logical agent identifier")
@click.option(
    "--mode",
    default="observe",
    type=click.Choice(["observe", "enforce"]),
    help="observe (log only) or enforce (block violations)",
)
@click.option("--session-id", default=None, help="Session ID (auto-generated if omitted)")
@click.option("--policy", "policy_path", default=None, help="Path to YAML policy file")
def sse(
    upstream_url: str,
    port: int,
    agent_id: str,
    mode: str,
    session_id: Optional[str],
    policy_path: Optional[str],
) -> None:
    """Run an MCP proxy over HTTP/SSE (wrap a remote MCP server).

    Example::

        agentguard mcp proxy sse \\
            --upstream-url http://mcp-server:3000 \\
            --port 8899 \\
            --agent-id researcher \\
            --mode enforce
    """
    asyncio.run(_run_sse(upstream_url, port, agent_id, mode, session_id, policy_path))


async def _run_stdio(
    upstream_cmd: str,
    agent_id: str,
    policy_path: Optional[str],
    mode: str,
    session_id: Optional[str],
) -> None:
    """Wire up and run a stdio MCP proxy."""
    from .bus import EventBus
    from .mcp.client import StdioUpstreamClient
    from .mcp.interceptor import MCPInterceptor
    from .mcp.proxy import MCPProxy
    from .mcp.server import StdioProxyServer

    sid = session_id or str(uuid.uuid4())
    bus = EventBus()
    upstream = StdioUpstreamClient(command=upstream_cmd.split())
    await upstream.start()

    interceptor = MCPInterceptor(
        agent_id=agent_id,
        session_id=sid,
        bus=bus,
        policy_path=policy_path,
        mode=mode,
    )
    proxy = MCPProxy(upstream=upstream, interceptor=interceptor, mode=mode)
    server = StdioProxyServer(proxy=proxy)

    try:
        await server.run()
    finally:
        await upstream.stop()


async def _run_sse(
    upstream_url: str,
    port: int,
    agent_id: str,
    mode: str,
    session_id: Optional[str],
    policy_path: Optional[str],
) -> None:
    """Wire up and run an SSE MCP proxy."""
    from .bus import EventBus
    from .mcp.client import SSEUpstreamClient
    from .mcp.interceptor import MCPInterceptor
    from .mcp.proxy import MCPProxy
    from .mcp.server import SSEProxyServer

    sid = session_id or str(uuid.uuid4())
    bus = EventBus()
    upstream = SSEUpstreamClient(base_url=upstream_url)
    await upstream.start()

    interceptor = MCPInterceptor(
        agent_id=agent_id,
        session_id=sid,
        bus=bus,
        policy_path=policy_path,
        mode=mode,
    )
    proxy = MCPProxy(upstream=upstream, interceptor=interceptor, mode=mode)
    server = SSEProxyServer(proxy=proxy, port=port)

    try:
        await server.run()
    finally:
        await upstream.stop()
