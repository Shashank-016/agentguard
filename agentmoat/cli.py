"""AgentMoat command-line interface."""

from __future__ import annotations

import asyncio
import logging
import uuid

import click

logger = logging.getLogger(__name__)


@click.group()
@click.option("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
def cli(log_level: str) -> None:
    """AgentMoat — security observability for AI agents."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@cli.group()
def audit() -> None:
    """Inspect and verify durable JSONL audit logs."""


@audit.command(name="verify")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def audit_verify(path: str) -> None:
    """Verify the tamper-evident hash chain of an audit log file.

    Exits 0 and prints a confirmation if the chain is intact, or exits 1 and
    reports the line where the chain first breaks.

    Example::

        agentmoat audit verify agentmoat_audit.jsonl
    """
    from .audit import AuditLogger

    log = AuditLogger(path=path, rotate_mb=0)
    result = log.verify()
    if result.valid:
        click.echo(f"✓ Chain intact — {result.records_checked:,} records verified")
        raise SystemExit(0)
    click.echo(
        f"✗ Chain broken at line {result.first_broken_line} — "
        "record was modified or a prior line was deleted"
    )
    click.echo(f"  {result.detail}")
    raise SystemExit(1)


@audit.command(name="tail")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("-n", "count", default=50, type=int, help="Number of records to print")
def audit_tail(path: str, count: int) -> None:
    """Print the last N records from an audit log, oldest first.

    Example::

        agentmoat audit tail agentmoat_audit.jsonl -n 50
    """
    from .audit import AuditLogger

    log = AuditLogger(path=path, rotate_mb=0)
    for event in reversed(log.tail(count)):
        click.echo(event.model_dump_json())


@audit.command(name="stats")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def audit_stats(path: str) -> None:
    """Print record counts by event_type and severity for an audit log.

    Example::

        agentmoat audit stats agentmoat_audit.jsonl
    """
    import json as _json
    from collections import Counter
    from pathlib import Path

    type_counts: Counter = Counter()
    severity_counts: Counter = Counter()
    total = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        total += 1
        type_counts[data.get("event_type", "unknown")] += 1
        severity_counts[data.get("severity", "unknown")] += 1

    click.echo(f"Total records: {total}")
    click.echo("By event_type:")
    for event_type, count in type_counts.most_common():
        click.echo(f"  {event_type}: {count}")
    click.echo("By severity:")
    for severity, count in severity_counts.most_common():
        click.echo(f"  {severity}: {count}")


@cli.group()
def mcp() -> None:
    """MCP (Model Context Protocol) proxy commands."""


@mcp.group()
def proxy() -> None:
    """Run AgentMoat as a transparent MCP proxy."""


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
    type=click.Choice(["observe", "enforce", "interactive"]),
    help="observe (log only) or enforce (block violations)",
)
@click.option("--session-id", default=None, help="Session ID (auto-generated if omitted)")
def stdio(
    upstream_cmd: str,
    agent_id: str,
    policy_path: str | None,
    mode: str,
    session_id: str | None,
) -> None:
    """Run an MCP proxy over stdio (wrap a local MCP server subprocess).

    Example::

        agentmoat mcp proxy stdio \\
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
    type=click.Choice(["observe", "enforce", "interactive"]),
    help="observe (log only) or enforce (block violations)",
)
@click.option("--session-id", default=None, help="Session ID (auto-generated if omitted)")
@click.option("--policy", "policy_path", default=None, help="Path to YAML policy file")
def sse(
    upstream_url: str,
    port: int,
    agent_id: str,
    mode: str,
    session_id: str | None,
    policy_path: str | None,
) -> None:
    """Run an MCP proxy over HTTP/SSE (wrap a remote MCP server).

    Example::

        agentmoat mcp proxy sse \\
            --upstream-url http://mcp-server:3000 \\
            --port 8899 \\
            --agent-id researcher \\
            --mode enforce
    """
    asyncio.run(_run_sse(upstream_url, port, agent_id, mode, session_id, policy_path))


async def _run_stdio(
    upstream_cmd: str,
    agent_id: str,
    policy_path: str | None,
    mode: str,
    session_id: str | None,
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
        # Drain the persistence worker so events emitted during this run
        # reach the store before the process exits — see EventBus.close().
        await asyncio.to_thread(bus.close)


async def _run_sse(
    upstream_url: str,
    port: int,
    agent_id: str,
    mode: str,
    session_id: str | None,
    policy_path: str | None,
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
        # Drain the persistence worker so events emitted during this run
        # reach the store before the process exits — see EventBus.close().
        await asyncio.to_thread(bus.close)
