"""MCP proxy servers — expose AgentGuard as an MCP-compatible stdio or HTTP server."""

from __future__ import annotations

import asyncio
import json
import logging
import sys

from .models import PARSE_ERROR
from .proxy import MCPProxy

logger = logging.getLogger(__name__)


class StdioProxyServer:
    """Runs the AgentGuard MCP proxy on stdio.

    The agent treats this process as an MCP server. All JSON-RPC messages
    arrive on stdin and responses are written to stdout — the standard MCP
    stdio transport.

    Usage via CLI::

        agentguard mcp proxy stdio \\
            --upstream-cmd "npx -y @modelcontextprotocol/server-filesystem /tmp" \\
            --agent-id researcher \\
            --policy policy.yaml \\
            --mode enforce

    Parameters
    ----------
    proxy:
        A configured :class:`~agentguard.mcp.proxy.MCPProxy`.
    """

    def __init__(self, proxy: MCPProxy) -> None:
        self._proxy = proxy

    async def run(self) -> None:
        """Read JSON-RPC messages from stdin, handle via proxy, write responses to stdout."""
        logger.info("[AgentGuard/MCP] StdioProxyServer started")
        loop = asyncio.get_event_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except (EOFError, KeyboardInterrupt):
                break

            if not line:
                break

            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
                response = await self._proxy.handle(raw)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": PARSE_ERROR, "message": "Parse error"},
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()
            except Exception:
                logger.exception("[AgentGuard/MCP] Error handling stdin message")

        logger.info("[AgentGuard/MCP] StdioProxyServer stopped")


class SSEProxyServer:
    """Runs the AgentGuard MCP proxy as an HTTP server with SSE transport.

    Agents connect to ``http://localhost:<port>/mcp`` (POST) or ``/sse`` (GET)
    instead of the real MCP server URL. All requests are intercepted and
    forwarded through :class:`~agentguard.mcp.proxy.MCPProxy`.

    Usage via CLI::

        agentguard mcp proxy sse \\
            --upstream-url http://localhost:3000 \\
            --port 8899 \\
            --agent-id researcher \\
            --mode enforce

    Parameters
    ----------
    proxy:
        A configured :class:`~agentguard.mcp.proxy.MCPProxy`.
    port:
        TCP port to listen on (default ``8899``).
    """

    def __init__(self, proxy: MCPProxy, port: int = 8899) -> None:
        self._proxy = proxy
        self.port = port
        self._app: object | None = None

    def _build_app(self) -> object:
        """Build and return the FastAPI application."""
        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import JSONResponse, StreamingResponse
        except ImportError as exc:
            raise RuntimeError(
                "fastapi is required for SSEProxyServer. Install with: pip install fastapi"
            ) from exc

        app = FastAPI(title="AgentGuard MCP Proxy")

        @app.get("/health")
        async def health() -> dict:
            return {"status": "ok", "service": "agentguard-mcp-proxy"}

        @app.post("/mcp")
        async def mcp_post(request: Request) -> JSONResponse:
            body = await request.json()
            response = await self._proxy.handle(body)
            return JSONResponse(content=response)

        @app.get("/sse")
        async def sse_stream(request: Request) -> StreamingResponse:
            """Server-Sent Events endpoint for async MCP transport."""

            async def event_generator():
                yield 'data: {"jsonrpc": "2.0", "method": "connected"}\n\n'
                # Keep connection alive
                while not await request.is_disconnected():
                    await asyncio.sleep(15)
                    yield ": keepalive\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        return app

    async def run(self) -> None:
        """Start the HTTP server and block until stopped."""
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                "uvicorn is required for SSEProxyServer. Install with: pip install uvicorn"
            ) from exc

        app = self._build_app()
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="info")
        server = uvicorn.Server(config)
        logger.info("[AgentGuard/MCP] SSEProxyServer listening on port %d", self.port)
        await server.serve()
