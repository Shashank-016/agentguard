"""MCP upstream clients — connect AgentGuard proxy to the real MCP server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

from .models import INTERNAL_ERROR, PARSE_ERROR, MCPRequest, MCPResponse

logger = logging.getLogger(__name__)


class StdioUpstreamClient:
    """Spawns a real MCP server as a subprocess and communicates via stdin/stdout.

    Parameters
    ----------
    command:
        Command and arguments to launch the upstream MCP server, e.g.
        ``["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]``.
    env:
        Optional environment variable overrides for the subprocess.
    """

    def __init__(
        self, command: list[str], env: Optional[dict[str, str]] = None
    ) -> None:
        self.command = command
        self.env = env
        self._process: Optional[asyncio.subprocess.Process] = None
        self._pending: dict[str | int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Spawn the upstream MCP server process."""
        merged_env = {**os.environ, **(self.env or {})}
        self._process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        logger.info(
            "[AgentGuard/MCP] Upstream process started: %s (pid=%d)",
            " ".join(self.command),
            self._process.pid,
        )

    async def send(self, request: MCPRequest) -> MCPResponse:
        """Send a JSON-RPC request to the upstream server and await its response.

        Raises
        ------
        RuntimeError
            If the upstream process is not running.
        """
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Upstream client not started — call start() first")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[MCPResponse] = loop.create_future()

        req_id = request.id
        if req_id is not None:
            self._pending[req_id] = future

        line = request.model_dump_json() + "\n"
        self._process.stdin.write(line.encode())
        await self._process.stdin.drain()

        if req_id is None:
            # Notification — no response expected
            return MCPResponse(id=None, result={})

        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            logger.error("[AgentGuard/MCP] Timeout waiting for response to id=%s", req_id)
            return MCPResponse(
                id=req_id,
                error={"code": INTERNAL_ERROR, "message": "Upstream response timeout"},
            )

    async def stop(self) -> None:
        """Terminate the upstream process cleanly."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                self._process.kill()
            logger.info("[AgentGuard/MCP] Upstream process terminated")

    async def _read_loop(self) -> None:
        """Background task: read stdout lines and resolve pending futures."""
        if self._process is None or self._process.stdout is None:
            return
        try:
            async for raw_line in self._process.stdout:
                line = raw_line.decode().strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    response = MCPResponse(**data)
                    req_id = response.id
                    if req_id is not None and req_id in self._pending:
                        future = self._pending.pop(req_id)
                        if not future.done():
                            future.set_result(response)
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    # Malformed/unexpected line from upstream — not a reader-loop bug.
                    logger.debug(
                        "[AgentGuard/MCP] Failed to parse upstream line: %s — %s",
                        line[:200],
                        exc,
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("[AgentGuard/MCP] Read loop crashed")
        finally:
            # Resolve all pending futures with an error
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(
                        RuntimeError("Upstream process terminated unexpectedly")
                    )
            self._pending.clear()

    async def _drain_stderr(self) -> None:
        """Background task: continuously read and log the upstream process's stderr.

        MCP servers often write diagnostics to stderr. Left unread, the pipe's
        OS buffer fills up and the subprocess blocks on its next write — a
        silent deadlock. Logging at DEBUG keeps this output available for
        troubleshooting without polluting normal operation.
        """
        if self._process is None or self._process.stderr is None:
            return
        try:
            async for raw_line in self._process.stderr:
                line = raw_line.decode(errors="replace").rstrip()
                if line:
                    logger.debug("[AgentGuard/MCP] upstream stderr: %s", line)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("[AgentGuard/MCP] stderr drain task crashed")


class SSEUpstreamClient:
    """Connects to a remote MCP server via HTTP + Server-Sent Events.

    Parameters
    ----------
    base_url:
        Base URL of the remote MCP server (e.g. ``"http://localhost:3000"``).
    headers:
        Optional HTTP headers (e.g. auth tokens) to include with every request.
    """

    def __init__(self, base_url: str, headers: Optional[dict[str, str]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self._client: Optional[object] = None

    async def start(self) -> None:
        """Initialize the HTTP client."""
        try:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=30.0,
            )
            logger.info("[AgentGuard/MCP] SSE upstream client connected to %s", self.base_url)
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for SSEUpstreamClient. "
                "Install with: pip install httpx"
            ) from exc

    async def send(self, request: MCPRequest) -> MCPResponse:
        """POST a JSON-RPC request to the upstream HTTP endpoint."""
        if self._client is None:
            raise RuntimeError("SSE upstream client not started — call start() first")

        import httpx

        try:
            response = await self._client.post(  # type: ignore[attr-defined]
                "/mcp",
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            return MCPResponse(**data)
        except httpx.HTTPStatusError as exc:
            logger.error("[AgentGuard/MCP] Upstream HTTP error: %s", exc)
            return MCPResponse(
                id=request.id,
                error={"code": INTERNAL_ERROR, "message": str(exc)},
            )
        except Exception as exc:
            logger.error("[AgentGuard/MCP] Upstream request failed: %s", exc)
            return MCPResponse(
                id=request.id,
                error={"code": INTERNAL_ERROR, "message": str(exc)},
            )

    async def stop(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()  # type: ignore[attr-defined]
            logger.info("[AgentGuard/MCP] SSE upstream client closed")
