"""Regression tests for StdioUpstreamClient stderr draining (Fix 4).

Before this fix, the upstream subprocess's stderr pipe was never read. MCP
servers routinely write diagnostics to stderr; once the OS pipe buffer fills
up, the subprocess blocks on its next write to stderr — a silent deadlock
that looks like a hung upstream server. A background task must continuously
drain stderr (logged at DEBUG) for the lifetime of the process, and be
cancelled cleanly in ``stop()``.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentmoat.mcp.client import StdioUpstreamClient


class _FakeStream:
    """Minimal async-iterable stand-in for ``asyncio.StreamReader``."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._lines:
            await asyncio.sleep(3600)  # block "forever" — like a live pipe with no more output
        return self._lines.pop(0)


def _fake_process(stdout_lines=None, stderr_lines=None):
    process = MagicMock()
    process.pid = 4242
    process.stdin = MagicMock()
    process.stdin.write = MagicMock()
    process.stdin.drain = AsyncMock()
    process.stdout = _FakeStream(stdout_lines or [])
    process.stderr = _FakeStream(stderr_lines or [])
    process.terminate = MagicMock()
    process.wait = AsyncMock(return_value=0)
    return process


@pytest.fixture
def caplog_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="agentmoat.mcp.client")
    return caplog


class TestStderrDrain:
    @pytest.mark.asyncio
    async def test_stderr_lines_are_logged_at_debug(self, caplog_debug):
        process = _fake_process(
            stderr_lines=[b"warning: cache miss\n", b"booting upstream server\n"]
        )

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            client = StdioUpstreamClient(command=["fake-server"])
            await client.start()
            try:
                # Give the background drain task a few ticks to consume both lines.
                for _ in range(20):
                    await asyncio.sleep(0)
                    if (
                        "warning: cache miss" in caplog_debug.text
                        and "booting upstream server" in caplog_debug.text
                    ):
                        break
            finally:
                await client.stop()

        assert "warning: cache miss" in caplog_debug.text
        assert "booting upstream server" in caplog_debug.text
        assert "upstream stderr" in caplog_debug.text

    @pytest.mark.asyncio
    async def test_stderr_task_created_and_cancelled_on_stop(self):
        process = _fake_process()

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            client = StdioUpstreamClient(command=["fake-server"])
            await client.start()

            assert client._stderr_task is not None
            assert not client._stderr_task.done()

            await client.stop()

            # The task catches CancelledError internally and exits cleanly —
            # done but not in the "cancelled" state, mirroring _read_loop.
            assert client._stderr_task.done()
            assert client._stderr_task.exception() is None

    @pytest.mark.asyncio
    async def test_stop_is_safe_when_stderr_is_none(self):
        """``_drain_stderr`` must not blow up if the process has no stderr pipe."""
        process = _fake_process()
        process.stderr = None

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            client = StdioUpstreamClient(command=["fake-server"])
            await client.start()
            # The drain task returns immediately when stderr is None.
            for _ in range(10):
                await asyncio.sleep(0)
            await client.stop()

        assert client._stderr_task is not None
        assert client._stderr_task.done()

    @pytest.mark.asyncio
    async def test_send_uses_running_loop_without_runtime_error(self):
        """Regression for the ``asyncio.get_event_loop()`` deprecation/footgun —
        ``send`` must work cleanly from inside a running event loop."""
        process = _fake_process(
            stdout_lines=[b'{"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}\n']
        )

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            client = StdioUpstreamClient(command=["fake-server"])
            await client.start()
            try:
                from agentmoat.mcp.models import MCPRequest

                response = await client.send(MCPRequest(id=1, method="tools/list", params={}))
                assert response.result == {"ok": True}
            finally:
                await client.stop()
