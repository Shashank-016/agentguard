"""Tests for the MCP upstream transports — StdioUpstreamClient and SSEUpstreamClient."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agentguard.mcp.client import SSEUpstreamClient, StdioUpstreamClient
from agentguard.mcp.models import INTERNAL_ERROR, MCPRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeStream:
    """Async-iterable stand-in for ``asyncio.StreamReader`` that blocks "forever"
    once its lines are exhausted — mimics a live pipe with no more output."""

    def __init__(self, lines: list[bytes] | None = None) -> None:
        self._lines = list(lines or [])

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._lines:
            await asyncio.sleep(3600)
        return self._lines.pop(0)


class _EOFStream:
    """Async-iterable that raises ``StopAsyncIteration`` once exhausted, like
    a closed pipe."""

    def __init__(self, lines: list[bytes] | None = None) -> None:
        self._lines = list(lines or [])

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _RaisingStream:
    """Async-iterable whose first iteration raises a non-cancellation error."""

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        raise RuntimeError("stream broke")


def _fake_process(stdout_lines=None, stderr_lines=None, stdout=None, stderr=None):
    process = MagicMock()
    process.pid = 4242
    process.stdin = MagicMock()
    process.stdin.write = MagicMock()
    process.stdin.drain = AsyncMock()
    process.stdout = stdout if stdout is not None else _FakeStream(stdout_lines)
    process.stderr = stderr if stderr is not None else _FakeStream(stderr_lines)
    process.terminate = MagicMock()
    process.kill = MagicMock()
    process.wait = AsyncMock(return_value=0)
    return process


# ---------------------------------------------------------------------------
# StdioUpstreamClient.send
# ---------------------------------------------------------------------------


class TestStdioSend:
    @pytest.mark.asyncio
    async def test_send_without_start_raises_runtime_error(self):
        client = StdioUpstreamClient(command=["fake-server"])
        with pytest.raises(RuntimeError, match="not started"):
            await client.send(MCPRequest(id=1, method="tools/list", params={}))

    @pytest.mark.asyncio
    async def test_send_notification_returns_empty_response(self):
        process = _fake_process()
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            client = StdioUpstreamClient(command=["fake-server"])
            await client.start()
            try:
                response = await client.send(MCPRequest(id=None, method="notify", params={}))
                assert response.id is None
                assert response.result == {}
            finally:
                await client.stop()

    @pytest.mark.asyncio
    async def test_send_timeout_returns_internal_error_response(self):
        process = _fake_process()
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            client = StdioUpstreamClient(command=["fake-server"])
            await client.start()
            try:
                with patch(
                    "agentguard.mcp.client.asyncio.wait_for",
                    side_effect=asyncio.TimeoutError,
                ):
                    response = await client.send(
                        MCPRequest(id=99, method="tools/list", params={})
                    )
                assert response.id == 99
                assert response.error is not None
                assert response.error["code"] == INTERNAL_ERROR
                assert 99 not in client._pending
            finally:
                await client.stop()


# ---------------------------------------------------------------------------
# StdioUpstreamClient._read_loop
# ---------------------------------------------------------------------------


class TestStdioReadLoop:
    @pytest.mark.asyncio
    async def test_read_loop_returns_immediately_without_process(self):
        client = StdioUpstreamClient(command=["fake-server"])
        await client._read_loop()  # no process — should return without error

    @pytest.mark.asyncio
    async def test_read_loop_skips_empty_and_malformed_lines(self, caplog):
        caplog.set_level(logging.DEBUG, logger="agentguard.mcp.client")
        process = _fake_process(
            stdout_lines=[
                b"\n",
                b"not json\n",
                b'{"jsonrpc": "2.0", "id": 7, "result": {"ok": true}}\n',
            ]
        )
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            client = StdioUpstreamClient(command=["fake-server"])
            await client.start()
            try:
                response = await client.send(MCPRequest(id=7, method="tools/list", params={}))
                assert response.result == {"ok": True}
            finally:
                await client.stop()

        assert "Failed to parse upstream line" in caplog.text

    @pytest.mark.asyncio
    async def test_read_loop_logs_exception_and_resolves_pending_futures(self, caplog):
        caplog.set_level(logging.ERROR, logger="agentguard.mcp.client")
        process = _fake_process(stdout=_RaisingStream())
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            client = StdioUpstreamClient(command=["fake-server"])
            await client.start()

            loop = asyncio.get_running_loop()
            pending: asyncio.Future = loop.create_future()
            client._pending["pending-id"] = pending

            await client._reader_task

            with pytest.raises(RuntimeError, match="terminated unexpectedly"):
                await pending

            await client.stop()

        assert "Read loop crashed" in caplog.text


# ---------------------------------------------------------------------------
# StdioUpstreamClient._drain_stderr
# ---------------------------------------------------------------------------


class TestStdioDrainStderr:
    @pytest.mark.asyncio
    async def test_drain_stderr_logs_exception_on_unexpected_error(self, caplog):
        caplog.set_level(logging.ERROR, logger="agentguard.mcp.client")
        process = _fake_process(stderr=_RaisingStream())
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            client = StdioUpstreamClient(command=["fake-server"])
            await client.start()
            await client._stderr_task
            await client.stop()

        assert "stderr drain task crashed" in caplog.text


# ---------------------------------------------------------------------------
# StdioUpstreamClient.stop
# ---------------------------------------------------------------------------


class TestStdioStop:
    @pytest.mark.asyncio
    async def test_stop_swallows_cancelled_error_from_stderr_task(self):
        client = StdioUpstreamClient(command=["fake-server"])
        client._reader_task = None
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        fut.cancel()
        client._stderr_task = fut

        await client.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_stop_kills_process_when_wait_times_out(self):
        process = _fake_process()
        process.wait = AsyncMock(side_effect=asyncio.TimeoutError)
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            client = StdioUpstreamClient(command=["fake-server"])
            await client.start()
            await client.stop()

        process.kill.assert_called_once()


# ---------------------------------------------------------------------------
# SSEUpstreamClient
# ---------------------------------------------------------------------------


class TestSSEUpstreamClient:
    def test_init_sets_base_url_and_headers(self):
        client = SSEUpstreamClient(base_url="http://localhost:3000/", headers={"X-Test": "1"})
        assert client.base_url == "http://localhost:3000"
        assert client.headers == {"X-Test": "1"}
        assert client._client is None

    @pytest.mark.asyncio
    async def test_start_creates_httpx_client(self):
        client = SSEUpstreamClient(base_url="http://localhost:3000")
        await client.start()
        try:
            assert client._client is not None
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_start_raises_runtime_error_if_httpx_missing(self):
        client = SSEUpstreamClient(base_url="http://localhost:3000")
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(RuntimeError, match="httpx is required"):
                await client.start()

    @pytest.mark.asyncio
    async def test_send_without_start_raises_runtime_error(self):
        client = SSEUpstreamClient(base_url="http://localhost:3000")
        with pytest.raises(RuntimeError, match="not started"):
            await client.send(MCPRequest(id=1, method="tools/list", params={}))

    @pytest.mark.asyncio
    async def test_send_success_returns_response(self):
        client = SSEUpstreamClient(base_url="http://localhost:3000")
        await client.start()
        try:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = MagicMock(
                return_value={"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
            )
            client._client.post = AsyncMock(return_value=mock_response)

            response = await client.send(MCPRequest(id=1, method="tools/list", params={}))
            assert response.result == {"ok": True}
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_send_http_status_error_returns_error_response(self):
        client = SSEUpstreamClient(base_url="http://localhost:3000")
        await client.start()
        try:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "server error", request=MagicMock(), response=MagicMock()
                )
            )
            client._client.post = AsyncMock(return_value=mock_response)

            response = await client.send(MCPRequest(id=2, method="tools/list", params={}))
            assert response.id == 2
            assert response.error is not None
            assert response.error["code"] == INTERNAL_ERROR
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_send_generic_exception_returns_error_response(self):
        client = SSEUpstreamClient(base_url="http://localhost:3000")
        await client.start()
        try:
            client._client.post = AsyncMock(side_effect=RuntimeError("connection refused"))

            response = await client.send(MCPRequest(id=3, method="tools/list", params={}))
            assert response.id == 3
            assert response.error is not None
            assert response.error["code"] == INTERNAL_ERROR
            assert "connection refused" in response.error["message"]
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_stop_closes_client(self):
        client = SSEUpstreamClient(base_url="http://localhost:3000")
        await client.start()
        await client.stop()
        assert client._client.is_closed
