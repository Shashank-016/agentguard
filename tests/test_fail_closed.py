"""Regression tests for fail-closed engine-error handling (Fix 2).

Before this fix, an unexpected exception inside the security engine
(``InjectionDetector.scan``/``scan_messages``, ``ToolPolicyEngine.check``/
``check_arguments``, ``TrustScorer.should_flag``, ``MCPToolCallParams(**params)``)
would propagate out of the wrapper uncaught — or in some paths, silently
swallow the underlying tool call — leaving the agent's behavior undefined and
no record of the failure. The contract is now: in ``enforce``/``interactive``
mode the call fails *closed* (raises / blocks) and a critical ``engine_error``
event is always emitted; in ``observe`` mode the call still proceeds (fails
*open*) but the failure is recorded just as loudly.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentmoat.async_client import AsyncGuardedClient
from agentmoat.bus import EventBus
from agentmoat.client import AgentMoatException, GuardedClient
from agentmoat.mcp.interceptor import MCPInterceptor
from agentmoat.mcp.models import AGENTMOAT_ENGINE_ERROR, MCPRequest
from agentmoat.openai_client import GuardedOpenAI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_anthropic_client():
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(type="text", text="Hello!")]
    client.messages.create.return_value = response
    return client


def _tool_use_anthropic_client(tool_name: str = "write_file", tool_input: dict | None = None):
    client = MagicMock()
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input or {}
    response = MagicMock()
    response.content = [block]
    client.messages.create.return_value = response
    return client


def _mock_async_anthropic_client():
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(type="text", text="Hello!")]
    client.messages.create = AsyncMock(return_value=response)
    return client


def _tool_use_async_anthropic_client(tool_name: str = "write_file", tool_input: dict | None = None):
    client = MagicMock()
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input or {}
    response = MagicMock()
    response.content = [block]
    client.messages.create = AsyncMock(return_value=response)
    return client


def _mock_openai_client():
    client = MagicMock()
    message = MagicMock(content="Hello!", tool_calls=None)
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    client.chat.completions.create.return_value = response
    return client


def _tool_call_openai_client(tool_name: str = "write_file", tool_arguments: dict | None = None):
    client = MagicMock()
    function = MagicMock(arguments=json.dumps(tool_arguments or {}))
    function.name = tool_name
    tool_call = MagicMock(function=function)
    message = MagicMock(content=None, tool_calls=[tool_call])
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    client.chat.completions.create.return_value = response
    return client


def _engine_errors(bus, session_id):
    return [e for e in bus.get_session_events(session_id) if e.event_type == "engine_error"]


SIMPLE_MESSAGES = [{"role": "user", "content": "Hello there"}]


# ---------------------------------------------------------------------------
# Sync GuardedClient (Anthropic SDK)
# ---------------------------------------------------------------------------


class TestSyncGuardedClientFailClosed:
    def test_enforce_mode_raises_and_emits_engine_error_on_scan_failure(self):
        bus = EventBus()
        gc = GuardedClient(
            _mock_anthropic_client(), session_id="fc-sync-1", bus=bus, mode="enforce"
        )

        with patch.object(
            gc._injection_detector, "scan_messages", side_effect=RuntimeError("scan boom")
        ):
            with pytest.raises(AgentMoatException):
                gc.messages.create(model="m", max_tokens=10, messages=SIMPLE_MESSAGES)

        errors = _engine_errors(bus, "fc-sync-1")
        assert len(errors) == 1
        assert errors[0].severity == "critical"
        assert errors[0].payload["phase"] == "injection_scan"
        assert errors[0].payload["error_type"] == "RuntimeError"

    def test_observe_mode_proceeds_and_emits_engine_error_on_scan_failure(self):
        bus = EventBus()
        gc = GuardedClient(
            _mock_anthropic_client(), session_id="fc-sync-2", bus=bus, mode="observe"
        )

        with patch.object(
            gc._injection_detector, "scan_messages", side_effect=RuntimeError("scan boom")
        ):
            response = gc.messages.create(model="m", max_tokens=10, messages=SIMPLE_MESSAGES)

        assert response is not None
        errors = _engine_errors(bus, "fc-sync-2")
        assert len(errors) == 1
        assert errors[0].severity == "critical"
        assert errors[0].payload["phase"] == "injection_scan"

    def test_enforce_mode_raises_on_tool_evaluation_failure(self):
        bus = EventBus()
        client = _tool_use_anthropic_client(tool_name="write_file", tool_input={"path": "/tmp/x"})
        gc = GuardedClient(client, session_id="fc-sync-3", bus=bus, mode="enforce")

        with patch.object(gc._trust_scorer, "should_flag", side_effect=RuntimeError("trust boom")):
            with pytest.raises(AgentMoatException):
                gc.messages.create(model="m", max_tokens=10, messages=SIMPLE_MESSAGES)

        errors = _engine_errors(bus, "fc-sync-3")
        assert len(errors) == 1
        assert errors[0].payload["phase"] == "post_call_tool_evaluation"

    def test_observe_mode_proceeds_on_tool_evaluation_failure(self):
        bus = EventBus()
        client = _tool_use_anthropic_client(tool_name="write_file", tool_input={"path": "/tmp/x"})
        gc = GuardedClient(client, session_id="fc-sync-4", bus=bus, mode="observe")

        with patch.object(gc._trust_scorer, "should_flag", side_effect=RuntimeError("trust boom")):
            response = gc.messages.create(model="m", max_tokens=10, messages=SIMPLE_MESSAGES)

        assert response is not None
        errors = _engine_errors(bus, "fc-sync-4")
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# AsyncGuardedClient (Anthropic SDK)
# ---------------------------------------------------------------------------


class TestAsyncGuardedClientFailClosed:
    def test_enforce_mode_raises_and_emits_engine_error_on_scan_failure(self):
        bus = EventBus()
        gc = AsyncGuardedClient(
            _mock_async_anthropic_client(), session_id="fc-async-1", bus=bus, mode="enforce"
        )

        async def run():
            with patch.object(
                gc._injection_detector, "scan_messages", side_effect=RuntimeError("scan boom")
            ):
                with pytest.raises(AgentMoatException):
                    await gc.messages.create(model="m", max_tokens=10, messages=SIMPLE_MESSAGES)

        asyncio.run(run())

        errors = _engine_errors(bus, "fc-async-1")
        assert len(errors) == 1
        assert errors[0].severity == "critical"
        assert errors[0].payload["phase"] == "injection_scan"

    def test_observe_mode_proceeds_and_emits_engine_error_on_tool_evaluation_failure(self):
        bus = EventBus()
        client = _tool_use_async_anthropic_client(
            tool_name="write_file", tool_input={"path": "/tmp/x"}
        )
        gc = AsyncGuardedClient(client, session_id="fc-async-2", bus=bus, mode="observe")

        async def run():
            with patch.object(
                gc._trust_scorer, "should_flag", side_effect=RuntimeError("trust boom")
            ):
                return await gc.messages.create(model="m", max_tokens=10, messages=SIMPLE_MESSAGES)

        response = asyncio.run(run())

        assert response is not None
        errors = _engine_errors(bus, "fc-async-2")
        assert len(errors) == 1
        assert errors[0].payload["phase"] == "post_call_tool_evaluation"


# ---------------------------------------------------------------------------
# GuardedOpenAI (sync)
# ---------------------------------------------------------------------------


class TestGuardedOpenAIFailClosed:
    def test_enforce_mode_raises_and_emits_engine_error_on_scan_failure(self):
        bus = EventBus()
        gc = GuardedOpenAI(_mock_openai_client(), session_id="fc-openai-1", bus=bus, mode="enforce")

        with patch.object(
            gc._injection_detector, "scan_messages", side_effect=RuntimeError("scan boom")
        ):
            with pytest.raises(AgentMoatException):
                gc.chat.completions.create(model="m", messages=SIMPLE_MESSAGES)

        errors = _engine_errors(bus, "fc-openai-1")
        assert len(errors) == 1
        assert errors[0].severity == "critical"
        assert errors[0].payload["phase"] == "injection_scan"

    def test_enforce_mode_raises_on_tool_evaluation_failure(self):
        bus = EventBus()
        client = _tool_call_openai_client(tool_name="write_file", tool_arguments={"path": "/tmp/x"})
        gc = GuardedOpenAI(client, session_id="fc-openai-2", bus=bus, mode="enforce")

        with patch.object(
            gc._policy_engine, "check_arguments", side_effect=RuntimeError("policy boom")
        ):
            with pytest.raises(AgentMoatException):
                gc.chat.completions.create(model="m", messages=SIMPLE_MESSAGES)

        errors = _engine_errors(bus, "fc-openai-2")
        assert len(errors) == 1
        assert errors[0].payload["phase"] == "post_call_tool_evaluation"

    def test_observe_mode_proceeds_and_emits_engine_error(self):
        bus = EventBus()
        client = _tool_call_openai_client(tool_name="write_file", tool_arguments={"path": "/tmp/x"})
        gc = GuardedOpenAI(client, session_id="fc-openai-3", bus=bus, mode="observe")

        with patch.object(
            gc._policy_engine, "check_arguments", side_effect=RuntimeError("policy boom")
        ):
            response = gc.chat.completions.create(model="m", messages=SIMPLE_MESSAGES)

        assert response is not None
        errors = _engine_errors(bus, "fc-openai-3")
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# MCPInterceptor
# ---------------------------------------------------------------------------


def _make_interceptor(mode: str, bus: EventBus | None = None) -> MCPInterceptor:
    return MCPInterceptor(
        agent_id="test-agent",
        session_id="test-session",
        bus=bus or EventBus(),
        mode=mode,
    )


def _tools_call_request(tool_name: str = "write_file", arguments: dict | None = None) -> MCPRequest:
    return MCPRequest(
        id=1, method="tools/call", params={"name": tool_name, "arguments": arguments or {}}
    )


class TestMCPInterceptorFailClosed:
    def test_enforce_mode_blocks_and_emits_engine_error_on_scan_failure(self):
        bus = EventBus()
        interceptor = _make_interceptor("enforce", bus)

        with patch.object(interceptor._injection, "scan", side_effect=RuntimeError("scan boom")):
            result = interceptor.intercept(_tools_call_request())

        assert result.allowed is False
        assert result.block_code == AGENTMOAT_ENGINE_ERROR
        assert "Internal security engine error" in result.block_reason

        engine_errors = [e for e in result.events if e.event_type == "engine_error"]
        assert len(engine_errors) == 1
        assert engine_errors[0].severity == "critical"
        assert engine_errors[0].payload["error_type"] == "RuntimeError"

    def test_interactive_mode_blocks_and_emits_engine_error_on_check_arguments_failure(self):
        bus = EventBus()
        interceptor = _make_interceptor("interactive", bus)

        with patch.object(
            interceptor._policy, "check_arguments", side_effect=RuntimeError("policy boom")
        ):
            result = interceptor.intercept(_tools_call_request())

        assert result.allowed is False
        assert result.block_code == AGENTMOAT_ENGINE_ERROR
        assert any(e.event_type == "engine_error" for e in result.events)

    def test_observe_mode_fails_open_but_emits_engine_error(self):
        bus = EventBus()
        interceptor = _make_interceptor("observe", bus)

        with patch.object(interceptor._injection, "scan", side_effect=RuntimeError("scan boom")):
            result = interceptor.intercept(_tools_call_request())

        assert result.allowed is True
        assert result.block_reason is None
        assert result.block_code is None

        engine_errors = [e for e in result.events if e.event_type == "engine_error"]
        assert len(engine_errors) == 1
        assert engine_errors[0].severity == "critical"

    def test_enforce_mode_blocks_on_malformed_params(self):
        """``MCPToolCallParams(**params)`` itself can raise (e.g. wrong shape)."""
        bus = EventBus()
        interceptor = _make_interceptor("enforce", bus)

        bad_request = MCPRequest(
            id=1, method="tools/call", params={"arguments": "not-a-dict-of-args", "name": 12345}
        )
        # name=12345 / arguments=str will fail Pydantic validation inside MCPToolCallParams
        result = interceptor.intercept(bad_request)

        assert result.allowed is False
        assert result.block_code == AGENTMOAT_ENGINE_ERROR
        assert any(e.event_type == "engine_error" for e in result.events)
