"""Tests for the AgentMoat CLI — audit subcommands and MCP proxy entry points."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from agentmoat.audit import AuditLogger
from agentmoat.cli import cli
from agentmoat.events import SecurityEvent


def _make_event(
    event_type: str = "tool_call", severity: str = "info", session_id: str = "s1"
) -> SecurityEvent:
    return SecurityEvent(
        session_id=session_id,
        agent_id="test-agent",
        source="mcp",
        event_type=event_type,
        severity=severity,
        payload={"n": 1},
    )


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# agentmoat audit verify
# ---------------------------------------------------------------------------


class TestAuditVerify:
    def test_verify_intact_chain_exits_zero(self, tmp_path, runner):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path))
        for _ in range(3):
            log.write(_make_event())

        result = runner.invoke(cli, ["audit", "verify", str(path)])

        assert result.exit_code == 0
        assert "Chain intact" in result.output

    def test_verify_broken_chain_exits_one(self, tmp_path, runner):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path))
        for _ in range(3):
            log.write(_make_event())

        lines = path.read_text(encoding="utf-8").splitlines()
        tampered = json.loads(lines[1])
        tampered["payload"] = {"n": 9999}
        lines[1] = json.dumps(tampered)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = runner.invoke(cli, ["audit", "verify", str(path)])

        assert result.exit_code == 1
        assert "Chain broken" in result.output


# ---------------------------------------------------------------------------
# agentmoat audit tail
# ---------------------------------------------------------------------------


class TestAuditTail:
    def test_tail_prints_records_oldest_first(self, tmp_path, runner):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path))
        for i in range(5):
            log.write(_make_event(session_id=f"s{i}"))

        result = runner.invoke(cli, ["audit", "tail", str(path), "-n", "2"])

        assert result.exit_code == 0
        lines = [line for line in result.output.splitlines() if line.strip()]
        records = [json.loads(line) for line in lines]
        assert [r["session_id"] for r in records] == ["s3", "s4"]

    def test_tail_default_count_returns_all_records(self, tmp_path, runner):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path))
        for i in range(3):
            log.write(_make_event(session_id=f"s{i}"))

        result = runner.invoke(cli, ["audit", "tail", str(path)])

        assert result.exit_code == 0
        lines = [line for line in result.output.splitlines() if line.strip()]
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# agentmoat audit stats
# ---------------------------------------------------------------------------


class TestAuditStats:
    def test_stats_prints_counts_by_type_and_severity(self, tmp_path, runner):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path))
        log.write(_make_event(event_type="tool_call", severity="info"))
        log.write(_make_event(event_type="tool_call", severity="info"))
        log.write(_make_event(event_type="injection_detected", severity="critical"))

        result = runner.invoke(cli, ["audit", "stats", str(path)])

        assert result.exit_code == 0
        assert "Total records: 3" in result.output
        assert "tool_call: 2" in result.output
        assert "injection_detected: 1" in result.output
        assert "info: 2" in result.output
        assert "critical: 1" in result.output

    def test_stats_skips_malformed_lines(self, tmp_path, runner):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path))
        log.write(_make_event(event_type="tool_call", severity="info"))
        with path.open("a", encoding="utf-8") as f:
            f.write("not json\n")
            f.write("\n")

        result = runner.invoke(cli, ["audit", "stats", str(path)])

        assert result.exit_code == 0
        assert "Total records: 1" in result.output


# ---------------------------------------------------------------------------
# agentmoat mcp proxy stdio|sse
# ---------------------------------------------------------------------------


class TestMcpProxyEntryPoints:
    def test_stdio_invokes_run_stdio(self, runner):
        with patch("agentmoat.cli._run_stdio", new=AsyncMock()) as mock_run:
            result = runner.invoke(
                cli,
                [
                    "mcp",
                    "proxy",
                    "stdio",
                    "--upstream-cmd",
                    "echo hi",
                    "--agent-id",
                    "researcher",
                    "--mode",
                    "enforce",
                    "--session-id",
                    "sess-1",
                ],
            )

        assert result.exit_code == 0
        mock_run.assert_called_once_with("echo hi", "researcher", None, "enforce", "sess-1")

    def test_sse_invokes_run_sse(self, runner):
        with patch("agentmoat.cli._run_sse", new=AsyncMock()) as mock_run:
            result = runner.invoke(
                cli,
                [
                    "mcp",
                    "proxy",
                    "sse",
                    "--upstream-url",
                    "http://localhost:9000",
                    "--port",
                    "9100",
                    "--agent-id",
                    "researcher",
                ],
            )

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            "http://localhost:9000", 9100, "researcher", "observe", None, None
        )
