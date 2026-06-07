"""Tests for ArgumentConstraintChecker and ToolPolicyEngine.check_arguments()."""

from __future__ import annotations

import pytest

from agentguard.bus import EventBus
from agentguard.engine.constraints import ArgumentConstraintChecker, ConstraintViolation
from agentguard.engine.policy import ToolConstraints, ToolPolicyEngine
from agentguard.mcp.interceptor import MCPInterceptor
from agentguard.mcp.models import MCPRequest

POLICY_YAML = """
version: "1"
agents:
  writer:
    allowed_tools:
      - write_file
      - read_file
      - fetch
    denied_tools:
      - execute_code
    tool_constraints:
      write_file:
        path_allowlist: ["/tmp/**"]
      read_file:
        path_denylist: ["/etc/**"]
      fetch:
        url_denylist: ["169.254.169.254", "localhost", "10.*"]
"""


@pytest.fixture
def policy_file(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(POLICY_YAML)
    return str(path)


@pytest.fixture
def engine(policy_file) -> ToolPolicyEngine:
    return ToolPolicyEngine(policy_path=policy_file)


@pytest.fixture
def checker() -> ArgumentConstraintChecker:
    return ArgumentConstraintChecker()


# ---------------------------------------------------------------------------
# Built-in detectors
# ---------------------------------------------------------------------------


class TestBuiltinDetectors:
    def test_path_traversal(self, checker):
        violations = checker.check("read_file", {"path": "../../etc/passwd"})
        assert any(v.constraint == "path_traversal" for v in violations)

    def test_path_traversal_url_encoded(self, checker):
        violations = checker.check("read_file", {"path": "/tmp/%2e%2e/etc/passwd"})
        assert any(v.constraint == "path_traversal" for v in violations)

    def test_ssrf_target(self, checker):
        violations = checker.check("fetch", {"url": "http://169.254.169.254/latest/meta-data"})
        assert any(v.constraint == "ssrf_target" for v in violations)

    def test_ssrf_target_rfc1918(self, checker):
        violations = checker.check("fetch", {"url": "http://192.168.1.5/admin"})
        assert any(v.constraint == "ssrf_target" for v in violations)

    def test_shell_metachar(self, checker):
        violations = checker.check("execute", {"command": "ls; rm -rf /"})
        assert any(v.constraint == "shell_metachar" for v in violations)

    def test_shell_metachar_not_flagged_for_non_exec_tool(self, checker):
        violations = checker.check("read_file", {"path": "ls; rm -rf /"})
        assert not any(v.constraint == "shell_metachar" for v in violations)

    def test_sensitive_path(self, checker):
        violations = checker.check("read_file", {"path": "/etc/shadow"})
        assert any(v.constraint == "sensitive_path" for v in violations)

    def test_sensitive_path_ssh_key(self, checker):
        violations = checker.check("read_file", {"path": "~/.ssh/id_rsa"})
        assert any(v.constraint == "sensitive_path" for v in violations)

    def test_clean_call_returns_empty(self, checker):
        violations = checker.check("read_file", {"path": "/tmp/notes.txt"})
        assert violations == []

    def test_violation_flag_property(self):
        v = ConstraintViolation(
            constraint="path_traversal", argument="path", value="../x", detail="x"
        )
        assert v.flag == "constraint:path_traversal"

    def test_violation_value_truncated(self, checker):
        long_path = "/etc/" + "a" * 500
        violations = checker.check("read_file", {"path": long_path})
        sensitive = [v for v in violations if v.constraint == "sensitive_path"]
        assert sensitive
        assert len(sensitive[0].value) <= 201


# ---------------------------------------------------------------------------
# Per-tool configured constraints
# ---------------------------------------------------------------------------


class TestConfiguredConstraints:
    def test_path_allowlist_passes(self, checker):
        constraints = ToolConstraints(path_allowlist=["/tmp/**"])
        violations = checker.check("write_file", {"path": "/tmp/ok.txt"}, constraints=constraints)
        assert not any(v.constraint == "path_not_allowed" for v in violations)

    def test_path_allowlist_blocks_outside(self, checker):
        constraints = ToolConstraints(path_allowlist=["/tmp/**"])
        violations = checker.check("write_file", {"path": "/home/x"}, constraints=constraints)
        assert any(v.constraint == "path_not_allowed" for v in violations)

    def test_path_denylist(self, checker):
        constraints = ToolConstraints(path_denylist=["/etc/**"])
        violations = checker.check("write_file", {"path": "/etc/crontab"}, constraints=constraints)
        assert any(v.constraint == "path_denied" for v in violations)

    def test_url_denylist_bare_host(self, checker):
        constraints = ToolConstraints(url_denylist=["169.254.169.254"])
        violations = checker.check("fetch", {"url": "169.254.169.254"}, constraints=constraints)
        assert any(v.constraint == "url_denied" for v in violations)

    def test_url_denylist_full_url(self, checker):
        constraints = ToolConstraints(url_denylist=["169.254.169.254"])
        violations = checker.check(
            "fetch", {"url": "http://169.254.169.254/latest/meta-data"}, constraints=constraints
        )
        assert any(v.constraint == "url_denied" for v in violations)

    def test_url_denylist_glob(self, checker):
        constraints = ToolConstraints(url_denylist=["10.*"])
        violations = checker.check(
            "fetch", {"url": "http://10.0.0.5/internal"}, constraints=constraints
        )
        assert any(v.constraint == "url_denied" for v in violations)

    def test_url_allowlist_blocks_non_matching(self, checker):
        constraints = ToolConstraints(url_allowlist=["api.example.com"])
        violations = checker.check(
            "fetch", {"url": "http://evil.example.org/x"}, constraints=constraints
        )
        assert any(v.constraint == "url_not_allowed" for v in violations)

    def test_max_arg_length(self, checker):
        constraints = ToolConstraints(max_arg_length=10)
        violations = checker.check("write_file", {"content": "x" * 100}, constraints=constraints)
        assert any(v.constraint == "oversized_argument" for v in violations)

    def test_arg_denylist(self, checker):
        constraints = ToolConstraints(arg_denylist=["DROP TABLE"])
        violations = checker.check(
            "run_query", {"query": "SELECT 1; DROP TABLE users;"}, constraints=constraints
        )
        assert any(v.constraint == "arg_denylist" for v in violations)

    def test_clean_call_with_constraints_passes(self, checker):
        constraints = ToolConstraints(path_allowlist=["/tmp/**"], max_arg_length=1000)
        violations = checker.check(
            "write_file", {"path": "/tmp/ok.txt", "content": "hello"}, constraints=constraints
        )
        assert violations == []


# ---------------------------------------------------------------------------
# ToolPolicyEngine.check_arguments
# ---------------------------------------------------------------------------


class TestPolicyEngineCheckArguments:
    def test_loads_tool_constraints_from_yaml(self, engine):
        violations = engine.check_arguments("writer", "write_file", {"path": "/home/x"})
        assert any(v.constraint == "path_not_allowed" for v in violations)

    def test_path_denylist_from_yaml(self, engine):
        violations = engine.check_arguments("writer", "read_file", {"path": "/etc/crontab"})
        assert any(v.constraint == "path_denied" for v in violations)

    def test_url_denylist_from_yaml(self, engine):
        violations = engine.check_arguments(
            "writer", "fetch", {"url": "http://169.254.169.254/latest/meta-data"}
        )
        assert any(v.constraint == "url_denied" for v in violations)

    def test_builtin_detectors_run_without_per_tool_config(self, engine):
        violations = engine.check_arguments("writer", "write_file", {"path": "/etc/shadow"})
        assert any(v.constraint == "sensitive_path" for v in violations)

    def test_no_policy_still_runs_builtins(self):
        engine = ToolPolicyEngine(policy_path=None)
        violations = engine.check_arguments("anyone", "read_file", {"path": "../../etc/passwd"})
        assert any(v.constraint == "path_traversal" for v in violations)

    def test_clean_call_returns_empty(self, engine):
        violations = engine.check_arguments("writer", "write_file", {"path": "/tmp/ok.txt"})
        assert violations == []

    def test_backward_compat_check_unchanged(self, engine):
        result = engine.check("writer", "write_file")
        assert result.allowed is True
        result = engine.check("writer", "execute_code")
        assert result.allowed is False


# ---------------------------------------------------------------------------
# MCP interceptor — argument constraint enforcement
# ---------------------------------------------------------------------------


class TestMCPInterceptorConstraints:
    def test_enforce_mode_blocks_constraint_violation(self, policy_file):
        bus = EventBus()
        interceptor = MCPInterceptor(
            agent_id="writer",
            session_id="sess-1",
            bus=bus,
            policy_path=policy_file,
            mode="enforce",
        )
        request = MCPRequest(
            id=1,
            method="tools/call",
            params={"name": "write_file", "arguments": {"path": "/home/outside.txt"}},
        )
        result = interceptor.intercept(request)

        assert result.allowed is False
        assert any(e.event_type == "policy_violation" for e in result.events)
        violation_event = next(e for e in result.events if e.event_type == "policy_violation")
        assert any(f.startswith("constraint:") for f in violation_event.flags)

    def test_observe_mode_logs_but_allows(self, policy_file):
        bus = EventBus()
        interceptor = MCPInterceptor(
            agent_id="writer",
            session_id="sess-1",
            bus=bus,
            policy_path=policy_file,
            mode="observe",
        )
        request = MCPRequest(
            id=1,
            method="tools/call",
            params={"name": "write_file", "arguments": {"path": "/home/outside.txt"}},
        )
        result = interceptor.intercept(request)

        assert result.allowed is True
        assert any(e.event_type == "policy_violation" for e in result.events)

    def test_clean_call_passes(self, policy_file):
        bus = EventBus()
        interceptor = MCPInterceptor(
            agent_id="writer",
            session_id="sess-1",
            bus=bus,
            policy_path=policy_file,
            mode="enforce",
        )
        request = MCPRequest(
            id=1,
            method="tools/call",
            params={"name": "write_file", "arguments": {"path": "/tmp/ok.txt"}},
        )
        result = interceptor.intercept(request)

        assert result.allowed is True
        assert not any(f.startswith("constraint:") for e in result.events for f in e.flags)
