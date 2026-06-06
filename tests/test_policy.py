"""Tests for ToolPolicyEngine — allow/deny rules and rate limiting."""

import pytest
import tempfile
import os
from agentguard.engine.policy import ToolPolicyEngine

POLICY_YAML = """
version: "1"
agents:
  researcher:
    allowed_tools:
      - web_search
      - read_file
    denied_tools:
      - write_file
      - execute_code
    rate_limits:
      web_search: 3/minute

  writer:
    allowed_tools:
      - write_file
      - read_file
    denied_tools:
      - web_search
      - execute_code
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
def no_policy_engine() -> ToolPolicyEngine:
    return ToolPolicyEngine(policy_path=None)


# ---------------------------------------------------------------------------
# Allow / deny
# ---------------------------------------------------------------------------

class TestAllowDeny:
    def test_researcher_allowed_tool(self, engine):
        result = engine.check("researcher", "web_search")
        assert result.allowed is True

    def test_researcher_allowed_read_file(self, engine):
        result = engine.check("researcher", "read_file")
        assert result.allowed is True

    def test_researcher_denied_write_file(self, engine):
        result = engine.check("researcher", "write_file")
        assert result.allowed is False
        assert result.rule_name == "policy:tool_denied"

    def test_researcher_denied_execute_code(self, engine):
        result = engine.check("researcher", "execute_code")
        assert result.allowed is False

    def test_researcher_unknown_tool_not_in_allowlist(self, engine):
        result = engine.check("researcher", "some_other_tool")
        assert result.allowed is False
        assert result.rule_name == "policy:tool_not_allowed"

    def test_writer_allowed_write_file(self, engine):
        result = engine.check("writer", "write_file")
        assert result.allowed is True

    def test_writer_denied_web_search(self, engine):
        result = engine.check("writer", "web_search")
        assert result.allowed is False

    def test_deny_takes_precedence_over_allow(self, engine):
        """If a tool appears in both lists, deny wins."""
        # write_file is denied for researcher even if we hypothetically had it in allowed.
        result = engine.check("researcher", "write_file")
        assert result.allowed is False

    def test_unknown_agent_permissive(self, engine):
        """Agents without a policy entry get unrestricted access."""
        result = engine.check("unknown_agent", "any_tool")
        assert result.allowed is True
        assert result.rule_name == "policy:no_policy"


# ---------------------------------------------------------------------------
# No policy file
# ---------------------------------------------------------------------------

class TestNoPolicyFile:
    def test_all_tools_allowed(self, no_policy_engine):
        for tool in ["write_file", "execute_code", "web_search", "anything"]:
            result = no_policy_engine.check("any_agent", tool)
            assert result.allowed is True

    def test_list_agents_empty(self, no_policy_engine):
        assert no_policy_engine.list_agents() == []


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_within_rate_limit(self, engine):
        for _ in range(3):
            result = engine.check("researcher", "web_search")
            assert result.allowed is True

    def test_exceeds_rate_limit(self, engine):
        for _ in range(3):
            engine.check("researcher", "web_search")
        # 4th call should be blocked.
        result = engine.check("researcher", "web_search")
        assert result.allowed is False
        assert result.rule_name == "policy:rate_limit_exceeded"

    def test_rate_limit_not_applied_to_writer(self, engine):
        # writer has no rate limit on read_file.
        for _ in range(10):
            result = engine.check("writer", "read_file")
            assert result.allowed is True


# ---------------------------------------------------------------------------
# Sensitive tool detection
# ---------------------------------------------------------------------------

class TestSensitiveTool:
    def test_write_file_is_sensitive(self, engine):
        assert engine.is_sensitive("write_file") is True

    def test_execute_code_is_sensitive(self, engine):
        assert engine.is_sensitive("execute_code") is True

    def test_web_search_not_sensitive(self, engine):
        assert engine.is_sensitive("web_search") is False

    def test_read_file_not_sensitive(self, engine):
        assert engine.is_sensitive("read_file") is False


# ---------------------------------------------------------------------------
# PolicyResult fields
# ---------------------------------------------------------------------------

class TestPolicyResult:
    def test_result_has_reason(self, engine):
        result = engine.check("researcher", "write_file")
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_result_has_rule_name(self, engine):
        result = engine.check("researcher", "write_file")
        assert result.rule_name.startswith("policy:")
