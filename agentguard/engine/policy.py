"""Tool policy engine — YAML-driven allow/deny rules with rate limiting."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from ._state import DEFAULT_MAX_ENTRIES, DEFAULT_TTL_SECONDS, BoundedStateStore
from .constraints import ArgumentConstraintChecker, ConstraintViolation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyResult:
    """Result of a policy check.

    Attributes
    ----------
    allowed:
        Whether the tool call is permitted.
    reason:
        Human-readable explanation of the decision.
    rule_name:
        Stable identifier of the matched rule, e.g.
        ``"policy:tool_not_allowed"``.
    """

    allowed: bool
    reason: str
    rule_name: str


@dataclass
class ToolConstraints:
    """Per-tool argument constraints, declared under ``tool_constraints`` in the policy file.

    Attributes
    ----------
    path_allowlist:
        Glob patterns (supporting ``**``); a path-like argument that matches
        none of these patterns is a ``path_not_allowed`` violation.
    path_denylist:
        Glob patterns; a path-like argument matching any of these is a
        ``path_denied`` violation.
    url_allowlist:
        Glob patterns matched against the host of a URL-like argument; a
        non-matching host is a ``url_not_allowed`` violation.
    url_denylist:
        Glob patterns matched against the host of a URL-like argument; a
        matching host is a ``url_denied`` violation.
    max_arg_length:
        Maximum allowed length, in characters, for any string argument value.
        ``None`` disables the check. Violations are flagged ``oversized_argument``.
    arg_denylist:
        Substrings that must not appear in any argument value. A match is an
        ``arg_denylist`` violation.
    """

    path_allowlist: list[str] = field(default_factory=list)
    path_denylist: list[str] = field(default_factory=list)
    url_allowlist: list[str] = field(default_factory=list)
    url_denylist: list[str] = field(default_factory=list)
    max_arg_length: Optional[int] = None
    arg_denylist: list[str] = field(default_factory=list)


@dataclass
class _AgentPolicy:
    allowed_tools: Optional[list[str]]  # None = wildcard
    denied_tools: list[str]
    rate_limits: dict[str, tuple[int, int]]  # tool → (count, window_seconds)
    tool_constraints: dict[str, ToolConstraints] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rate limiter (token bucket via sliding window)
# ---------------------------------------------------------------------------

class _SlidingWindowCounter:
    """Tracks call counts in a sliding time window per (agent_id, tool_name).

    Bounded by :class:`~agentguard.engine._state.BoundedStateStore` (TTL +
    LRU eviction) so a long-running proxy serving many distinct agents/tools
    doesn't accumulate one deque per pair forever.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        # (agent_id, tool_name) → deque of timestamps
        self._windows: BoundedStateStore[tuple[str, str], "deque[float]"] = BoundedStateStore(
            max_entries=max_entries, ttl_seconds=ttl_seconds
        )

    def is_allowed(self, agent_id: str, tool_name: str, limit: int, window_seconds: int) -> bool:
        key = (agent_id, tool_name)
        now = time.monotonic()
        window = self._windows.get_or_create(key, deque)
        # Evict timestamps outside the window.
        cutoff = now - window_seconds
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= limit:
            return False
        window.append(now)
        return True


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_SENSITIVE_TOOLS = frozenset(
    {
        "write_file",
        "file_write",
        "execute_code",
        "run_command",
        "shell",
        "bash",
        "send_email",
        "send_message",
        "http_post",
        "delete_file",
        "rm",
    }
)


class ToolPolicyEngine:
    """Evaluates tool calls against a YAML policy file.

    If no policy file is provided, all tools are permitted (observe-only
    behaviour). Agents not named in the policy fall back to a default policy
    that allows everything — add an explicit ``default`` agent block to lock
    this down.

    Policy file format::

        version: "1"
        agents:
          researcher:
            allowed_tools: [web_search, read_file]
            denied_tools:  [write_file, execute_code]
            rate_limits:
              web_search: 10/minute

    Parameters
    ----------
    policy_path:
        Path to a YAML policy file. May be ``None`` (no restrictions).
    max_rate_limit_entries:
        Maximum number of distinct (agent_id, tool_name) rate-limit windows
        to retain at once — bounds memory for long-running proxies.
    rate_limit_ttl_seconds:
        Rate-limit windows untouched for longer than this are evicted lazily.
    """

    def __init__(
        self,
        policy_path: Optional[str] = None,
        max_rate_limit_entries: int = DEFAULT_MAX_ENTRIES,
        rate_limit_ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._policies: dict[str, _AgentPolicy] = {}
        self._rate_counter = _SlidingWindowCounter(
            max_entries=max_rate_limit_entries, ttl_seconds=rate_limit_ttl_seconds
        )
        self._constraint_checker = ArgumentConstraintChecker()
        if policy_path:
            self._load(policy_path)

    def _load(self, path: str) -> None:
        data = yaml.safe_load(Path(path).read_text())
        agents = data.get("agents", {})
        for agent_id, cfg in agents.items():
            raw_limits = cfg.get("rate_limits", {})
            parsed_limits: dict[str, tuple[int, int]] = {}
            for tool, spec in raw_limits.items():
                parsed_limits[tool] = _parse_rate(spec)

            raw_constraints = cfg.get("tool_constraints", {})
            parsed_constraints: dict[str, ToolConstraints] = {}
            for tool, spec in raw_constraints.items():
                spec = spec or {}
                parsed_constraints[tool] = ToolConstraints(
                    path_allowlist=list(spec.get("path_allowlist", [])),
                    path_denylist=list(spec.get("path_denylist", [])),
                    url_allowlist=list(spec.get("url_allowlist", [])),
                    url_denylist=list(spec.get("url_denylist", [])),
                    max_arg_length=spec.get("max_arg_length"),
                    arg_denylist=list(spec.get("arg_denylist", [])),
                )

            self._policies[agent_id] = _AgentPolicy(
                allowed_tools=cfg.get("allowed_tools"),
                denied_tools=cfg.get("denied_tools", []),
                rate_limits=parsed_limits,
                tool_constraints=parsed_constraints,
            )
        logger.info(
            "ToolPolicyEngine loaded policies for agents: %s",
            list(self._policies.keys()),
        )

    def check(self, agent_id: str, tool_name: str) -> PolicyResult:
        """Check whether ``agent_id`` may call ``tool_name``.

        Returns a :class:`PolicyResult` with ``allowed=True/False`` and a
        descriptive reason. Never raises.
        """
        policy = self._policies.get(agent_id)

        if policy is None:
            # No named policy → permissive default.
            return PolicyResult(
                allowed=True,
                reason=f"No policy defined for agent '{agent_id}' — all tools allowed",
                rule_name="policy:no_policy",
            )

        # Deny list is checked first (explicit deny beats allow).
        if tool_name in policy.denied_tools:
            return PolicyResult(
                allowed=False,
                reason=f"'{tool_name}' is in the denied_tools list for agent '{agent_id}'",
                rule_name="policy:tool_denied",
            )

        # Allow list (None = wildcard).
        if policy.allowed_tools is not None and tool_name not in policy.allowed_tools:
            return PolicyResult(
                allowed=False,
                reason=(
                    f"'{tool_name}' is not in the allowed_tools list for agent '{agent_id}'. "
                    f"Allowed: {policy.allowed_tools}"
                ),
                rule_name="policy:tool_not_allowed",
            )

        # Rate limit check.
        if tool_name in policy.rate_limits:
            limit, window = policy.rate_limits[tool_name]
            if not self._rate_counter.is_allowed(agent_id, tool_name, limit, window):
                return PolicyResult(
                    allowed=False,
                    reason=(
                        f"Rate limit exceeded for '{tool_name}' on agent '{agent_id}' "
                        f"({limit} calls per {window}s)"
                    ),
                    rule_name="policy:rate_limit_exceeded",
                )

        return PolicyResult(
            allowed=True,
            reason=f"'{tool_name}' is permitted for agent '{agent_id}'",
            rule_name="policy:allowed",
        )

    def check_arguments(
        self, agent_id: str, tool_name: str, arguments: dict
    ) -> list[ConstraintViolation]:
        """Run argument-level constraint checks for a tool call.

        Combines built-in dangerous-pattern detectors with any per-tool
        constraints defined in the policy file. Returns all violations.
        Never raises.
        """
        policy = self._policies.get(agent_id)
        constraints = policy.tool_constraints.get(tool_name) if policy else None
        return self._constraint_checker.check(tool_name, arguments, constraints=constraints)

    def is_sensitive(self, tool_name: str) -> bool:
        """Return True if ``tool_name`` is considered a high-impact operation."""
        return tool_name.lower() in _SENSITIVE_TOOLS or any(
            kw in tool_name.lower() for kw in ("write", "exec", "delete", "send", "shell")
        )

    def list_agents(self) -> list[str]:
        """Return agent IDs with explicit policies."""
        return list(self._policies.keys())


def _parse_rate(spec: str) -> tuple[int, int]:
    """Parse a rate limit string like ``'10/minute'`` → ``(10, 60)``."""
    _WINDOWS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
    parts = str(spec).split("/")
    count = int(parts[0].strip())
    unit = parts[1].strip().lower() if len(parts) > 1 else "minute"
    return count, _WINDOWS.get(unit, 60)
