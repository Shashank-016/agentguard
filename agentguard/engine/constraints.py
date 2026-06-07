"""Argument-level tool constraint checking — path, URL, and pattern-based guards.

Tool *names* are not the whole story: an agent permitted to call ``write_file``
can still be driven to write outside its sandbox, and an agent permitted to
``fetch`` can be driven to hit the cloud metadata endpoint. This module
inspects tool call *arguments* against a library of built-in dangerous-pattern
detectors plus optional per-tool constraints declared in the policy file.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from .policy import ToolConstraints

logger = logging.getLogger(__name__)

_MAX_VALUE_CHARS = 200

# ---------------------------------------------------------------------------
# Violation model
# ---------------------------------------------------------------------------


@dataclass
class ConstraintViolation:
    """A single argument constraint violation found in a tool call.

    Attributes
    ----------
    constraint:
        Stable identifier of the violated constraint, e.g. ``"path_denylist"``.
    argument:
        Name of the argument that triggered the violation, e.g. ``"path"``.
    value:
        The offending value, truncated to :data:`_MAX_VALUE_CHARS` characters.
    detail:
        Human-readable explanation of why this is a violation.
    """

    constraint: str
    argument: str
    value: str
    detail: str

    @property
    def flag(self) -> str:
        """Return the event-flag identifier for this violation, e.g.
        ``"constraint:ssrf_target"``."""
        return f"constraint:{self.constraint}"


# ---------------------------------------------------------------------------
# Built-in dangerous-pattern detection
# ---------------------------------------------------------------------------

_PATH_TRAVERSAL_RE = re.compile(r"\.\.[/\\]|%2e%2e", re.IGNORECASE)

_SSRF_HOSTS = (
    "169.254.169.254",
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "metadata.google.internal",
)
_SSRF_HOST_PREFIXES = ("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.")
_SSRF_HOST_PREFIXES += tuple(f"172.{n}." for n in range(16, 32))

_SHELL_METACHAR_RE = re.compile(r"[;|`<>]|&&|\$\(")
_SHELL_TOOL_KEYWORDS = ("exec", "shell", "command", "run", "bash", "sh")

_SENSITIVE_PATH_RE = re.compile(
    r"/etc/|/root/|~/\.ssh|id_rsa|\.env|credentials|/proc/", re.IGNORECASE
)

_PATH_NAME_HINTS = ("path", "file", "dir", "dest", "src", "target")
_PATH_VALUE_PREFIXES = ("/", "./", "../", "~")
_FS_PATH_RE = re.compile(r"^[a-zA-Z]:[\\/]|^[\\/]|^\.{1,2}[\\/]")

_URL_NAME_HINTS = ("url", "uri", "endpoint", "host")
_URL_VALUE_PREFIXES = ("http://", "https://", "ftp://")


def _truncate(value: str) -> str:
    if len(value) > _MAX_VALUE_CHARS:
        return value[:_MAX_VALUE_CHARS] + "…"
    return value


def _is_path_like(name: str, value: str) -> bool:
    lname = name.lower()
    if any(hint in lname for hint in _PATH_NAME_HINTS):
        return True
    if value.startswith(_PATH_VALUE_PREFIXES):
        return True
    return bool(_FS_PATH_RE.match(value))


def _is_url_like(name: str, value: str) -> bool:
    lname = name.lower()
    if any(hint in lname for hint in _URL_NAME_HINTS):
        return True
    return value.startswith(_URL_VALUE_PREFIXES)


def _extract_host(value: str) -> str | None:
    """Extract a hostname from a URL string or return the bare value as a host candidate."""
    if "://" in value:
        try:
            return urlparse(value).hostname
        except ValueError:
            return None
    # Bare host or host:port.
    return value.split("/")[0].split(":")[0]


def _matches_ssrf_host(host: str | None) -> bool:
    if not host:
        return False
    host = host.lower()
    if host in _SSRF_HOSTS:
        return True
    return any(host.startswith(prefix) for prefix in _SSRF_HOST_PREFIXES)


def _iter_string_arguments(arguments: dict[str, Any]) -> list[tuple[str, str]]:
    """Flatten a tool-call argument dict into ``(name, value)`` string pairs."""
    pairs: list[tuple[str, str]] = []
    for name, value in arguments.items():
        if isinstance(value, str):
            pairs.append((name, value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    pairs.append((name, item))
    return pairs


def _matches_any_glob(value: str, patterns: list[str]) -> bool:
    return any(fnmatch(value, pattern) for pattern in patterns)


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class ArgumentConstraintChecker:
    """Evaluates tool-call arguments against built-in and per-tool constraints.

    Built-in detectors (path traversal, SSRF targets, shell metacharacters,
    sensitive path access) run on every call when ``builtins_enabled`` is
    True. Per-tool constraints — path/URL allow- and deny-lists, max argument
    length, and argument substring deny-lists — are supplied via a
    :class:`~agentguard.engine.policy.ToolConstraints` instance loaded from
    the policy file.

    Parameters
    ----------
    builtins_enabled:
        Whether to run the built-in dangerous-pattern detectors. Default True.
    """

    def __init__(self, builtins_enabled: bool = True) -> None:
        self._builtins_enabled = builtins_enabled

    def check(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        constraints: ToolConstraints | None = None,
    ) -> list[ConstraintViolation]:
        """Return all constraint violations for a tool call. Empty list = clean."""
        violations: list[ConstraintViolation] = []
        string_args = _iter_string_arguments(arguments)

        if self._builtins_enabled:
            violations.extend(self._check_builtins(tool_name, string_args))

        if constraints is not None:
            violations.extend(self._check_configured(string_args, constraints))

        return violations

    # ------------------------------------------------------------------
    # Built-in detectors
    # ------------------------------------------------------------------

    def _check_builtins(
        self, tool_name: str, string_args: list[tuple[str, str]]
    ) -> list[ConstraintViolation]:
        violations: list[ConstraintViolation] = []
        is_exec_tool = any(kw in tool_name.lower() for kw in _SHELL_TOOL_KEYWORDS)

        for name, value in string_args:
            if _PATH_TRAVERSAL_RE.search(value):
                violations.append(
                    ConstraintViolation(
                        constraint="path_traversal",
                        argument=name,
                        value=_truncate(value),
                        detail=f"Argument '{name}' contains a path traversal sequence",
                    )
                )

            if _SENSITIVE_PATH_RE.search(value):
                violations.append(
                    ConstraintViolation(
                        constraint="sensitive_path",
                        argument=name,
                        value=_truncate(value),
                        detail=f"Argument '{name}' references a sensitive system path",
                    )
                )

            if _is_url_like(name, value) and _matches_ssrf_host(_extract_host(value)):
                violations.append(
                    ConstraintViolation(
                        constraint="ssrf_target",
                        argument=name,
                        value=_truncate(value),
                        detail=(
                            f"Argument '{name}' targets a host commonly used for SSRF "
                            "(cloud metadata, loopback, or private network range)"
                        ),
                    )
                )

            if is_exec_tool and _SHELL_METACHAR_RE.search(value):
                violations.append(
                    ConstraintViolation(
                        constraint="shell_metachar",
                        argument=name,
                        value=_truncate(value),
                        detail=(
                            f"Argument '{name}' contains shell metacharacters and "
                            f"'{tool_name}' looks like a command-execution tool"
                        ),
                    )
                )

        return violations

    # ------------------------------------------------------------------
    # Per-tool configured constraints
    # ------------------------------------------------------------------

    def _check_configured(
        self,
        string_args: list[tuple[str, str]],
        constraints: ToolConstraints,
    ) -> list[ConstraintViolation]:
        violations: list[ConstraintViolation] = []

        for name, value in string_args:
            if constraints.max_arg_length is not None and len(value) > constraints.max_arg_length:
                violations.append(
                    ConstraintViolation(
                        constraint="oversized_argument",
                        argument=name,
                        value=_truncate(value),
                        detail=(
                            f"Argument '{name}' is {len(value)} chars, "
                            f"exceeding the limit of {constraints.max_arg_length}"
                        ),
                    )
                )

            for needle in constraints.arg_denylist:
                if needle in value:
                    violations.append(
                        ConstraintViolation(
                            constraint="arg_denylist",
                            argument=name,
                            value=_truncate(value),
                            detail=f"Argument '{name}' contains denied substring '{needle}'",
                        )
                    )

            is_path = _is_path_like(name, value)
            is_url = _is_url_like(name, value)

            if is_path:
                if constraints.path_denylist and _matches_any_glob(
                    value, constraints.path_denylist
                ):
                    violations.append(
                        ConstraintViolation(
                            constraint="path_denied",
                            argument=name,
                            value=_truncate(value),
                            detail=f"Argument '{name}' matches a denied path pattern",
                        )
                    )
                if constraints.path_allowlist and not _matches_any_glob(
                    value, constraints.path_allowlist
                ):
                    violations.append(
                        ConstraintViolation(
                            constraint="path_not_allowed",
                            argument=name,
                            value=_truncate(value),
                            detail=(
                                f"Argument '{name}' does not match any allowed path pattern: "
                                f"{constraints.path_allowlist}"
                            ),
                        )
                    )

            if is_url:
                host = _extract_host(value)
                if (
                    constraints.url_denylist
                    and host
                    and _matches_any_glob(host, constraints.url_denylist)
                ):
                    violations.append(
                        ConstraintViolation(
                            constraint="url_denied",
                            argument=name,
                            value=_truncate(value),
                            detail=f"Argument '{name}' targets a denied host '{host}'",
                        )
                    )
                if (
                    constraints.url_allowlist
                    and host
                    and not _matches_any_glob(host, constraints.url_allowlist)
                ):
                    violations.append(
                        ConstraintViolation(
                            constraint="url_not_allowed",
                            argument=name,
                            value=_truncate(value),
                            detail=(
                                f"Argument '{name}' targets host '{host}', which does not "
                                f"match any allowed host pattern: {constraints.url_allowlist}"
                            ),
                        )
                    )

        return violations
