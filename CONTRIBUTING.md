# Contributing to AgentGuard

Thanks for your interest in contributing. AgentGuard is a security layer for AI agents, and
contributions of all kinds are welcome — bug reports, detection patterns, framework adapters,
docs, and tests.

## Development setup

```bash
git clone https://github.com/Shashank-016/agentguard
cd agentguard
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,langgraph,openai]"
```

## Common tasks

```bash
make test        # run the full test suite
make lint        # ruff
make type        # mypy
make format      # ruff format
make demo        # run the MCP proxy attack demo
make dashboard   # run the React dashboard (needs the API running)
```

(or run the underlying commands directly: `pytest`, `ruff check .`, `mypy agentguard`)

## Architecture

`CLAUDE.md` contains the module map and the rationale behind key design decisions (the background
persistence worker, fail-closed semantics, trust decay, the MCP proxy transport). Read it before
making structural changes.

## Guidelines

- Match existing conventions: Pydantic v2 models, `from __future__ import annotations`,
  module-level `logging`, full type hints, docstrings on public symbols, no `print()` in library
  code.
- Add or update tests for any behavior change. The suite must stay green.
- For new detection patterns or constraints, include both a true-positive and a false-positive
  test.
- Keep commits focused and write clear messages (`feat:`, `fix:`, `docs:`, `test:`).
- For security-sensitive changes, note the threat-model implications in the PR description.

## Reporting issues

Use the issue templates. For security vulnerabilities, follow `SECURITY.md` (do not open a public
issue).
