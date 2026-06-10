# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Core instrumentation:** `GuardedClient` / `AsyncGuardedClient` (drop-in wrappers for the
  Anthropic SDK, sync + async + streaming), `AgentMoatCallback` for LangGraph, and a unified
  `SecurityEvent` bus with SQLite store and hash-chained JSONL audit logging.
- **Detection engine:** prompt-injection detector (regex + optional embeddings), tool-policy
  engine (allow/deny + rate limits), and a cross-agent trust scorer.
- **MCP transparent proxy:** stdio and SSE proxy that intercepts and enforces on any
  MCP-compatible tool server with no agent code changes.
- **Argument-level tool constraints:** path-traversal, SSRF, shell-metacharacter, and
  sensitive-path detection, plus configurable per-tool path/URL allow- and deny-lists.
- **Tamper-evident audit log:** SHA-256 hash chaining with `agentmoat audit verify`.
- **Response controls:** `observe` / `enforce` / `interactive` modes, human-in-the-loop approval
  gate, and a process-wide kill switch (with HTTP control endpoints).
- **OpenAI SDK support:** `GuardedOpenAI` / `AsyncGuardedOpenAI` reusing the same engine.
- **Audit API + dashboard:** FastAPI service (authenticated) and a React event/timeline UI.

### Security / hardening
- Reliable event persistence in sync and async contexts (background persistence worker).
- Fail-closed on internal engine errors in enforce/interactive mode; fail-open-but-logged in
  observe mode.
- API-key authentication on the audit API.
- Subprocess stderr draining in the MCP stdio client (deadlock fix).
- Bounded, TTL/LRU in-memory state for trust and rate-limit data.
- Secret/PII redaction before persistence.
- Timezone-aware timestamps throughout.

[Unreleased]: https://github.com/Shashank-016/agentmoat/commits/master
