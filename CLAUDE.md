# AgentMoat — Claude Code Session Guide

AgentMoat is a security observability layer for AI agents that detects prompt injection, tool policy violations, and trust degradation in real-time. It instruments both LangGraph-orchestrated agents and direct Anthropic SDK usage via a drop-in client wrapper and callback handler.

## How to Run

```bash
# Install in editable mode (installs core + langgraph extras)
pip install -e ".[langgraph]"

# Start the audit API (SQLite by default)
uvicorn api.main:app --reload
# → http://localhost:8000/docs

# Start the React dashboard
cd dashboard && npm install && npm run dev
# → http://localhost:5173

# Run the LangGraph demo (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-... python examples/langgraph_demo.py

# Run the sync SDK demo
ANTHROPIC_API_KEY=sk-... python examples/sdk_demo.py

# Run the async SDK demo (AsyncGuardedClient)
ANTHROPIC_API_KEY=sk-... python examples/async_sdk_demo.py

# Run the OpenAI SDK demo (GuardedOpenAI — requires the [openai] extra)
OPENAI_API_KEY=sk-... python examples/openai_sdk_demo.py

# Run the async OpenAI SDK demo (AsyncGuardedOpenAI)
OPENAI_API_KEY=sk-... python examples/async_openai_sdk_demo.py

# Run the MCP proxy demo (requires Node.js + npx for the real MCP server)
python examples/mcp_proxy_demo.py

# Run the AgentMoat MCP proxy CLI (stdio mode)
agentmoat mcp proxy stdio \
    --upstream-cmd "npx -y @modelcontextprotocol/server-filesystem /tmp" \
    --agent-id researcher \
    --mode enforce

# Run tests
pytest
```

## MCP Proxy Architecture

```
Agent (any framework)
    ↓  MCP protocol (stdio or SSE)
AgentMoat MCP Proxy  ←→  MCPInterceptor  ←→  EventBus
    ↓  MCP protocol (forwarded if allowed)
Real MCP Server (filesystem, web, database, etc.)
```

The proxy is framework-agnostic — it operates at the transport layer, not the application
layer, so it works with LangGraph, raw SDK, LlamaIndex, CrewAI, or anything that speaks MCP.

**StdioUpstreamClient uses a background reader task.**
JSON-RPC responses arrive asynchronously on the upstream stdout. A background `asyncio.Task`
reads stdout continuously and resolves pending `asyncio.Future` objects matched by request ID.
This handles out-of-order responses (though most MCP servers are sequential).

**MCPInterceptor.intercept() is synchronous.**
Same rationale as EventBus.emit() — it keeps the interceptor usable from both sync and
async call sites without the overhead of an event loop requirement.

## Key Architectural Decisions

**EventBus.emit() is synchronous.**
LangGraph callbacks (BaseCallbackHandler) and the Anthropic SDK intercepts run in synchronous contexts with no event loop guarantee. Making emit() synchronous means it's safe to call from anywhere. Async persistence is scheduled via `asyncio.create_task()` only when a loop is running — otherwise silently skipped. This is an intentional trade-off: reliability of emission over guaranteed durability.

**GuardedClient proxies `__getattr__`.**
`GuardedClient` only intercepts `messages` (the property). Everything else — `beta`, `files`, `with_raw_response`, API key access — is transparently forwarded to `self._client` via `__getattr__`. This makes it a true drop-in: the user can pass a `GuardedClient` anywhere an `anthropic.Anthropic` is accepted.

**Trust scores degrade multiplicatively, not additively.**
A session reading two external documents doesn't just accumulate risk linearly — it compounds. `score = score * EXTERNAL (0.3)` on each external content event. This models real-world provenance accurately: every hop through untrusted data multiplies the uncertainty. Resetting to 0 on injection (rather than multiplicative) reflects that injection is a binary, qualitative event.

**Observe-don't-block is the default.**
In `mode="observe"` (default), AgentMoat detects and logs but never interrupts the agent. This is intentional: a security tool that breaks production agents won't get deployed. Users opt into blocking via `mode="enforce"` once they've validated the detection quality in their environment.

**Rule-based detection runs before embedding-based.**
Regex patterns have zero latency and high precision on known attack patterns. Embeddings catch paraphrased variants the regex misses, but require loading a ~80MB model. Embeddings are opt-in via `use_embeddings=True`.

**Policy deny-list is checked before allow-list.**
Explicit denies always beat explicit allows. If a tool appears in both (misconfiguration), it's blocked. This is the safer default for a security tool.

**`control.py` never imports from `client.py`.**
`AgentMoatException`/`AgentMoatKilled` live in `client.py`; `ApprovalGate`/`KillSwitch` live in `control.py`. Importing the exceptions into `control.py` would create a cycle (`client.py` needs `ApprovalGate`/`KillSwitch` to implement `mode="interactive"`). `KillSwitch.is_killed()` returns a plain `bool` — the actual `raise AgentMoatKilled(...)` happens at each call site (`client.py`, `async_client.py`, `mcp/interceptor.py`), which already import the exception classes.

**Interactive mode gives humans finer control than enforce mode's blanket policy.**
`trust_flag` events are warning-only and never hard-block in `mode="enforce"` (a low trust score alone shouldn't halt an agent). But in `mode="interactive"`, they still route through the `ApprovalGate` — an explicit human "deny" blocks the call. This asymmetry is intentional: enforce mode encodes a fixed policy ahead of time, interactive mode lets a human apply judgment in the moment, including to things that wouldn't trigger a hard block on their own.

**`GuardedOpenAI`/`AsyncGuardedOpenAI` reuse `client.py`'s/`async_client.py`'s violation-dispatch helpers.**
`_dispatch_violation`/`_raise_if_killed` (and their `_async` counterparts) are duck-typed against shared attribute names (`session_id`, `agent_id`, `mode`, `_bus`, `_approval_gate`, `_kill_switch`) rather than `GuardedClient`/`AsyncGuardedClient` specifically — their `guard:` parameter is typed `Any`. `agentmoat/openai_client.py` imports and reuses them directly instead of re-implementing the observe/enforce/interactive branching and kill-switch checks a third time. The OpenAI-shape differences (nested `{"type": "function", "function": {...}}` tool definitions, JSON-string `function.arguments` instead of structured `input` dicts) are bridged locally via `_tool_def_name()`/`_parse_tool_arguments()`.

**`KillSwitch` is a process-wide singleton by default.**
`get_default_kill_switch()` returns one shared instance so a single `kill_all()` — whether called programmatically, via the `/control/kill-all` API route, or from a different `GuardedClient`/`MCPInterceptor` — halts every in-process session immediately. Pass an explicit `kill_switch=` to isolate a client from the shared switch (e.g. in tests). Multi-process deployments need a shared backing store (Redis, per the roadmap) for one trip to halt every worker.

**`EventBus` persists durably via a background daemon thread, and must be drained on shutdown.**
`emit()`/`emit_async()` schedule persistence on a lazily-started worker thread (`_ensure_worker`) that owns its own event loop and writes through `asyncio.run_coroutine_threadsafe`. This keeps emission non-blocking and safe from both sync and async call sites. Because the thread is a daemon, anything emitted just before process exit can be lost unless the bus is drained first — call `bus.flush()` to wait for in-flight persistence (e.g. mid-run, after a burst of events) or `bus.close()` at shutdown (flushes, then stops the loop and joins the thread). `agentmoat/cli.py`'s `_run_stdio`/`_run_sse` call `bus.close()` in their `finally` blocks alongside `upstream.stop()`.

**Engine errors fail closed in `enforce`/`interactive` modes, fail open in `observe`.**
If the security engine itself raises mid-evaluation (regex blowup, malformed policy YAML, embedding model error, etc.), every call path — `GuardedClient`, `AsyncGuardedClient`, `GuardedOpenAI`/`AsyncGuardedOpenAI`, and `MCPInterceptor` — catches it, emits a critical `engine_error` event (flag `engine:internal_error`), and then: in `enforce`/`interactive` mode, blocks the call (raises `AgentMoatException` for SDK clients, returns `allowed=False` with `block_code=AGENTMOAT_ENGINE_ERROR` for the MCP interceptor) rather than letting a broken detector silently wave everything through; in `observe` mode it logs and lets the call proceed, consistent with observe's "never interrupt" contract. Per-call evaluation logic is factored into standalone `_evaluate_*_sync`/`_evaluate_*_async` helpers specifically so the whole evaluation can be wrapped in one `try/except`.

**The audit API's optional `AGENTMOAT_API_KEY` is backward-compatible by design.**
`require_api_key` (in `api/main.py`, wired as a router-level dependency on `control`/`events`/`sessions`) only enforces a key if `AGENTMOAT_API_KEY` is set — accepting it via `X-API-Key` or `Authorization: Bearer <token>`. If unset, every request is allowed through exactly as before this check existed, but a one-time warning is logged so an operator notices the gap rather than silently running an open API. `/health` is always open (it's a liveness probe, not a security boundary).

**`StdioUpstreamClient` drains the upstream subprocess's stderr continuously.**
A background `_drain_stderr` task reads `process.stderr` line-by-line and logs each line at `DEBUG`. Without this, stderr output from the wrapped MCP server fills its OS pipe buffer once full, and the subprocess blocks on the next write — silently wedging the whole proxy. The task is created in `start()` and cancelled/awaited in `stop()` alongside the stdout `_read_loop` task; like `_read_loop`, it swallows `asyncio.CancelledError` and finishes normally rather than ending up in the asyncio-"cancelled" state, so tests assert `.done()` + `.exception() is None`, not `.cancelled()`.

**`TrustScorer` and the policy rate limiter bound their in-memory state via a shared `BoundedStateStore`.**
Both `TrustScorer._sessions` (per-session trust state) and `ToolPolicyEngine`'s `_SlidingWindowCounter._windows` (per agent+tool rate-limit windows) are unbounded dicts that accumulate one entry per session/key for the life of the process — a slow memory leak for long-running deployments with many short-lived sessions. `agentmoat/engine/_state.py` provides a generic, thread-safe `BoundedStateStore[K, V]` with combined LRU (`max_entries`, default `10_000`) and TTL (`ttl_seconds`, default `3600`) eviction, exploiting the invariant that `OrderedDict` move-to-end order exactly matches access-time order — so TTL eviction only ever needs to inspect entries at the front until it finds one that hasn't expired (O(1) amortized, no full-dict scans). Both `TrustScorer` and `ToolPolicyEngine` accept `max_*`/`*_ttl_seconds` constructor overrides that pass through to the store.

**Event payloads are redacted before being persisted, on by default.**
`agentmoat/redaction.py`'s `redact()` walks `make_payload`'s kwargs (recursing through dicts/lists/tuples) and replaces recognizable secrets/PII — OpenAI/AWS/GitHub keys, JWTs, PEM private-key blocks, email addresses — with `«REDACTED:<kind>»` placeholders, *before* `_truncate` runs (so a placeholder is never clipped and a secret can't dodge the pattern by being split across the truncation boundary). Disable it via `AGENTMOAT_REDACT=0/false/no/off` (e.g. for local debugging where you want to see raw payloads) or override programmatically with `set_redaction_enabled(True/False/None)` — the latter takes precedence over the env var and is mainly for tests/embedders. Note `redact_text()` (the raw pattern-substitution primitive) always redacts regardless of the toggle; only `redact()` — and therefore `make_payload` — checks `is_redaction_enabled()`.

## What's Not Built Yet

- OpenTelemetry span export
- Multi-process EventBus (Redis-backed)
- Policy hot-reload from disk
- SARIF output format for CI pipelines
- Slack / PagerDuty alert sinks

## Module Map

```
agentmoat/events.py          -> SecurityEvent Pydantic model + make_payload() (redacts, then truncates)
agentmoat/redaction.py       -> Secret/PII redaction (redact/redact_text, AGENTMOAT_REDACT toggle)
agentmoat/bus.py             -> EventBus (sync + async emit, subscribers, background persistence worker, flush()/close())
agentmoat/store.py           -> SQLAlchemy async store (SQLite/Postgres)
agentmoat/audit.py           -> AuditLogger (hash-chained tamper-evident JSONL append, rotation, search, verify)
agentmoat/control.py         -> ApprovalGate + KillSwitch (interactive mode, human-in-the-loop, kill switch)
agentmoat/client.py          -> GuardedClient + GuardedMessages + GuardedStream + AgentMoatException/Killed
agentmoat/async_client.py    -> AsyncGuardedClient + AsyncGuardedMessages + AsyncGuardedStream
agentmoat/openai_client.py   -> GuardedOpenAI/AsyncGuardedOpenAI + GuardedChatCompletions (OpenAI Chat Completions wrappers)
agentmoat/callbacks.py       -> AgentMoatCallback (LangGraph BaseCallbackHandler)
agentmoat/cli.py             -> Click CLI (agentmoat mcp proxy stdio|sse, audit verify|tail|stats)
agentmoat/engine/
  injection.py               -> InjectionDetector (regex + optional embeddings)
  policy.py                  -> ToolPolicyEngine (YAML rules, rate limiting, argument constraints)
  constraints.py             -> ArgumentConstraintChecker (path traversal, SSRF, shell metachars, sensitive paths)
  trust.py                   -> TrustScorer (provenance tracking)
  _state.py                  -> BoundedStateStore (thread-safe TTL+LRU eviction, shared by trust.py/policy.py)
agentmoat/mcp/
  models.py                  -> MCPRequest/MCPResponse Pydantic models + error codes
  interceptor.py             -> MCPInterceptor (security checks on MCP tool calls)
  client.py                  -> StdioUpstreamClient + SSEUpstreamClient
  proxy.py                   -> MCPProxy (intercept → check → forward)
  server.py                  -> StdioProxyServer + SSEProxyServer
api/main.py                   -> FastAPI app with lifespan + CORS
api/routes/control.py         -> POST /control/kill/{id}, /kill-all, /revive/{id}, GET /control/status
api/routes/events.py          -> GET /events, /events/{id}, /events/alerts
api/routes/sessions.py        -> GET /sessions, /sessions/{id}
dashboard/src/App.tsx         -> Tab shell (feed / timeline / alerts)
dashboard/src/components/
  EventFeed.tsx              -> Polling event table with expand-on-click
  SessionTimeline.tsx        -> Vertical timeline per session
  AlertBadge.tsx             -> Count badge for nav tab
examples/langgraph_demo.py    -> Multi-agent demo with malicious document
examples/sdk_demo.py          -> Sync SDK demo with injection attempt
examples/async_sdk_demo.py    -> Async SDK demo with AsyncGuardedClient
examples/openai_sdk_demo.py   -> Sync OpenAI SDK demo with GuardedOpenAI
examples/async_openai_sdk_demo.py -> Async OpenAI SDK demo with AsyncGuardedOpenAI
examples/mcp_proxy_demo.py    -> MCP proxy demo (transparent interception)
```

## Audit Logging

Every event can be durably persisted to a JSONL file by passing `audit_log=`:

```python
# GuardedClient
client = GuardedClient(anthropic.Anthropic(), audit_log="logs/audit.jsonl")

# AgentMoatCallback
callback = AgentMoatCallback(session_id="run-001", audit_log="logs/audit.jsonl")
```

`AuditLogger` is a synchronous, thread-safe, append-only writer. Each line is
a self-contained JSON object. Supports rotation (`rotate_mb=50`), `.tail()`,
`.search()`, and `.stats()`. File is created with parent dirs on first event.
