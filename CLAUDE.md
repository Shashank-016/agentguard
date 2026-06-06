# AgentGuard — Claude Code Session Guide

AgentGuard is a security observability layer for AI agents that detects prompt injection, tool policy violations, and trust degradation in real-time. It instruments both LangGraph-orchestrated agents and direct Anthropic SDK usage via a drop-in client wrapper and callback handler.

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

# Run the SDK demo
ANTHROPIC_API_KEY=sk-... python examples/sdk_demo.py

# Run tests
pytest
```

## Key Architectural Decisions

**EventBus.emit() is synchronous.**
LangGraph callbacks (BaseCallbackHandler) and the Anthropic SDK intercepts run in synchronous contexts with no event loop guarantee. Making emit() synchronous means it's safe to call from anywhere. Async persistence is scheduled via `asyncio.create_task()` only when a loop is running — otherwise silently skipped. This is an intentional trade-off: reliability of emission over guaranteed durability.

**GuardedClient proxies `__getattr__`.**
`GuardedClient` only intercepts `messages` (the property). Everything else — `beta`, `files`, `with_raw_response`, API key access — is transparently forwarded to `self._client` via `__getattr__`. This makes it a true drop-in: the user can pass a `GuardedClient` anywhere an `anthropic.Anthropic` is accepted.

**Trust scores degrade multiplicatively, not additively.**
A session reading two external documents doesn't just accumulate risk linearly — it compounds. `score = score * EXTERNAL (0.3)` on each external content event. This models real-world provenance accurately: every hop through untrusted data multiplies the uncertainty. Resetting to 0 on injection (rather than multiplicative) reflects that injection is a binary, qualitative event.

**Observe-don't-block is the default.**
In `mode="observe"` (default), AgentGuard detects and logs but never interrupts the agent. This is intentional: a security tool that breaks production agents won't get deployed. Users opt into blocking via `mode="enforce"` once they've validated the detection quality in their environment.

**Rule-based detection runs before embedding-based.**
Regex patterns have zero latency and high precision on known attack patterns. Embeddings catch paraphrased variants the regex misses, but require loading a ~80MB model. Embeddings are opt-in via `use_embeddings=True`.

**Policy deny-list is checked before allow-list.**
Explicit denies always beat explicit allows. If a tool appears in both (misconfiguration), it's blocked. This is the safer default for a security tool.

## What's Not Built Yet

- Streaming support (`messages.stream()` interception)
- OpenTelemetry span export
- Multi-process EventBus (Redis-backed)
- MCP (Model Context Protocol) server integration
- Async GuardedClient wrapping `AsyncAnthropic`
- Policy hot-reload from disk
- SARIF output format for CI pipelines
- Slack / PagerDuty alert sinks

## Module Map

```
agentguard/events.py       -> SecurityEvent Pydantic model + make_payload()
agentguard/bus.py          -> EventBus (sync emit, async persist, subscribers)
agentguard/store.py        -> SQLAlchemy async store (SQLite/Postgres)
agentguard/audit.py        -> AuditLogger (sync JSONL append, rotation, search)
agentguard/client.py       -> GuardedClient + GuardedMessages
agentguard/callbacks.py    -> AgentGuardCallback (LangGraph BaseCallbackHandler)
agentguard/engine/
  injection.py             -> InjectionDetector (regex + optional embeddings)
  policy.py                -> ToolPolicyEngine (YAML rules, rate limiting)
  trust.py                 -> TrustScorer (provenance tracking)
api/main.py                -> FastAPI app with lifespan + CORS
api/routes/events.py       -> GET /events, /events/{id}, /events/alerts
api/routes/sessions.py     -> GET /sessions, /sessions/{id}
dashboard/src/App.tsx      -> Tab shell (feed / timeline / alerts)
dashboard/src/components/
  EventFeed.tsx            -> Polling event table with expand-on-click
  SessionTimeline.tsx      -> Vertical timeline per session
  AlertBadge.tsx           -> Count badge for nav tab
examples/langgraph_demo.py -> Multi-agent demo with malicious document
examples/sdk_demo.py       -> Direct SDK demo with injection attempt
```

## Audit Logging

Every event can be durably persisted to a JSONL file by passing `audit_log=`:

```python
# GuardedClient
client = GuardedClient(anthropic.Anthropic(), audit_log="logs/audit.jsonl")

# AgentGuardCallback
callback = AgentGuardCallback(session_id="run-001", audit_log="logs/audit.jsonl")
```

`AuditLogger` is a synchronous, thread-safe, append-only writer. Each line is
a self-contained JSON object. Supports rotation (`rotate_mb=50`), `.tail()`,
`.search()`, and `.stats()`. File is created with parent dirs on first event.
