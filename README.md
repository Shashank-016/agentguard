# AgentGuard

**Real-time security observability for AI agents — detect prompt injection, tool policy violations, and trust degradation before they become incidents.**

---

## The Problem

49% of organizations deploying AI agents have no visibility into what those agents actually do at runtime. An agent that reads a malicious document, gets injected with rogue instructions, and then calls `write_file` or `send_email` on behalf of an attacker is indistinguishable from normal operation — unless you're watching.

AgentGuard is the missing security layer. It instruments your agents at the SDK and framework level, detecting threats in real-time and surfacing them through an audit API and dashboard — without requiring you to change how your agents work.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Your Agent Code                         │
│                                                                 │
│   GuardedClient.messages.create(...)                            │
│   AgentGuardCallback attached to LangGraph graph                │
└────────────────┬────────────────────────────────────────────────┘
                 │ intercepts
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AgentGuard Core                           │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ InjectionDetect │  │ ToolPolicyEngine  │  │  TrustScorer  │  │
│  │  regex + embed  │  │  YAML rules       │  │  provenance   │  │
│  └────────┬────────┘  └────────┬─────────┘  └───────┬───────┘  │
│           └───────────────────┬┘                    │           │
│                               ▼                     │           │
│                         EventBus ◄──────────────────┘           │
│                               │                                 │
│                               ▼                                 │
│                         EventStore (SQLite / Postgres)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     FastAPI Audit API           React Dashboard
     GET /events                 Live feed
     GET /sessions               Session timeline
     GET /alerts                 Alerts panel
```

---

## Quick Start

```bash
pip install agentguard
```

### Drop-in SDK wrapper

```python
import anthropic
from agentguard import GuardedClient

# Exact same interface as anthropic.Anthropic()
client = GuardedClient(
    anthropic.Anthropic(),
    agent_id="researcher",
    policy_path="policy.yaml",   # optional
    mode="observe",              # "enforce" to block violations
)

# Use exactly as before — nothing else changes
response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    messages=[{"role": "user", "content": user_input}],
)
```

### Async usage

```python
import anthropic
from agentguard import AsyncGuardedClient

# Drop-in replacement for anthropic.AsyncAnthropic()
client = AsyncGuardedClient(
    anthropic.AsyncAnthropic(),
    agent_id="researcher",
    mode="observe",
)

# Use exactly as before — same interface, full async
response = await client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    messages=[{"role": "user", "content": user_input}],
)

# Streaming also supported
async with client.messages.stream(model=..., messages=...) as stream:
    async for text in stream.text_stream:
        print(text, end="", flush=True)
```

`AsyncGuardedClient` intercepts all `messages.create()` and `messages.stream()` calls,
scanning for injection, enforcing policy, and tracking trust — all without blocking the
event loop. Every attribute not explicitly intercepted (`beta`, `models`, etc.) is
transparently proxied to the underlying `AsyncAnthropic` instance.

### MCP Proxy

AgentGuard can run as a **transparent MCP proxy** — sitting between your agent and any
MCP server, validating every tool call without requiring changes to your agent code.

#### Stdio proxy (wrap any MCP CLI server)

In your MCP client config, replace the upstream command with AgentGuard:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "agentguard",
      "args": [
        "mcp", "proxy", "stdio",
        "--upstream-cmd", "npx -y @modelcontextprotocol/server-filesystem /tmp",
        "--agent-id", "my-agent",
        "--policy", "policy.yaml",
        "--mode", "enforce"
      ]
    }
  }
}
```

#### HTTP/SSE proxy (wrap any remote MCP server)

```bash
agentguard mcp proxy sse \
    --upstream-url http://mcp-server:3000 \
    --port 8899 \
    --agent-id my-agent \
    --mode enforce
```

Then point your agent at `http://localhost:8899` instead of the real server.

#### What the proxy checks

Every `tools/call` request is evaluated against:
- **Injection detector** — scans tool arguments for injected instructions
- **Tool policy** — validates the tool name against your `policy.yaml` allow/deny lists
- **Trust scorer** — flags sensitive tool calls in low-trust sessions

All other MCP methods (`tools/list`, `resources/read`, etc.) are logged and forwarded
without blocking.

### LangGraph callback

```python
from agentguard import AgentGuardCallback

callback = AgentGuardCallback(
    session_id="run-001",
    agent_id="researcher",
    policy_path="policy.yaml",
)

# Attach to any LangGraph / LangChain runnable
result = graph.invoke(state, config={"callbacks": [callback]})

# Print a security report at the end
print(callback.session_report())
```

---

## Policy File

```yaml
version: "1"
agents:
  researcher:
    allowed_tools: [web_search, read_file]
    denied_tools:  [write_file, execute_code]
    rate_limits:
      web_search: 10/minute

  writer:
    allowed_tools: [write_file, read_file]
    denied_tools:  [web_search, execute_code]
```

---

## What Gets Detected

| Threat | Detection Method | Default Severity |
|--------|-----------------|-----------------|
| Jailbreak attempt | Regex: "ignore previous instructions", "you are now DAN" | Critical |
| System prompt exfiltration | Regex: "print your system prompt", "repeat everything above" | Critical |
| Role override | Regex: "act as if you have no restrictions" | Critical |
| Tool abuse via injection | Regex: "call the write_file tool" | Critical |
| Indirect injection (docs, web) | Regex + embedding similarity | Warning/Critical |
| Tool policy violation | YAML policy engine | Critical |
| Rate limit exceeded | Sliding window counter | Critical |
| Low-trust agent calling sensitive tools | Trust score degradation | Warning |
| Multi-agent trust chain poisoning | Multiplicative provenance tracking | Warning |

---

## Running the API + Dashboard

```bash
# 1. Install
pip install -e ".[langgraph]"

# 2. Start the audit API
uvicorn api.main:app --reload

# 3. Start the dashboard
cd dashboard
npm install
npm run dev
# → http://localhost:5173

# 4. Run the demo
python examples/langgraph_demo.py
```

---

## Running Tests

```bash
pytest
```

---

## Trust Scoring

AgentGuard tracks *information provenance* across agent hops. When a session processes external content (a file, a web page, a user upload), its trust score degrades:

```
Initial:          1.0  (TRUSTED  — human instructions)
After file read:  0.3  (EXTERNAL — external content processed)
After handoff:    0.21 (EXTERNAL — downstream agent inherits low trust)
After injection:  0.0  (UNTRUSTED — flagged)
```

When trust drops below 0.5, any attempt to call a sensitive tool (write, execute, send, delete) emits a `trust_flag` warning even if the tool is otherwise policy-allowed.

---

## Roadmap

- [x] **Async GuardedClient** — `AsyncGuardedClient` wraps `AsyncAnthropic` for async codebases
- [x] **Streaming support** — `GuardedStream` / `AsyncGuardedStream` intercept `messages.stream()`
- [x] **MCP server integration** — transparent stdio + SSE proxy for Model Context Protocol
- [ ] **OpenTelemetry export** — emit spans/traces to any OTEL-compatible backend
- [ ] **Multi-process bus** — Redis-backed EventBus for distributed agent deployments
- [ ] **Slack/PagerDuty alerting** — push critical events to on-call channels
- [ ] **SARIF export** — machine-readable security findings for CI integration
- [ ] **Policy hot-reload** — watch policy.yaml for changes without restart

---

## License

MIT — see [LICENSE](LICENSE)
