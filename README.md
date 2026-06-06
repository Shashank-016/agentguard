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

### OpenAI SDK wrapper

```bash
pip install "agentguard[openai]"
```

```python
import openai
from agentguard import GuardedOpenAI

# Exact same interface as openai.OpenAI()
client = GuardedOpenAI(
    openai.OpenAI(),
    agent_id="researcher",
    policy_path="policy.yaml",   # optional
    mode="observe",              # "enforce" / "interactive" also supported
)

# Use exactly as before — nothing else changes
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": user_input}],
    tools=[{"type": "function", "function": {"name": "read_file", "parameters": {...}}}],
)
```

`AsyncGuardedOpenAI` mirrors `AsyncGuardedClient` for `openai.AsyncOpenAI()`:

```python
import openai
from agentguard import AsyncGuardedOpenAI

client = AsyncGuardedOpenAI(openai.AsyncOpenAI(), agent_id="researcher")

response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": user_input}],
)
```

Both wrappers run the exact same injection → policy → trust → argument-constraint
pipeline as the Anthropic clients (same event types, same `mode` semantics, same
`KillSwitch`/`ApprovalGate` integration), adapted to the Chat Completions
request/response shapes — including parsing `tool_calls[i].function.arguments`
(a JSON string) before constraint checking. Streaming is not yet intercepted;
`chat.completions` is proxied via `__getattr__` for anything beyond `.create()`.

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

### Argument constraints

Tool *names* are only half the story — `write_file("/etc/crontab", payload)` passes a
name-level check for any agent allowed to use `write_file`. `ToolPolicyEngine.check_arguments()`
inspects the *arguments* of a tool call, combining always-on built-in detectors with
per-tool rules declared in the policy file:

```yaml
agents:
  writer:
    tool_constraints:
      write_file:
        path_allowlist: ["/tmp/**", "./output/**"]   # only these globs are permitted
        path_denylist:  ["/etc/**", "~/.ssh/**"]      # these are always blocked
        max_arg_length: 10000                         # flag oversized argument values
      fetch:
        url_denylist: ["169.254.169.254", "localhost", "10.*"]
        # url_allowlist, arg_denylist also supported
```

Built-in detectors run on every tool call regardless of configuration:

| Detector | Flag | Triggers on |
|----------|------|-------------|
| Path traversal | `constraint:path_traversal` | `../`, `..\`, or URL-encoded `%2e%2e` in any argument |
| SSRF targets | `constraint:ssrf_target` | URLs/hosts pointing at `169.254.169.254`, `localhost`, `127.0.0.1`, RFC-1918 ranges, `metadata.google.internal` |
| Shell metacharacters | `constraint:shell_metachar` | `;`, `\|`, `&&`, `` ` ``, `$(`, `>`, `<` in arguments to tools whose name suggests command execution (`exec`, `shell`, `command`, `run`, `bash`, `sh`) |
| Sensitive path access | `constraint:sensitive_path` | `/etc/`, `/root/`, `~/.ssh`, `id_rsa`, `.env`, `credentials`, `/proc/` |

Violations are emitted as `policy_violation` events with `severity="critical"` and raise
`AgentGuardException` in `enforce` mode — both from the SDK wrappers (checked against the
arguments the model produced, before the agent runtime executes the tool) and from the MCP
proxy (checked before the call is forwarded upstream, where blocking actually prevents execution).

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

## Response modes

Every guarded client, callback, and the MCP proxy take a `mode`:

| Mode | Behavior |
|------|----------|
| `"observe"` (default) | Detect and log everything — never interrupts the agent. |
| `"enforce"` | Raise `AgentGuardException` (or return a JSON-RPC error from the MCP proxy) on any hard violation. A fixed, pre-decided policy. |
| `"interactive"` | Route violations to a human (or programmatic approver) for a real-time decision via `ApprovalGate`. A "deny" blocks the call just like enforce mode; an "approve" lets it through. |

`"interactive"` mode is for situations where a blanket policy is too coarse — let
a human apply judgment to a specific borderline case instead of pre-encoding
every exception:

```python
from agentguard import GuardedClient, ApprovalGate
from agentguard.control import ApprovalRequest, ApprovalDecision

def slack_approval_handler(request: ApprovalRequest) -> ApprovalDecision:
    # Post to Slack, wait for a thumbs-up/thumbs-down reaction, etc.
    ...
    return "approve"  # or "deny"

client = GuardedClient(
    anthropic.Anthropic(),
    agent_id="researcher",
    mode="interactive",
    approval_gate=ApprovalGate(handler=slack_approval_handler),
)
```

Each request emits `approval_required`, then `approval_granted` or
`approval_denied`, so the full decision trail lands in the audit log. The
default handler (when no `approval_gate=` is supplied) prompts on the CLI with
a y/N confirmation — fine for local development, but register your own handler
(Slack, a web UI, a queue) for anything running unattended. A misbehaving or
exception-raising handler defaults to `"deny"` — approval gates fail closed.

Note: `trust_flag` warnings never hard-block in `enforce` mode (a low trust
score alone shouldn't halt an agent), but in `interactive` mode they still
route through the approval gate — a human's explicit "deny" blocks the call.
This gives interactive mode finer-grained control than a blanket policy.

### Kill switch

Independent of `mode`, any session — or every session in the process — can be
halted immediately via `KillSwitch`:

```python
from agentguard.control import get_default_kill_switch

switch = get_default_kill_switch()
switch.kill_session("session-123")   # halt one session
switch.kill_all()                    # halt every session in this process
switch.revive_session("session-123") # restore it
switch.status()                      # {"global": False, "killed_sessions": [...]}
```

A killed session's next intercepted action raises `AgentGuardKilled` (a subclass
of `AgentGuardException`) — or, for the MCP proxy, returns a JSON-RPC error
(`AGENTGUARD_SESSION_KILLED`) — *before* any API call or tool execution happens.
A critical `session_end` event with `flags=["kill:tripped"]` is emitted first,
so the halt is visible in the audit trail.

The same switch is reachable over HTTP once the audit API is running:

```bash
curl -X POST http://localhost:8000/control/kill/session-123
curl -X POST http://localhost:8000/control/kill-all
curl -X POST http://localhost:8000/control/revive/session-123
curl http://localhost:8000/control/status
```

These endpoints affect sessions in the API process only — a multi-process
deployment needs a shared backing store (see Roadmap) for one trip to halt
every worker. They're also unauthenticated for now; put them behind your own
auth/network controls before exposing them.

---

## Tamper-evident audit log

`AuditLogger` (passed via `audit_log=` to any guarded client/callback) writes one JSON
object per line to a durable JSONL file. By default (`chained=True`) every record also
carries `prev_hash` — the SHA-256 `record_hash` of the previous line, with a genesis value
of 64 zeros for the first line in a fresh file — and its own `record_hash`, a digest over
the record's canonical JSON plus `prev_hash`. Editing or deleting any line breaks the link
to the next record, so tampering is always detectable, not just guessable. The chain
survives process restarts (it resumes from the last line on disk) and rotations (the new
file's first record continues from the rotated file's last hash).

```bash
agentguard audit verify agentguard_audit.jsonl
# ✓ Chain intact — 1,432 records verified
#   (or, if a line was edited or removed:)
# ✗ Chain broken at line 87 — record was modified or a prior line was deleted

agentguard audit tail agentguard_audit.jsonl -n 50
agentguard audit stats agentguard_audit.jsonl   # counts by event_type and severity
```

This gives you a forensic trail suitable for SOC 2 / ISO 27001 evidence: an auditor (or an
incident responder) can independently confirm that the log they're looking at is the
complete, unaltered record AgentGuard produced — not a reconstruction. It does not, by
itself, prove *who* tampered with a file; pair it with filesystem-level access controls and
off-host replication for full chain-of-custody guarantees.

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

- [x] **OpenAI SDK support** — `GuardedOpenAI` / `AsyncGuardedOpenAI` wrap `openai.OpenAI` / `AsyncOpenAI`
- [x] **Async GuardedClient** — `AsyncGuardedClient` wraps `AsyncAnthropic` for async codebases
- [x] **Streaming support** — `GuardedStream` / `AsyncGuardedStream` intercept `messages.stream()`
- [x] **MCP server integration** — transparent stdio + SSE proxy for Model Context Protocol
- [x] **Tamper-evident audit log** — SHA-256 hash-chained JSONL with `agentguard audit verify`
- [x] **Human-in-the-loop approval** — `mode="interactive"` routes violations through `ApprovalGate`
- [x] **Kill switch** — halt any session (or every session) immediately, programmatically or via `/control`
- [ ] **OpenTelemetry export** — emit spans/traces to any OTEL-compatible backend
- [ ] **Multi-process bus** — Redis-backed EventBus for distributed agent deployments
- [ ] **Slack/PagerDuty alerting** — push critical events to on-call channels
- [ ] **SARIF export** — machine-readable security findings for CI integration
- [ ] **Policy hot-reload** — watch policy.yaml for changes without restart

---

## License

MIT — see [LICENSE](LICENSE)
