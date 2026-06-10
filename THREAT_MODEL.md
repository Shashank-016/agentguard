# AgentMoat Threat Model

This document states plainly what AgentMoat defends against, what it does not, and the
assumptions behind those boundaries. It is deliberately honest about limitations — a security
tool that overstates its protection is worse than none.

## What AgentMoat is

AgentMoat is a security observability and policy-enforcement layer that sits between an AI agent
and the model/tools it uses. It instruments three points: model calls (Anthropic & OpenAI SDKs),
orchestration steps (LangGraph callbacks), and tool calls (a transparent MCP proxy). At each
point it can detect, log, and — in `enforce`/`interactive` mode — block.

## Assumed attacker

The primary adversary is an attacker who controls *content the agent ingests* but not the host
running the agent: poisoned documents, web pages, API responses, tool outputs, or user input
crafted to hijack the agent's behavior (prompt injection, instruction override, data
exfiltration, abuse of tools the agent legitimately has). This is the dominant real-world threat
to autonomous agents.

We do **not** assume an attacker with code execution on the host. If the attacker can run code in
the same process, they can disable AgentMoat — see Limitations.

## What AgentMoat defends against

- **Prompt injection in inputs and tool results.** The injection detector scans user messages,
  system prompts, and (critically) tool/function-result messages — the main indirect-injection
  vector — for known override, jailbreak, exfiltration, and tool-abuse patterns.
- **Dangerous tool arguments.** Argument-level constraints catch path traversal (`../`), SSRF
  targets (cloud-metadata IPs, localhost, RFC-1918 ranges), shell metacharacters in
  command-style tools, and access to sensitive paths (`/etc`, `~/.ssh`, `.env`, credentials),
  plus configurable per-tool allow/deny lists for paths and URLs. This is enforced *before*
  execution at the MCP proxy layer.
- **Tool-policy violations.** Per-agent allow/deny lists and rate limits; a tool an agent is not
  permitted to call is blocked regardless of what the model decides.
- **Cross-agent trust abuse.** A trust score degrades as a session ingests external content and
  passes data between agents; sensitive actions taken under a low-trust context are flagged.
- **Audit tampering.** The audit log is hash-chained (SHA-256); editing or deleting any record
  breaks the chain and is detectable via `agentmoat audit verify`.
- **Runaway / unauthorized actions at machine speed**, via the kill switch (halt a session or all
  agents immediately) and human-in-the-loop approval for high-risk actions.

## What AgentMoat does NOT defend against (limitations)

- **Obfuscated injection.** The injection detector is pattern-based. It does not reliably catch
  payloads hidden via base64, unicode homoglyphs, zero-width characters, heavy paraphrasing, or
  translation into other languages. Optional embedding-based detection helps but is not
  exhaustive. Treat injection detection as defense-in-depth, not a guarantee.
- **`observe` mode is passive.** In the default `observe` mode AgentMoat logs and flags but does
  not block. Blocking requires `enforce` or `interactive` mode and correct policy configuration.
- **Misconfiguration.** Protection depends on the policy file. With no policy, all tools are
  allowed. An incomplete allow/deny list or missing argument constraints leaves gaps.
- **In-process compromise.** AgentMoat runs in the same process as the agent (except the MCP
  proxy, which is a separate process). An attacker who already has code execution on the host can
  disable or bypass it. AgentMoat protects against malicious *content*, not a compromised host.
- **Model-internal reasoning attacks.** AgentMoat inspects inputs, outputs, and tool calls — not
  the model's hidden chain of thought. Attacks that never surface in observable I/O are out of
  scope.
- **Multi-process / distributed limits.** Rate limits and in-memory trust/state are per-process.
  In a multi-replica deployment they are not globally coordinated (a shared backend is on the
  roadmap).
- **Streaming output detection is post-hoc.** Injection in streamed model *output* is detected
  after the stream closes; by then content has already reached the consumer. It is an audit
  signal, not prevention, for the streaming-output case.

## Defense-in-depth posture

AgentMoat is one layer. It is most effective combined with: least-privilege tool design,
sandboxed tool execution, scoped/short-lived credentials, and human review of high-impact
actions. No single layer — including this one — should be relied on alone.
