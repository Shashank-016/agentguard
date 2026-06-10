# Security Policy

## Reporting a vulnerability

AgentMoat is a security tool, so we take vulnerabilities in AgentMoat itself seriously.

**Please do not open a public issue for security vulnerabilities.** Instead, report privately
via GitHub's [security advisory form](https://github.com/Shashank-016/agentmoat/security/advisories/new)
("Report a vulnerability" under the Security tab). This keeps the details private until a fix is
available.

When reporting, please include:

- A description of the vulnerability and its impact
- Steps to reproduce (a minimal proof of concept if possible)
- The AgentMoat version or commit you tested against

## What to expect

- Acknowledgement of your report within a few days.
- An assessment and, where applicable, a fix or mitigation.
- Credit in the release notes if you would like it.

## Scope

In scope: bypasses of the detection or enforcement logic, ways to defeat the tamper-evident audit
log, authentication gaps in the audit API, and any path that causes AgentMoat to fail open when
it should fail closed.

Out of scope (by design, see `THREAT_MODEL.md`): obfuscation-based bypasses of the regex injection
detector that are already documented as limitations, `observe` mode being passive (it logs but
does not block), and attacks that require pre-existing code execution on the host running
AgentMoat.

## Supported versions

AgentMoat is pre-1.0 and under active development. Security fixes are applied to the latest
release and `master`. Pin a specific version and review the changelog before upgrading.
