"""AgentGuard — security observability for AI agents.

Instruments LangGraph and Anthropic SDK agents to detect prompt injection,
tool policy violations, and trust degradation in real-time.

Quick start::

    import anthropic
    from agentguard import GuardedClient

    client = GuardedClient(
        anthropic.Anthropic(),
        agent_id="researcher",
        policy_path="policy.yaml",
    )
    response = client.messages.create(model="claude-opus-4-7", ...)

Async usage::

    import anthropic
    from agentguard import AsyncGuardedClient

    client = AsyncGuardedClient(
        anthropic.AsyncAnthropic(),
        agent_id="researcher",
        mode="observe",
    )
    response = await client.messages.create(model="claude-haiku-4-5-20251001", ...)
"""

from .async_client import AsyncGuardedClient
from .audit import AuditLogger
from .bus import EventBus
from .callbacks import AgentGuardCallback
from .client import AgentGuardException, GuardedClient
from .events import SecurityEvent
from .store import EventStore

__version__ = "0.1.0"
__all__ = [
    "GuardedClient",
    "AsyncGuardedClient",
    "AgentGuardCallback",
    "AgentGuardException",
    "AuditLogger",
    "EventBus",
    "EventStore",
    "SecurityEvent",
]
