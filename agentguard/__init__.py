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
    # Use exactly like anthropic.Anthropic():
    response = client.messages.create(model="claude-opus-4-7", ...)
"""

from .audit import AuditLogger
from .bus import EventBus
from .callbacks import AgentGuardCallback
from .client import AgentGuardException, GuardedClient
from .events import SecurityEvent
from .store import EventStore

__version__ = "0.1.0"
__all__ = [
    "GuardedClient",
    "AgentGuardCallback",
    "AgentGuardException",
    "AuditLogger",
    "EventBus",
    "EventStore",
    "SecurityEvent",
]
