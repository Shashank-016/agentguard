"""AgentGuard MCP (Model Context Protocol) proxy module.

Provides a transparent MCP proxy that intercepts all tool calls between agents
and MCP servers, applying injection detection, policy enforcement, and trust
scoring at the transport layer.
"""

from .client import SSEUpstreamClient, StdioUpstreamClient
from .interceptor import InterceptResult, MCPInterceptor
from .models import (
    AGENTGUARD_INJECTION_DETECTED,
    AGENTGUARD_POLICY_VIOLATION,
    AGENTGUARD_TRUST_VIOLATION,
    MCPRequest,
    MCPResponse,
    MCPToolCallParams,
    MCPToolDefinition,
)
from .proxy import MCPProxy
from .server import SSEProxyServer, StdioProxyServer

__all__ = [
    "MCPProxy",
    "MCPInterceptor",
    "InterceptResult",
    "StdioUpstreamClient",
    "SSEUpstreamClient",
    "StdioProxyServer",
    "SSEProxyServer",
    "MCPRequest",
    "MCPResponse",
    "MCPToolCallParams",
    "MCPToolDefinition",
    "AGENTGUARD_INJECTION_DETECTED",
    "AGENTGUARD_POLICY_VIOLATION",
    "AGENTGUARD_TRUST_VIOLATION",
]
