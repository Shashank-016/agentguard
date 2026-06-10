"""MCP (Model Context Protocol) JSON-RPC 2.0 protocol models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class MCPRequest(BaseModel):
    """A JSON-RPC 2.0 request message."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    method: str
    params: dict[str, Any] | None = None


class MCPResponse(BaseModel):
    """A JSON-RPC 2.0 response message."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class MCPToolCallParams(BaseModel):
    """Parameters for a ``tools/call`` request."""

    name: str
    arguments: dict[str, Any] = {}


class MCPToolDefinition(BaseModel):
    """A tool definition returned by ``tools/list``."""

    name: str
    description: str | None = None
    inputSchema: dict[str, Any] | None = None


class MCPError(BaseModel):
    """A JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any | None = None


# Standard JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# AgentMoat custom error codes
AGENTMOAT_POLICY_VIOLATION = -32001
AGENTMOAT_INJECTION_DETECTED = -32002
AGENTMOAT_TRUST_VIOLATION = -32003
AGENTMOAT_SESSION_KILLED = -32004
AGENTMOAT_ENGINE_ERROR = -32005
