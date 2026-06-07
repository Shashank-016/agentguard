"""AgentGuard FastAPI application — audit log REST API."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agentguard.store import EventStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application-level store singleton
# ---------------------------------------------------------------------------

_store: EventStore | None = None


def get_store() -> EventStore:
    """FastAPI dependency — returns the initialised EventStore."""
    if _store is None:
        raise RuntimeError("Store not initialised — lifespan not running")
    return _store


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------
#
# Set AGENTGUARD_API_KEY to require callers to present it via the
# `X-API-Key` header or `Authorization: Bearer <key>`. If the env var is
# unset, the API stays open — same as before this check existed — but logs a
# warning on every request so an operator notices the gap. This keeps the
# change backward-compatible for existing deployments while making the
# unauthenticated state loud rather than silent.

_WARNED_NO_API_KEY = False


def _configured_api_key() -> str | None:
    return os.getenv("AGENTGUARD_API_KEY") or None


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """FastAPI dependency enforcing the optional ``AGENTGUARD_API_KEY``.

    Accepts the key via ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``.
    Raises 401 on a missing/incorrect key. If no key is configured, requests
    are allowed through (with a one-time warning) for backward compatibility.
    """
    global _WARNED_NO_API_KEY
    expected = _configured_api_key()
    if expected is None:
        if not _WARNED_NO_API_KEY:
            logger.warning(
                "AGENTGUARD_API_KEY is not set — the audit API is running without "
                "authentication. Set AGENTGUARD_API_KEY to require callers to "
                "authenticate via the X-API-Key or Authorization header."
            )
            _WARNED_NO_API_KEY = True
        return

    presented = x_api_key
    if presented is None and authorization is not None:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            presented = token

    if presented is None or presented != expected:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key. Provide it via the X-API-Key "
            "header or 'Authorization: Bearer <key>'.",
        )


# ---------------------------------------------------------------------------
# Lifespan (replaces @app.on_event which is deprecated)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _store
    db_url = os.getenv("AGENTGUARD_DB_URL", "sqlite+aiosqlite:///agentguard.db")
    _store = EventStore(database_url=db_url)
    await _store.initialize()
    logger.info("AgentGuard API started — store: %s", db_url)
    yield
    await _store.close()
    logger.info("AgentGuard API shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgentGuard Audit API",
    description=(
        "Real-time security observability for AI agents. "
        "Query SecurityEvents emitted by GuardedClient and AgentGuardCallback instrumentation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Liveness check — returns event count from the persistent store."""
    count = await _store.count() if _store else 0
    return {"status": "ok", "event_count": count}


# Import routes after app is defined to avoid circular imports.
from api.routes.control import router as control_router  # noqa: E402
from api.routes.events import router as events_router  # noqa: E402
from api.routes.sessions import router as sessions_router  # noqa: E402

app.include_router(control_router)
app.include_router(events_router)
app.include_router(sessions_router)
