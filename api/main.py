"""AgentGuard FastAPI application — audit log REST API."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
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
# Lifespan (replaces @app.on_event which is deprecated)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _store
    db_url = os.getenv(
        "AGENTGUARD_DB_URL", "sqlite+aiosqlite:///agentguard.db"
    )
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
from api.routes.events import router as events_router  # noqa: E402
from api.routes.sessions import router as sessions_router  # noqa: E402

app.include_router(events_router)
app.include_router(sessions_router)
