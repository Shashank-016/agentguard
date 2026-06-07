"""Event query routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from agentguard.events import SecurityEvent
from agentguard.store import EventStore

from ..main import get_store, require_api_key

router = APIRouter(prefix="/events", tags=["events"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=list[SecurityEvent])
async def list_events(
    session_id: str | None = Query(None),
    severity: str | None = Query(None),
    event_type: str | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    store: EventStore = Depends(get_store),
) -> list[SecurityEvent]:
    """List security events with optional filters."""
    return await store.list_events(
        session_id=session_id,
        severity=severity,
        event_type=event_type,
        limit=limit,
        offset=offset,
    )


@router.get("/alerts", response_model=list[SecurityEvent])
async def list_alerts(
    limit: int = Query(100, ge=1, le=500),
    store: EventStore = Depends(get_store),
) -> list[SecurityEvent]:
    """Return all events with severity 'warning' or 'critical'."""
    warning = await store.list_events(severity="warning", limit=limit)
    critical = await store.list_events(severity="critical", limit=limit)
    combined = sorted(warning + critical, key=lambda e: e.timestamp, reverse=True)
    return combined[:limit]


@router.get("/{event_id}", response_model=SecurityEvent)
async def get_event(
    event_id: str,
    store: EventStore = Depends(get_store),
) -> SecurityEvent:
    """Fetch a single event by ID."""
    event = await store.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")
    return event
