"""Session query routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agentmoat.events import SecurityEvent
from agentmoat.store import EventStore

from ..main import get_store, require_api_key

router = APIRouter(prefix="/sessions", tags=["sessions"], dependencies=[Depends(require_api_key)])


@router.get("")
async def list_sessions(store: EventStore = Depends(get_store)) -> list[dict]:
    """List all unique session IDs with summary statistics."""
    return await store.list_sessions()


@router.get("/{session_id}", response_model=list[SecurityEvent])
async def get_session(
    session_id: str,
    store: EventStore = Depends(get_store),
) -> list[SecurityEvent]:
    """Return all events for a session, ordered by timestamp ascending."""
    events = await store.list_events(session_id=session_id, limit=1000)
    if not events:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return sorted(events, key=lambda e: e.timestamp)
