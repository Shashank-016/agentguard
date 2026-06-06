"""In-memory event bus with optional persistent store backend."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Callable, Optional

from .events import SecurityEvent

logger = logging.getLogger(__name__)


class EventBus:
    """Central dispatcher for SecurityEvents.

    Designed to be called from synchronous callback handlers (LangGraph,
    Anthropic SDK hooks) without requiring an active event loop. Persistence
    is handled asynchronously via ``asyncio.create_task`` when a store is
    attached and a loop is running.

    Parameters
    ----------
    store:
        Optional async store. When provided, every emitted event is also
        persisted. Safe to omit for in-process demos and tests.
    max_buffer:
        Maximum number of events held in the in-memory ring buffer.
    """

    def __init__(self, store=None, max_buffer: int = 1000) -> None:
        self._buffer: deque[SecurityEvent] = deque(maxlen=max_buffer)
        self._store = store
        self._subscribers: list[Callable[[SecurityEvent], None]] = []

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(self, event: SecurityEvent) -> None:
        """Emit an event synchronously.

        Appends to the in-memory buffer, notifies subscribers, and schedules
        an async persist if a store and running event loop are available.
        Failures in persistence are logged but never propagated — the bus must
        never crash agent code.
        """
        self._buffer.append(event)
        logger.debug(
            "[AgentGuard] %s: %s → %s  flags=%s",
            event.event_type,
            event.agent_id,
            event.severity.upper(),
            event.flags,
        )
        for fn in self._subscribers:
            try:
                fn(event)
            except Exception:
                logger.exception("EventBus subscriber raised an exception")

        if self._store:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._persist(event))
            except RuntimeError:
                # No running loop — caller is sync-only; skip async persist.
                pass

    async def emit_async(self, event: SecurityEvent) -> None:
        """Async variant — awaits persistence before returning."""
        self._buffer.append(event)
        for fn in self._subscribers:
            try:
                fn(event)
            except Exception:
                logger.exception("EventBus subscriber raised an exception")
        if self._store:
            await self._persist(event)

    async def _persist(self, event: SecurityEvent) -> None:
        try:
            await self._store.save(event)
        except Exception:
            logger.exception("Failed to persist event %s", event.event_id)

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, fn: Callable[[SecurityEvent], None]) -> None:
        """Register a callback invoked synchronously on every emit."""
        self._subscribers.append(fn)

    def unsubscribe(self, fn: Callable[[SecurityEvent], None]) -> None:
        """Remove a previously registered subscriber."""
        self._subscribers = [s for s in self._subscribers if s is not fn]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_session_events(self, session_id: str) -> list[SecurityEvent]:
        """Return all buffered events for a given session, oldest first."""
        return [e for e in self._buffer if e.session_id == session_id]

    def get_flagged(self) -> list[SecurityEvent]:
        """Return all buffered events with severity warning or critical."""
        return [e for e in self._buffer if e.severity in ("warning", "critical")]

    def all_events(self) -> list[SecurityEvent]:
        """Return all buffered events, oldest first."""
        return list(self._buffer)

    def clear(self) -> None:
        """Flush the in-memory buffer (does not affect the persistent store)."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
