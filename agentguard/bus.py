"""In-memory event bus with optional persistent store backend."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections import deque
from collections.abc import Callable

from .events import SecurityEvent

logger = logging.getLogger(__name__)

_WORKER_THREAD_NAME = "agentguard-persist"


class EventBus:
    """Central dispatcher for SecurityEvents.

    Designed to be called from synchronous callback handlers (LangGraph,
    Anthropic SDK hooks) without requiring an active event loop, while still
    durably persisting every event when a store is attached.

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

        # Background persistence worker — a daemon thread that owns its own
        # asyncio event loop. Started lazily on first emit() when a store is
        # attached, so sync callers (no running loop) still get durable
        # persistence via asyncio.run_coroutine_threadsafe().
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._worker_thread: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._pending_futures: set[concurrent.futures.Future] = set()
        self._pending_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Background persistence worker
    # ------------------------------------------------------------------

    def _ensure_worker(self) -> asyncio.AbstractEventLoop:
        """Lazily start the daemon persistence thread and its event loop.

        Safe to call from any thread/context — the loop is created inside the
        worker thread itself (an asyncio loop must be run from the thread that
        owns it), and the caller blocks only until it's confirmed running.
        """
        with self._worker_lock:
            if (
                self._worker_loop is not None
                and self._worker_thread is not None
                and self._worker_thread.is_alive()
            ):
                return self._worker_loop

            ready = threading.Event()
            holder: dict[str, asyncio.AbstractEventLoop] = {}

            def _run() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                holder["loop"] = loop
                ready.set()
                try:
                    loop.run_forever()
                finally:
                    loop.close()

            thread = threading.Thread(target=_run, name=_WORKER_THREAD_NAME, daemon=True)
            thread.start()
            ready.wait()

            self._worker_loop = holder["loop"]
            self._worker_thread = thread
            return self._worker_loop

    def _schedule_persist(self, event: SecurityEvent) -> None:
        """Schedule durable persistence on the background worker loop.

        Non-blocking for the caller — works whether or not the caller has a
        running event loop of its own. The returned ``concurrent.futures.Future``
        is tracked in ``_pending_futures`` so it can't be garbage-collected
        mid-flight (which would silently cancel the persist), and a done
        callback removes it and logs any exception so a failing save is
        visible rather than swallowed.
        """
        loop = self._ensure_worker()
        future = asyncio.run_coroutine_threadsafe(self._persist(event), loop)

        with self._pending_lock:
            self._pending_futures.add(future)

        def _on_done(fut: concurrent.futures.Future) -> None:
            with self._pending_lock:
                self._pending_futures.discard(fut)
            if fut.cancelled():
                return
            exc = fut.exception()
            if exc is not None:
                logger.error("[AgentGuard] Failed to persist event %s: %s", event.event_id, exc)

        future.add_done_callback(_on_done)

    def flush(self, timeout: float = 5.0) -> None:
        """Block until all currently-pending persistence operations complete.

        Drains the futures tracked at the time of the call (events emitted
        afterward are not waited on). Call this after a burst of ``emit()``
        calls — e.g. in tests, or before process shutdown via :meth:`close` —
        to guarantee every event has reached the store.
        """
        deadline = time.monotonic() + timeout
        with self._pending_lock:
            futures = list(self._pending_futures)
        for future in futures:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                future.result(timeout=remaining)
            except Exception:
                logger.exception("[AgentGuard] Error while flushing persistence future")

    def close(self, timeout: float = 5.0) -> None:
        """Drain pending persistence work and stop the worker loop and thread.

        Safe to call even if the worker was never started (no-op). Call this
        from application shutdown (e.g. FastAPI's lifespan) to ensure no
        events are lost and the daemon thread exits cleanly.
        """
        with self._worker_lock:
            loop = self._worker_loop
            thread = self._worker_thread
            self._worker_loop = None
            self._worker_thread = None

        if loop is None or thread is None:
            return

        self.flush(timeout=timeout)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(self, event: SecurityEvent) -> None:
        """Emit an event synchronously.

        Appends to the in-memory buffer, notifies subscribers, and — if a
        store is attached — schedules durable persistence on the background
        worker thread via :meth:`_schedule_persist`. This is reliable in both
        sync and async contexts: persistence never depends on a currently
        running event loop, so it can never be silently skipped.
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
            self._schedule_persist(event)

    async def emit_async(self, event: SecurityEvent) -> None:
        """Async-safe emit. Use this from async contexts (AsyncGuardedClient, MCP proxy, etc.)

        Persists directly via ``await self._persist(event)`` rather than going
        through the background worker — the caller already has a running loop,
        so there's no benefit to hopping threads, and awaiting here means the
        event is durably stored before this coroutine returns.
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
                if asyncio.iscoroutinefunction(fn):
                    await fn(event)
                else:
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
