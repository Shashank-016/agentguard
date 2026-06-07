"""Regression tests for EventBus durable persistence (Fix 1).

Before this fix, ``emit()`` only scheduled persistence when an event loop was
already running — every synchronous caller (LangGraph callbacks, sync
GuardedClient, the MCP interceptor) silently lost its events to the durable
store. The background persistence worker must make this reliable regardless
of whether the caller is sync or async.
"""

from __future__ import annotations

import threading

from agentguard.bus import EventBus
from agentguard.events import SecurityEvent


class _FakeAsyncStore:
    """Minimal async store that counts saved events."""

    def __init__(self) -> None:
        self.saved: list[SecurityEvent] = []
        self._lock = threading.Lock()

    async def save(self, event: SecurityEvent) -> None:
        with self._lock:
            self.saved.append(event)

    @property
    def n(self) -> int:
        with self._lock:
            return len(self.saved)


def _make_event(**overrides) -> SecurityEvent:
    defaults = dict(
        session_id="sess-1",
        agent_id="agent-1",
        source="sdk",
        event_type="tool_call",
    )
    defaults.update(overrides)
    return SecurityEvent(**defaults)


class TestSyncEmitPersists:
    def test_sync_emit_with_no_running_loop_persists_after_flush(self):
        """The exact scenario that used to lose data: emit() from a plain
        synchronous test function (no running event loop) must still persist
        once flush() is called."""
        store = _FakeAsyncStore()
        bus = EventBus(store=store)

        bus.emit(_make_event())
        bus.flush()

        assert store.n == 1
        bus.close()

    def test_bulk_sync_emit_persists_all_events(self):
        """100 rapid sync emits — no drops, no GC loss of scheduled futures."""
        store = _FakeAsyncStore()
        bus = EventBus(store=store)

        for i in range(100):
            bus.emit(_make_event(agent_id=f"agent-{i}"))
        bus.flush()

        assert store.n == 100
        bus.close()


class TestClose:
    def test_close_stops_worker_thread(self):
        store = _FakeAsyncStore()
        bus = EventBus(store=store)

        bus.emit(_make_event())
        bus.flush()

        worker_thread = bus._worker_thread
        assert worker_thread is not None
        assert worker_thread.is_alive()

        bus.close()

        assert not worker_thread.is_alive()

    def test_close_without_emitting_is_a_noop(self):
        store = _FakeAsyncStore()
        bus = EventBus(store=store)
        bus.close()  # must not raise even though the worker never started
