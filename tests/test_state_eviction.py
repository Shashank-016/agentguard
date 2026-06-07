"""Regression tests for bounded in-memory engine state (Fix 5).

Before this fix, ``TrustScorer._sessions`` and the policy engine's sliding
-window rate-limit counters were plain unbounded dicts: a long-running proxy
serving many distinct sessions/agents would accumulate one entry per key
forever — a slow memory leak. ``BoundedStateStore`` now caps both by count
(LRU eviction) and by age (TTL eviction), and is safe to touch from multiple
threads via an internal ``threading.Lock``.
"""

from __future__ import annotations

import threading
import time

import pytest

from agentguard.engine._state import BoundedStateStore
from agentguard.engine.policy import ToolPolicyEngine
from agentguard.engine.trust import TrustScorer

# ---------------------------------------------------------------------------
# BoundedStateStore — the shared primitive
# ---------------------------------------------------------------------------


class TestBoundedStateStore:
    def test_lru_eviction_caps_size(self):
        store: BoundedStateStore[int, str] = BoundedStateStore(max_entries=3, ttl_seconds=3600)
        for i in range(5):
            store.get_or_create(i, lambda i=i: f"value-{i}")
        assert len(store) == 3

    def test_lru_eviction_drops_least_recently_used(self):
        store: BoundedStateStore[str, str] = BoundedStateStore(max_entries=2, ttl_seconds=3600)
        store.get_or_create("a", lambda: "A")
        store.get_or_create("b", lambda: "B")
        # Touch "a" so "b" becomes the least-recently-used entry.
        store.get_or_create("a", lambda: "A")
        store.get_or_create("c", lambda: "C")

        created = []

        def factory_b():
            created.append("b-recreated")
            return "B2"

        store.get_or_create("b", factory_b)
        assert created == ["b-recreated"]  # "b" had been evicted, so it was recreated
        assert len(store) == 2

    def test_ttl_eviction_drops_stale_entries(self):
        store: BoundedStateStore[str, str] = BoundedStateStore(max_entries=100, ttl_seconds=0.05)
        store.get_or_create("stale", lambda: "old")
        time.sleep(0.1)

        created = []

        def factory():
            created.append("recreated")
            return "fresh"

        value = store.get_or_create("stale", factory)
        assert value == "fresh"
        assert created == ["recreated"]

    def test_repeated_access_keeps_entry_alive(self):
        """Touching an entry refreshes its TTL — it should survive longer than
        a single TTL window if accessed regularly."""
        store: BoundedStateStore[str, int] = BoundedStateStore(max_entries=100, ttl_seconds=0.15)
        for _ in range(4):
            store.get_or_create("hot", lambda: 1)
            time.sleep(0.06)
        assert len(store) == 1

    def test_thread_safe_concurrent_creation(self):
        store: BoundedStateStore[int, int] = BoundedStateStore(max_entries=10_000, ttl_seconds=3600)
        errors: list[BaseException] = []

        def worker(start: int):
            try:
                for i in range(start, start + 200):
                    store.get_or_create(i, lambda i=i: i)
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i * 200,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store) == 1600


# ---------------------------------------------------------------------------
# TrustScorer — bounded session state
# ---------------------------------------------------------------------------


class TestTrustScorerBoundedState:
    def test_max_sessions_evicts_least_recently_used(self):
        scorer = TrustScorer(max_sessions=3, ttl_seconds=3600)
        for sid in ("s1", "s2", "s3", "s4"):
            scorer.record_external_content(sid, "web")

        assert len(scorer._sessions) == 3
        # s1 was evicted — accessing it again starts a brand-new TRUSTED session.
        assert scorer.score("s1") == pytest.approx(1.0)

    def test_ttl_evicts_stale_sessions(self):
        scorer = TrustScorer(max_sessions=100, ttl_seconds=0.05)
        scorer.record_injection_flag("stale-session")
        assert scorer.is_flagged("stale-session") is True

        time.sleep(0.1)

        # The session was evicted; querying it creates a fresh, untouched one.
        assert scorer.is_flagged("stale-session") is False
        assert scorer.score("stale-session") == pytest.approx(1.0)

    def test_default_bounds_are_generous(self):
        scorer = TrustScorer()
        scorer.record_external_content("s1", "web")
        assert len(scorer._sessions) == 1


# ---------------------------------------------------------------------------
# ToolPolicyEngine — bounded rate-limit windows
# ---------------------------------------------------------------------------


class TestRateLimiterBoundedState:
    def test_rate_limit_windows_are_bounded_by_lru(self):
        engine = ToolPolicyEngine(max_rate_limit_entries=2, rate_limit_ttl_seconds=3600)
        # Force the internal sliding-window counter to track three distinct
        # (agent, tool) pairs directly — this is the structure that used to
        # grow without bound.
        counter = engine._rate_counter
        for agent in ("agent-a", "agent-b", "agent-c"):
            counter.is_allowed(agent, "search", limit=10, window_seconds=60)

        assert len(counter._windows) == 2

    def test_rate_limit_still_enforced_after_eviction_pressure(self):
        engine = ToolPolicyEngine(max_rate_limit_entries=2, rate_limit_ttl_seconds=3600)
        counter = engine._rate_counter

        for agent in ("agent-a", "agent-b", "agent-c"):
            counter.is_allowed(agent, "search", limit=10, window_seconds=60)

        # agent-c is the most recent entry — its limit must still be enforced.
        # (The setup loop above already counts as agent-c's first call.)
        for _ in range(9):
            assert counter.is_allowed("agent-c", "search", limit=10, window_seconds=60) is True
        assert counter.is_allowed("agent-c", "search", limit=10, window_seconds=60) is False
