"""Bounded, thread-safe in-memory state with TTL + LRU eviction.

Shared by :class:`~agentguard.engine.trust.TrustScorer` and the policy
engine's sliding-window rate limiter — both accumulate one entry per
session/agent (and tool) for the lifetime of the process. Left unbounded, a
long-running proxy serving many distinct sessions or agents leaks memory
indefinitely. Eviction order matches access recency (an :class:`OrderedDict`
moved-to-end on every touch), so the least-recently-used entries are always
at the front — which lets both TTL and size-based eviction run in O(1)
amortized time without scanning the whole store.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")

DEFAULT_MAX_ENTRIES = 10_000
DEFAULT_TTL_SECONDS = 3600.0


class BoundedStateStore(Generic[K, V]):
    """Thread-safe ``dict``-like store bounded by entry count and age.

    Parameters
    ----------
    max_entries:
        Hard cap on the number of entries. The least-recently-used entry is
        evicted whenever a new key would push the store over this limit.
    ttl_seconds:
        Entries not touched for longer than this are evicted lazily, on the
        next call to :meth:`get_or_create`.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._entries: OrderedDict[K, V] = OrderedDict()
        self._access_times: dict[K, float] = {}
        self._lock = threading.Lock()

    def get_or_create(self, key: K, factory: Callable[[], V]) -> V:
        """Return the value for ``key``, creating it via ``factory`` if absent.

        Touching a key (whether it already existed or was just created) marks
        it as most-recently-used and refreshes its TTL.
        """
        with self._lock:
            self._evict_expired_locked()
            if key in self._entries:
                self._entries.move_to_end(key)
            else:
                self._entries[key] = factory()
                while len(self._entries) > self._max_entries:
                    evicted_key, _ = self._entries.popitem(last=False)
                    self._access_times.pop(evicted_key, None)
            self._touch_locked(key)
            return self._entries[key]

    def _touch_locked(self, key: K) -> None:
        # OrderedDict position already encodes recency (front = oldest); we
        # additionally stamp the access time so TTL eviction can compare ages
        # without assuming the system clock and insertion order ever diverge.
        self._access_times[key] = time.monotonic()

    def _evict_expired_locked(self) -> None:
        if not self._entries:
            return
        cutoff = time.monotonic() - self._ttl_seconds
        while self._entries:
            oldest_key = next(iter(self._entries))
            if self._access_times.get(oldest_key, cutoff) < cutoff:
                self._entries.popitem(last=False)
                self._access_times.pop(oldest_key, None)
            else:
                break

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
