"""Durable audit logger — appends every SecurityEvent to a JSONL file.

Designed for append-only auditability: one JSON object per line, each line a
complete self-contained record. The file survives process restarts and can be
tailed, grepped, and ingested by any log aggregator.

Thread-safe via a file-level lock. Supports automatic rotation when the file
exceeds a configurable size limit.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from .events import SecurityEvent

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "agentguard_audit.jsonl"
_DEFAULT_ROTATE_MB = 50


class AuditLogger:
    """Synchronous, append-only JSONL audit log.

    Subscribes to an :class:`~agentguard.bus.EventBus` and writes one JSON
    line per :class:`~agentguard.events.SecurityEvent`.  Every agent action
    — LLM calls, tool calls, injections, policy checks, session lifecycle —
    is recorded with full context and timestamp.

    Parameters
    ----------
    path:
        Path to the audit log file. Created (and parent directories) if it
        does not exist. Defaults to ``agentguard_audit.jsonl`` in the current
        working directory.
    rotate_mb:
        Rotate the log file when it exceeds this size in megabytes. The old
        file is renamed to ``<path>.<timestamp>``. Set to 0 to disable.
    include_payload:
        When False, ``payload`` and ``metadata`` fields are omitted from log
        lines to reduce verbosity. Useful for high-throughput environments
        where payload detail is available via the API. Default True.
    """

    def __init__(
        self,
        path: str = _DEFAULT_PATH,
        rotate_mb: float = _DEFAULT_ROTATE_MB,
        include_payload: bool = True,
    ) -> None:
        self.path = Path(path)
        self._rotate_bytes = int(rotate_mb * 1024 * 1024) if rotate_mb > 0 else 0
        self._include_payload = include_payload
        self._lock = threading.Lock()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("[AuditLogger] Writing to %s (rotate_mb=%s)", self.path, rotate_mb)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, event: SecurityEvent) -> None:
        """Write a single event. Safe to call from any thread."""
        self.write(event)

    def write(self, event: SecurityEvent) -> None:
        """Append a SecurityEvent as a single JSON line."""
        record = self._to_record(event)
        line = json.dumps(record, default=str) + "\n"

        with self._lock:
            if self._rotate_bytes and self._should_rotate():
                self._rotate()
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def tail(self, n: int = 50) -> list[SecurityEvent]:
        """Return the last ``n`` events from the audit log."""
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        tail_lines = lines[-n:] if len(lines) > n else lines
        events = []
        for line in reversed(tail_lines):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(SecurityEvent(**json.loads(line)))
            except Exception:
                pass
        return events

    def search(
        self,
        session_id: Optional[str] = None,
        severity: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 500,
    ) -> list[SecurityEvent]:
        """Scan the audit log and return matching events (newest first)."""
        if not self.path.exists():
            return []

        results: list[SecurityEvent] = []
        lines = self.path.read_text(encoding="utf-8").splitlines()

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if session_id and data.get("session_id") != session_id:
                    continue
                if severity and data.get("severity") != severity:
                    continue
                if event_type and data.get("event_type") != event_type:
                    continue
                results.append(SecurityEvent(**data))
                if len(results) >= limit:
                    break
            except Exception:
                pass

        return results

    def stats(self) -> dict:
        """Return summary statistics for the current audit log file."""
        if not self.path.exists():
            return {"exists": False, "path": str(self.path)}
        size = self.path.stat().st_size
        lines = self.path.read_text(encoding="utf-8").splitlines()
        valid = [l for l in lines if l.strip()]
        return {
            "path": str(self.path.resolve()),
            "size_bytes": size,
            "size_mb": round(size / 1024 / 1024, 3),
            "total_entries": len(valid),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _to_record(self, event: SecurityEvent) -> dict:
        data = {
            "event_id": event.event_id,
            "session_id": event.session_id,
            "agent_id": event.agent_id,
            "timestamp": event.timestamp.isoformat(),
            "source": event.source,
            "event_type": event.event_type,
            "severity": event.severity,
            "flags": event.flags,
            "parent_event_id": event.parent_event_id,
        }
        if self._include_payload:
            data["payload"] = event.payload
            data["metadata"] = event.metadata
        return data

    def _should_rotate(self) -> bool:
        return self.path.exists() and self.path.stat().st_size >= self._rotate_bytes

    def _rotate(self) -> None:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        rotated = self.path.with_suffix(f".{ts}.jsonl")
        self.path.rename(rotated)
        logger.info("[AuditLogger] Rotated %s -> %s", self.path, rotated)
