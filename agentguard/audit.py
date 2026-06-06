"""Durable audit logger — appends every SecurityEvent to a JSONL file.

Designed for append-only auditability: one JSON object per line, each line a
complete self-contained record. The file survives process restarts and can be
tailed, grepped, and ingested by any log aggregator.

Thread-safe via a file-level lock. Supports automatic rotation when the file
exceeds a configurable size limit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .events import SecurityEvent

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "agentguard_audit.jsonl"
_DEFAULT_ROTATE_MB = 50

# Hash chain genesis value — the ``prev_hash`` of the first record in a fresh file.
_GENESIS_HASH = "0" * 64


@dataclass
class ChainVerificationResult:
    """Result of walking and verifying an audit log's hash chain.

    Attributes
    ----------
    valid:
        True if every record's ``record_hash`` matches its recomputed hash and
        every record's ``prev_hash`` matches the previous record's ``record_hash``.
    records_checked:
        Number of records successfully verified before any break (or the total
        if the chain is valid).
    first_broken_line:
        1-indexed line number of the first detected break, or ``None`` if valid.
    detail:
        Human-readable description of the result or the break.
    """

    valid: bool
    records_checked: int
    first_broken_line: Optional[int]
    detail: str


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
    chained:
        When True (default), each record is hash-chained to the previous one
        via ``prev_hash``/``record_hash`` fields (SHA-256), making the log
        tamper-evident — see :meth:`verify`. When False, lines are written in
        the original independent-record format with no hash fields.

    Tamper-evident hash chain
    -------------------------
    Each chained record's ``record_hash`` is ``sha256(canonical_json(record) +
    prev_hash)``, where ``canonical_json`` is ``json.dumps(record, sort_keys=True,
    separators=(",", ":"), default=str)`` over the record dict (with ``prev_hash``
    populated, ``record_hash`` excluded). The first record in a fresh file chains
    from a genesis value of 64 zeros. On process restart, the chain resumes from
    the last line's ``record_hash``. On rotation, the new file's first record
    chains from the rotated file's last hash — the chain spans rotations, so
    verifying a rotated set means processing the files in chronological order.
    """

    def __init__(
        self,
        path: str = _DEFAULT_PATH,
        rotate_mb: float = _DEFAULT_ROTATE_MB,
        include_payload: bool = True,
        chained: bool = True,
    ) -> None:
        self.path = Path(path)
        self._rotate_bytes = int(rotate_mb * 1024 * 1024) if rotate_mb > 0 else 0
        self._include_payload = include_payload
        self._chained = chained
        self._lock = threading.Lock()
        self._last_hash = _GENESIS_HASH

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._chained:
            self._load_last_hash()
        logger.info(
            "[AuditLogger] Writing to %s (rotate_mb=%s, chained=%s)",
            self.path,
            rotate_mb,
            chained,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, event: SecurityEvent) -> None:
        """Write a single event. Safe to call from any thread."""
        self.write(event)

    def write(self, event: SecurityEvent) -> None:
        """Append a SecurityEvent as a single JSON line."""
        record = self._to_record(event)

        with self._lock:
            if self._rotate_bytes and self._should_rotate():
                self._rotate()
            if self._chained:
                record = dict(record)
                record["prev_hash"] = self._last_hash
                record["record_hash"] = self._compute_hash(record)
                self._last_hash = record["record_hash"]
            line = json.dumps(record, default=str) + "\n"
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)

    def verify(self, path: Optional[str] = None) -> ChainVerificationResult:
        """Walk an audit file and verify its hash chain is intact.

        Recomputes each line's expected ``record_hash`` from its content plus
        the previous line's ``record_hash``; the first mismatch — whether from
        an edited record or a deleted line breaking the ``prev_hash`` link —
        is reported as the tamper point.

        Parameters
        ----------
        path:
            File to verify. Defaults to this logger's own path.
        """
        target = Path(path) if path is not None else self.path
        if not target.exists():
            return ChainVerificationResult(
                valid=True,
                records_checked=0,
                first_broken_line=None,
                detail=f"{target} does not exist — nothing to verify",
            )

        lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
        prev_hash: Optional[str] = None

        for line_no, line in enumerate(lines, start=1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                return ChainVerificationResult(
                    valid=False,
                    records_checked=line_no - 1,
                    first_broken_line=line_no,
                    detail=f"Line {line_no} is not valid JSON",
                )

            if "record_hash" not in data or "prev_hash" not in data:
                return ChainVerificationResult(
                    valid=False,
                    records_checked=line_no - 1,
                    first_broken_line=line_no,
                    detail=f"Line {line_no} has no hash-chain fields (written with chained=False?)",
                )

            declared_prev = data["prev_hash"]
            declared_hash = data["record_hash"]

            # The first line's prev_hash is trusted as the chain's starting
            # point — it may be the genesis value, or carried over from a
            # rotated predecessor file that isn't part of this verification.
            expected_prev = declared_prev if line_no == 1 else prev_hash
            if declared_prev != expected_prev:
                return ChainVerificationResult(
                    valid=False,
                    records_checked=line_no - 1,
                    first_broken_line=line_no,
                    detail=(
                        f"Line {line_no} prev_hash does not match the previous "
                        "record's hash — a record was likely deleted"
                    ),
                )

            body = {k: v for k, v in data.items() if k != "record_hash"}
            if self._compute_hash(body) != declared_hash:
                return ChainVerificationResult(
                    valid=False,
                    records_checked=line_no - 1,
                    first_broken_line=line_no,
                    detail=f"Line {line_no} record_hash does not match its content — the record was modified",
                )

            prev_hash = declared_hash

        return ChainVerificationResult(
            valid=True,
            records_checked=len(lines),
            first_broken_line=None,
            detail=f"Chain intact — {len(lines)} records verified",
        )

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
        # self._last_hash is intentionally left untouched — the chain spans
        # rotations, so the new file's first record continues from here.

    @staticmethod
    def _compute_hash(record_with_prev_hash: dict) -> str:
        """Hash a record (which already carries ``prev_hash``, but not ``record_hash``)."""
        canonical = json.dumps(record_with_prev_hash, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256((canonical + record_with_prev_hash["prev_hash"]).encode("utf-8")).hexdigest()

    def _load_last_hash(self) -> None:
        """Resume the hash chain from the last record on disk, if any."""
        if not self.path.exists():
            return
        lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return
        try:
            data = json.loads(lines[-1])
            self._last_hash = data.get("record_hash", _GENESIS_HASH)
        except json.JSONDecodeError:
            logger.warning("[AuditLogger] Could not parse last record in %s — starting a new chain", self.path)
