"""Tests for AuditLogger's hash-chained tamper-evident mode."""

from __future__ import annotations

import json

from agentguard.audit import _GENESIS_HASH, AuditLogger, ChainVerificationResult
from agentguard.events import SecurityEvent


def _make_event(event_type: str = "llm_call", severity: str = "info") -> SecurityEvent:
    return SecurityEvent(
        session_id="test-session",
        agent_id="test-agent",
        source="sdk",
        event_type=event_type,
        severity=severity,
        payload={"n": 1},
    )


def _read_records(path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# ---------------------------------------------------------------------------
# Writing — chain fields
# ---------------------------------------------------------------------------


class TestChainedWrites:
    def test_records_carry_hash_fields(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path))
        for _ in range(5):
            log.write(_make_event())

        records = _read_records(path)
        assert len(records) == 5
        for record in records:
            assert "prev_hash" in record
            assert "record_hash" in record
            assert len(record["record_hash"]) == 64

    def test_genesis_prev_hash(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path))
        log.write(_make_event())
        records = _read_records(path)
        assert records[0]["prev_hash"] == _GENESIS_HASH

    def test_chain_links_consecutive_records(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path))
        for _ in range(5):
            log.write(_make_event())

        records = _read_records(path)
        for i in range(1, len(records)):
            assert records[i]["prev_hash"] == records[i - 1]["record_hash"]

    def test_chained_false_reproduces_old_format(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path), chained=False)
        log.write(_make_event())
        records = _read_records(path)
        assert "prev_hash" not in records[0]
        assert "record_hash" not in records[0]


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------


class TestVerify:
    def test_untampered_chain_is_valid(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path))
        for _ in range(10):
            log.write(_make_event())

        result = log.verify()
        assert isinstance(result, ChainVerificationResult)
        assert result.valid is True
        assert result.records_checked == 10
        assert result.first_broken_line is None

    def test_empty_or_missing_file_is_valid(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        log = AuditLogger(path=str(tmp_path / "audit.jsonl"))
        result = log.verify(str(path))
        assert result.valid is True
        assert result.records_checked == 0

    def test_edited_middle_line_breaks_chain(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path))
        for _ in range(5):
            log.write(_make_event())

        lines = path.read_text(encoding="utf-8").splitlines()
        tampered = json.loads(lines[2])
        tampered["payload"] = {"n": 9999}
        lines[2] = json.dumps(tampered)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = log.verify()
        assert result.valid is False
        assert result.first_broken_line == 3

    def test_deleted_middle_line_breaks_chain_at_following_line(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path))
        for _ in range(5):
            log.write(_make_event())

        lines = path.read_text(encoding="utf-8").splitlines()
        del lines[2]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = log.verify()
        assert result.valid is False
        # Deleted line was originally #3 — its successor (now physically the
        # 3rd line) is where the broken prev_hash link is detected.
        assert result.first_broken_line == 3

    def test_chain_continues_across_restart(self, tmp_path):
        path = tmp_path / "audit.jsonl"

        log1 = AuditLogger(path=str(path))
        for _ in range(3):
            log1.write(_make_event())
        del log1

        log2 = AuditLogger(path=str(path))
        for _ in range(2):
            log2.write(_make_event())

        records = _read_records(path)
        assert len(records) == 5
        for i in range(1, len(records)):
            assert records[i]["prev_hash"] == records[i - 1]["record_hash"]

        result = log2.verify()
        assert result.valid is True
        assert result.records_checked == 5

    def test_chained_false_reports_missing_hash_fields(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLogger(path=str(path), chained=False)
        log.write(_make_event())

        result = log.verify()
        assert result.valid is False
        assert result.first_broken_line == 1
