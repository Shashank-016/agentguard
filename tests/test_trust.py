"""Tests for TrustScorer — score degradation and flagging."""

import pytest
from agentguard.engine.trust import TrustScorer, TRUSTED, INTERNAL, EXTERNAL, UNTRUSTED


@pytest.fixture
def scorer() -> TrustScorer:
    return TrustScorer(sensitive_threshold=0.5)


class TestInitialState:
    def test_new_session_starts_trusted(self, scorer):
        assert scorer.score("new-session") == TRUSTED

    def test_trust_label_is_trusted(self, scorer):
        assert scorer.trust_label("new-session") == "TRUSTED"

    def test_not_flagged_initially(self, scorer):
        assert scorer.is_flagged("new-session") is False


class TestExternalContentDegradation:
    def test_single_external_degrades_score(self, scorer):
        scorer.record_external_content("s1", "web")
        assert scorer.score("s1") == pytest.approx(TRUSTED * EXTERNAL, abs=0.001)

    def test_double_external_degrades_multiplicatively(self, scorer):
        scorer.record_external_content("s2", "file")
        scorer.record_external_content("s2", "file")
        expected = round(TRUSTED * EXTERNAL * EXTERNAL, 4)
        assert scorer.score("s2") == pytest.approx(expected, abs=0.001)

    def test_trust_label_after_external(self, scorer):
        scorer.record_external_content("s3", "web")
        assert scorer.trust_label("s3") == "EXTERNAL"

    def test_history_recorded(self, scorer):
        scorer.record_external_content("s4", "file")
        history = scorer.history("s4")
        assert len(history) == 1
        assert history[0].event_type == "external_content"
        assert history[0].trust_delta < 0


class TestAgentHandoff:
    def test_handoff_after_external_degrades_further(self, scorer):
        scorer.record_external_content("s5", "web")
        score_before_handoff = scorer.score("s5")
        scorer.record_agent_handoff("s5", "researcher", "writer")
        assert scorer.score("s5") < score_before_handoff

    def test_handoff_from_trusted_session_no_degradation(self, scorer):
        score_before = scorer.score("s6")
        scorer.record_agent_handoff("s6", "a", "b")
        # If already at TRUSTED, handoff with INTERNAL multiplier doesn't degrade.
        assert scorer.score("s6") == score_before


class TestInjectionFlag:
    def test_injection_sets_score_to_zero(self, scorer):
        scorer.record_external_content("s7", "file")
        scorer.record_injection_flag("s7")
        assert scorer.score("s7") == UNTRUSTED

    def test_injection_sets_flagged(self, scorer):
        scorer.record_injection_flag("s8")
        assert scorer.is_flagged("s8") is True

    def test_injection_history_entry(self, scorer):
        scorer.record_injection_flag("s9")
        history = scorer.history("s9")
        assert any(e.event_type == "injection_flag" for e in history)


class TestShouldFlag:
    def test_trusted_session_write_not_flagged(self, scorer):
        assert scorer.should_flag("safe-session", "write_file") is False

    def test_low_trust_write_file_flagged(self, scorer):
        scorer.record_external_content("poisoned", "web")
        # score is now 0.3, below threshold of 0.5
        assert scorer.should_flag("poisoned", "write_file") is True

    def test_low_trust_read_file_not_flagged(self, scorer):
        scorer.record_external_content("read-session", "web")
        # read_file has no sensitive keywords
        assert scorer.should_flag("read-session", "read_file") is False

    def test_low_trust_execute_code_flagged(self, scorer):
        scorer.record_external_content("exec-session", "web")
        assert scorer.should_flag("exec-session", "execute_code") is True

    def test_low_trust_send_email_flagged(self, scorer):
        scorer.record_external_content("mail-session", "file")
        assert scorer.should_flag("mail-session", "send_email") is True


class TestSummary:
    def test_summary_structure(self, scorer):
        scorer.record_external_content("sum-session", "file")
        summary = scorer.summary("sum-session")
        assert "session_id" in summary
        assert "score" in summary
        assert "label" in summary
        assert "flagged" in summary
        assert "history" in summary
        assert isinstance(summary["history"], list)

    def test_summary_reflects_degradation(self, scorer):
        scorer.record_external_content("sum2", "web")
        summary = scorer.summary("sum2")
        assert summary["score"] < 1.0
        assert summary["label"] != "TRUSTED"
