"""Regression tests for secret/PII redaction (Fix 8).

Before this fix, ``make_payload`` stored raw strings verbatim — any API key,
access token, private key, JWT, or email address present in an LLM message or
tool argument would be persisted to the audit trail and shown on the
dashboard unmasked. ``redact()`` now runs inside ``make_payload`` (on by
default) and replaces recognizable secret/PII substrings with
``«REDACTED:<kind>»`` placeholders.
"""

from __future__ import annotations

import pytest

from agentguard.events import make_payload
from agentguard.redaction import (
    is_redaction_enabled,
    redact,
    redact_text,
    set_redaction_enabled,
)


@pytest.fixture(autouse=True)
def _reset_override():
    """Every test starts from a clean slate — no lingering programmatic override."""
    set_redaction_enabled(None)
    yield
    set_redaction_enabled(None)


# ---------------------------------------------------------------------------
# Pattern coverage
# ---------------------------------------------------------------------------

class TestRedactTextPatterns:
    def test_openai_api_key_is_redacted(self):
        text = "use this key: sk-ABCDEFGHIJ1234567890abcdef"
        result = redact_text(text)
        assert "sk-ABCDEFGHIJ1234567890abcdef" not in result
        assert "«REDACTED:openai_api_key»" in result

    def test_aws_access_key_id_is_redacted(self):
        text = "AKIAIOSFODNN7EXAMPLE is the access key"
        result = redact_text(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "«REDACTED:aws_access_key_id»" in result

    def test_github_token_is_redacted(self):
        text = "token=ghp_" + "a" * 36
        result = redact_text(text)
        assert "ghp_" + "a" * 36 not in result
        assert "«REDACTED:github_token»" in result

    def test_jwt_is_redacted(self):
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        result = redact_text(f"Authorization: Bearer {jwt}")
        assert jwt not in result
        assert "«REDACTED:jwt»" in result

    def test_private_key_header_is_redacted(self):
        text = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA1c7+9z5Pad7OejecsQ0bu3aozN3tihPI\n"
            "-----END RSA PRIVATE KEY-----"
        )
        result = redact_text(text)
        assert "MIIEpAIBAAKCAQEA1c7" not in result
        assert "«REDACTED:private_key»" in result

    def test_email_is_redacted(self):
        text = "Contact shashank@example.com for access"
        result = redact_text(text)
        assert "shashank@example.com" not in result
        assert "«REDACTED:email»" in result

    def test_clean_text_is_unchanged(self):
        text = "Please write a haiku about autumn leaves."
        assert redact_text(text) == text

    def test_multiple_secrets_in_one_string_all_redacted(self):
        text = "key=sk-ABCDEFGHIJ1234567890abcdef email=person@example.com"
        result = redact_text(text)
        assert "sk-ABCDEFGHIJ1234567890abcdef" not in result
        assert "person@example.com" not in result
        assert result.count("«REDACTED:") == 2


# ---------------------------------------------------------------------------
# Structural redaction (dict/list walking)
# ---------------------------------------------------------------------------

class TestRedactStructure:
    def test_redacts_nested_dicts_and_lists(self):
        value = {
            "tool_input": {
                "headers": {"Authorization": "Bearer sk-ABCDEFGHIJ1234567890abcdef"},
                "recipients": ["a@example.com", "clean text here"],
            },
            "note": "no secrets here",
        }
        result = redact(value)
        assert "sk-ABCDEFGHIJ1234567890abcdef" not in str(result)
        assert "a@example.com" not in str(result)
        assert result["note"] == "no secrets here"
        assert result["tool_input"]["recipients"][1] == "clean text here"

    def test_keys_are_left_intact(self):
        value = {"sk-looks-like-a-key-but-is-a-key-name": "clean value"}
        result = redact(value)
        assert "sk-looks-like-a-key-but-is-a-key-name" in result

    def test_non_string_values_pass_through(self):
        value = {"count": 5, "ratio": 0.5, "active": True, "missing": None}
        assert redact(value) == value


# ---------------------------------------------------------------------------
# make_payload() integration
# ---------------------------------------------------------------------------

class TestMakePayloadRedaction:
    def test_make_payload_redacts_by_default(self):
        payload = make_payload(message="my key is sk-ABCDEFGHIJ1234567890abcdef")
        assert "sk-ABCDEFGHIJ1234567890abcdef" not in payload["message"]
        assert "«REDACTED:openai_api_key»" in payload["message"]

    def test_make_payload_redacts_nested_tool_input(self):
        payload = make_payload(
            tool_name="send_email",
            tool_input={"to": "person@example.com", "body": "hello"},
        )
        assert "person@example.com" not in str(payload)
        assert "«REDACTED:email»" in str(payload)

    def test_make_payload_truncation_still_applies(self):
        payload = make_payload(big=("x" * 5000))
        assert len(payload["big"]) <= 4001  # 4000 chars + ellipsis


# ---------------------------------------------------------------------------
# Toggle: AGENTGUARD_REDACT env var + programmatic override
# ---------------------------------------------------------------------------

class TestRedactionToggle:
    def test_enabled_by_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("AGENTGUARD_REDACT", raising=False)
        assert is_redaction_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "False", "no", "off", "OFF"])
    def test_env_var_falsy_values_disable_redaction(self, monkeypatch, value):
        monkeypatch.setenv("AGENTGUARD_REDACT", value)
        assert is_redaction_enabled() is False
        # redact() is toggle-aware (unlike the raw redact_text() primitive).
        assert redact("sk-ABCDEFGHIJ1234567890abcdef") == "sk-ABCDEFGHIJ1234567890abcdef"

    @pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on"])
    def test_env_var_truthy_values_enable_redaction(self, monkeypatch, value):
        monkeypatch.setenv("AGENTGUARD_REDACT", value)
        assert is_redaction_enabled() is True

    def test_disabled_via_env_var_make_payload_keeps_raw_value(self, monkeypatch):
        monkeypatch.setenv("AGENTGUARD_REDACT", "false")
        payload = make_payload(message="my key is sk-ABCDEFGHIJ1234567890abcdef")
        assert payload["message"] == "my key is sk-ABCDEFGHIJ1234567890abcdef"

    def test_programmatic_override_takes_precedence_over_env_var(self, monkeypatch):
        monkeypatch.setenv("AGENTGUARD_REDACT", "true")
        set_redaction_enabled(False)
        assert is_redaction_enabled() is False
        assert redact("sk-ABCDEFGHIJ1234567890abcdef") == "sk-ABCDEFGHIJ1234567890abcdef"

    def test_clearing_override_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv("AGENTGUARD_REDACT", "false")
        set_redaction_enabled(True)
        assert is_redaction_enabled() is True
        set_redaction_enabled(None)
        assert is_redaction_enabled() is False
