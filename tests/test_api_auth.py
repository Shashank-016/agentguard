"""Regression tests for optional API-key authentication (Fix 3).

Before this fix, every route on the audit API — including the kill switch —
was reachable by anyone who could route to the process. ``AGENTGUARD_API_KEY``
now gates every router except ``/health`` (kept open for liveness probes).
For backward compatibility, leaving the env var unset keeps the API open
(with a logged warning) so existing deployments don't break on upgrade.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AGENTGUARD_DB_URL", "sqlite+aiosqlite:///:memory:")
    with TestClient(app) as test_client:
        yield test_client


class TestHealthIsAlwaysOpen:
    def test_health_accessible_without_key_when_key_configured(self, client, monkeypatch):
        monkeypatch.setenv("AGENTGUARD_API_KEY", "secret-123")
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_accessible_without_key_when_unconfigured(self, client, monkeypatch):
        monkeypatch.delenv("AGENTGUARD_API_KEY", raising=False)
        response = client.get("/health")
        assert response.status_code == 200


class TestBackwardCompatibilityWhenUnconfigured:
    def test_protected_route_allowed_without_key_when_unconfigured(self, client, monkeypatch):
        monkeypatch.delenv("AGENTGUARD_API_KEY", raising=False)
        response = client.get("/events")
        assert response.status_code == 200


class TestProtectedRoutesWhenKeyConfigured:
    def test_missing_key_is_rejected(self, client, monkeypatch):
        monkeypatch.setenv("AGENTGUARD_API_KEY", "secret-123")
        response = client.get("/events")
        assert response.status_code == 401

    def test_wrong_key_is_rejected(self, client, monkeypatch):
        monkeypatch.setenv("AGENTGUARD_API_KEY", "secret-123")
        response = client.get("/events", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401

    def test_correct_x_api_key_header_is_accepted(self, client, monkeypatch):
        monkeypatch.setenv("AGENTGUARD_API_KEY", "secret-123")
        response = client.get("/events", headers={"X-API-Key": "secret-123"})
        assert response.status_code == 200

    def test_correct_bearer_token_is_accepted(self, client, monkeypatch):
        monkeypatch.setenv("AGENTGUARD_API_KEY", "secret-123")
        response = client.get("/events", headers={"Authorization": "Bearer secret-123"})
        assert response.status_code == 200

    def test_malformed_bearer_token_is_rejected(self, client, monkeypatch):
        monkeypatch.setenv("AGENTGUARD_API_KEY", "secret-123")
        response = client.get("/events", headers={"Authorization": "Basic secret-123"})
        assert response.status_code == 401

    def test_sessions_route_is_protected(self, client, monkeypatch):
        monkeypatch.setenv("AGENTGUARD_API_KEY", "secret-123")
        assert client.get("/sessions").status_code == 401
        assert client.get("/sessions", headers={"X-API-Key": "secret-123"}).status_code == 200

    def test_control_route_is_protected(self, client, monkeypatch):
        monkeypatch.setenv("AGENTGUARD_API_KEY", "secret-123")
        assert client.get("/control/status").status_code == 401
        assert client.get("/control/status", headers={"X-API-Key": "secret-123"}).status_code == 200
