from contextlib import contextmanager

import psycopg
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.main import protect_admin_endpoints
from src.api.remediation_history import router


app = FastAPI()
app.middleware("http")(protect_admin_endpoints)
app.include_router(router)
client = TestClient(app)


class FakeCursor:
    def __init__(self, records=None):
        self.records = records or []
        self.result = None

    def execute(self, query, params=None):
        if "COUNT(*)" in query:
            self.result = (len(self.records),)
        else:
            limit = params[0]
            self.result = [(record,) for record in self.records[:limit]]

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result


class FakeConnection:
    def __init__(self, records=None):
        self.records = records or []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @contextmanager
    def cursor(self):
        yield FakeCursor(self.records)


def test_admin_endpoint_rejects_missing_token(monkeypatch):
    monkeypatch.setenv("MLPILOT_ADMIN_TOKEN", "test-secret")

    response = client.get("/admin/remediation-history")

    assert response.status_code == 401


def test_admin_endpoint_rejects_invalid_token(monkeypatch):
    monkeypatch.setenv("MLPILOT_ADMIN_TOKEN", "test-secret")

    response = client.get(
        "/admin/remediation-history",
        headers={"X-Admin-Token": "wrong-secret"},
    )

    assert response.status_code == 401


def test_admin_endpoint_accepts_valid_token(monkeypatch):
    monkeypatch.setenv("MLPILOT_ADMIN_TOKEN", "test-secret")

    records = [
        {"action_type": "remove_feature"},
        {"action_type": "change_rolling_window"},
    ]

    monkeypatch.setattr(
        "src.api.remediation_history.psycopg.connect",
        lambda uri: FakeConnection(records),
    )

    response = client.get(
        "/admin/remediation-history",
        headers={"X-Admin-Token": "test-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "total_records": 2,
        "returned_records": 2,
        "records": records,
    }


def test_admin_endpoint_fails_closed_without_configured_token(monkeypatch):
    monkeypatch.delenv("MLPILOT_ADMIN_TOKEN", raising=False)

    response = client.get(
        "/admin/remediation-history",
        headers={"X-Admin-Token": "anything"},
    )

    assert response.status_code == 503


def test_database_error_returns_500(monkeypatch):
    monkeypatch.setenv("MLPILOT_ADMIN_TOKEN", "test-secret")

    def raise_database_error(uri):
        raise psycopg.OperationalError("Database unavailable")

    monkeypatch.setattr(
        "src.api.remediation_history.psycopg.connect",
        raise_database_error,
    )

    response = client.get(
        "/admin/remediation-history",
        headers={"X-Admin-Token": "test-secret"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Could not retrieve remediation history from Neon."
    )