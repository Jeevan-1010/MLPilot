
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.remediation_history import router, HISTORY_PATH


app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_corrupted_history_file_returns_500(tmp_path, monkeypatch):
    corrupted_file = tmp_path / "remediation_history.jsonl"
    corrupted_file.write_text(
        '{"action_type": "remove_feature"}\n'
        '{"action_type": invalid_json}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.api.remediation_history.HISTORY_PATH",
        corrupted_file,
    )

    response = client.get("/admin/remediation-history")

    assert response.status_code == 500
    assert "Invalid JSON" in response.json()["detail"]


def test_missing_history_file_returns_empty_result(tmp_path, monkeypatch):
    missing_file = tmp_path / "missing.jsonl"

    monkeypatch.setattr(
        "src.api.remediation_history.HISTORY_PATH",
        missing_file,
    )

    response = client.get("/admin/remediation-history")

    assert response.status_code == 200
    assert response.json() == {
        "total_records": 0,
        "returned_records": 0,
        "records": [],
    }
