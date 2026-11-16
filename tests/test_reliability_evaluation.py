"""In-process API reliability checks; no Render or external services are contacted."""
import asyncio

import httpx

from src.api import main as api


class FlakyModel:
    def __init__(self):
        self.fail_once = True

    def predict(self, frame):
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("controlled one-time inference failure")
        return [42.0]


async def _exercise_api():
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        health_responses = [await client.get("/health") for _ in range(20)]
        valid_payload = {"lag_1": 1.0}
        failed_prediction = await client.post("/predict", json=valid_payload)
        recovered_prediction = await client.post("/predict", json=valid_payload)
        invalid_prediction = await client.post("/predict", json={})
        return health_responses, failed_prediction, recovered_prediction, invalid_prediction


def test_api_availability_error_handling_and_recovery(monkeypatch):
    monkeypatch.setattr(api, "model", FlakyModel())
    monkeypatch.setattr(api, "model_version", "fixture-v1")
    monkeypatch.setattr(api, "model_schema", {"feature_names": ["lag_1"]})
    monkeypatch.setattr(api, "request_count", 0)
    monkeypatch.setattr(api, "total_latency", 0.0)
    monkeypatch.setattr(api, "max_latency", 0.0)

    health, first, recovered, invalid = asyncio.run(_exercise_api())
    healthy = sum(
        response.status_code == 200 and response.json()["status"] == "healthy"
        for response in health
    )
    availability = healthy / len(health)
    assert availability == 1.0
    assert first.status_code == 500
    assert recovered.status_code == 200
    assert invalid.status_code == 400
    assert recovered.headers.get("X-Process-Time-ms") is not None

    recovery_rate = 1.0 if recovered.status_code == 200 else 0.0
    print(
        f"MLPILOT_API_INPROCESS health_checks={len(health)} "
        f"availability={availability:.3f} controlled_error_status={first.status_code} "
        f"recovered_on_next_request={recovery_rate:.3f} "
        f"invalid_input_status={invalid.status_code}"
    )
