"""Offline MLOps pipeline, deployment and rollback evaluations."""
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.pipeline import train_pipeline as pipeline
from src.pipeline import deploy_model as deployment


def _synthetic_training_frame():
    rng = np.random.default_rng(20261115)
    x1 = rng.uniform(-3, 3, 180)
    x2 = rng.uniform(-2, 2, 180)
    noise = rng.normal(0, 0.15, 180)
    y = 4 * np.sin(x1) + 2 * (x2 ** 2) + noise
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=180, freq="D"),
        "signal_a": x1,
        "signal_b": x2,
        "daily_demand": y,
    })


def test_pipeline_execution_and_reproducibility(monkeypatch):
    frame = _synthetic_training_frame()
    baseline = RandomForestRegressor(
        n_estimators=8, max_depth=1, random_state=7, n_jobs=1,
    ).fit(frame[["signal_a", "signal_b"]], frame["daily_demand"])
    monkeypatch.setattr(pipeline, "get_production_model", lambda: (baseline, "fixture-1"))

    params = {
        "n_estimators": 80,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    }
    durations = []
    results = []
    for _ in range(2):
        started = time.perf_counter()
        result = pipeline.run_pipeline(
            feature_data=frame,
            feature_columns=["signal_a", "signal_b"],
            candidate_params=params,
            register_candidate=False,
        )
        durations.append((time.perf_counter() - started) * 1000)
        results.append(result)

    assert all(item["status"] == "evaluation_passed" for item in results)
    assert all(item["registered"] is False for item in results)
    first = results[0]["candidate_metrics"]
    second = results[1]["candidate_metrics"]
    reproducible = all(
        first[key] == pytest.approx(second[key], rel=0, abs=1e-12)
        for key in ("MAE", "RMSE")
    )
    assert reproducible
    avg_ms = sum(durations) / len(durations)
    print(
        f"MLPILOT_PIPELINE execution_avg_ms={avg_ms:.2f} "
        f"reproducibility={int(reproducible)} runs={len(results)} "
        f"candidate_mae={first['MAE']:.4f} candidate_rmse={first['RMSE']:.4f}"
    )


class FakeVersion:
    def __init__(self, version):
        self.version = str(version)


class FakeRegistryClient:
    def __init__(self, version="1"):
        self.alias = str(version)

    def get_model_version_by_alias(self, name, alias):
        return FakeVersion(self.alias)

    def set_registered_model_alias(self, name, alias, version):
        self.alias = str(version)


def _prepare_deployment_mocks(monkeypatch, client):
    monkeypatch.setattr(deployment, "configure_mlflow", lambda: None)
    monkeypatch.setattr(deployment, "MlflowClient", lambda: client)
    def promote(version, approved=False):
        assert approved is True
        client.set_registered_model_alias(deployment.MODEL_NAME, "production", version)
        return {"status": "promoted", "previous_version": "1", "deployed_version": str(version)}
    monkeypatch.setattr(deployment, "promote_model", promote)


def test_deployment_success_with_mocked_registry(monkeypatch):
    client = FakeRegistryClient("1")
    _prepare_deployment_mocks(monkeypatch, client)
    result = deployment.deploy_model(
        "2", approved=True, reload_callback=lambda: None,
        verify_callback=lambda version: str(version) == "2",
    )
    assert result["status"] == "success"
    assert client.alias == "2"
    print("MLPILOT_DEPLOYMENT_SIMULATION scenarios=1 success=1 success_rate=1.000")


def test_deployment_failure_rolls_back_to_previous_version(monkeypatch):
    client = FakeRegistryClient("1")
    _prepare_deployment_mocks(monkeypatch, client)
    def verify(version):
        return str(version) == "1"
    result = deployment.deploy_model(
        "2", approved=True, reload_callback=lambda: None,
        verify_callback=verify,
    )
    assert result["status"] == "rolled_back"
    assert result["restored_version"] == "1"
    assert client.alias == "1"
    print("MLPILOT_ROLLBACK_SIMULATION scenarios=1 rollback_restored=1 recovery_rate=1.000")


def test_deployment_requires_explicit_approval():
    with pytest.raises(PermissionError):
        deployment.deploy_model("2", approved=False)
