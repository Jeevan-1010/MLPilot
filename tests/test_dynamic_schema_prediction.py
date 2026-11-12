
import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.main as api
from src.api.main import app


class DummyModel:
    def predict(self, dataframe):
        self.received_columns = list(dataframe.columns)
        self.received_values = dataframe.iloc[0].tolist()
        return np.array([123.45])


@pytest.fixture
def dynamic_api():
    with TestClient(app) as client:
        dummy_model = DummyModel()

        original_model = api.model
        original_version = api.model_version
        original_schema = api.model_schema

        api.model = dummy_model
        api.model_version = "test-schema"
        api.model_schema = {
            "schema_version": 1,
            "feature_names": [
                "rolling_mean_7",
                "lag_1",
            ],
            "target": "daily_demand",
            "rolling_windows": {
                "rolling_mean_7": 7,
            },
        }

        yield client, dummy_model

        api.model = original_model
        api.model_version = original_version
        api.model_schema = original_schema


def test_valid_dynamic_prediction(dynamic_api):
    client, dummy_model = dynamic_api

    response = client.post(
        "/predict",
        json={
            "rolling_mean_7": 11800,
            "lag_1": 12000,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "predicted_demand": 123.45,
        "model_version": "test-schema",
    }

    assert dummy_model.received_columns == [
        "rolling_mean_7",
        "lag_1",
    ]

    assert dummy_model.received_values == [
        11800.0,
        12000.0,
    ]

    print("Valid dynamic prediction passed!")


def test_missing_feature_rejected(dynamic_api):
    client, _ = dynamic_api

    response = client.post(
        "/predict",
        json={
            "rolling_mean_7": 11800,
        },
    )

    assert response.status_code == 400
    assert "lag_1" in response.json()["detail"]

    print("Missing feature rejection passed!")


def test_unexpected_feature_rejected(dynamic_api):
    client, _ = dynamic_api

    response = client.post(
        "/predict",
        json={
            "rolling_mean_7": 11800,
            "lag_1": 12000,
            "month": 5,
        },
    )

    assert response.status_code == 400
    assert "month" in response.json()["detail"]

    print("Unexpected feature rejection passed!")



def test_nan_input_rejected(dynamic_api):
    client, _ = dynamic_api

    response = client.post(
        "/predict",
        content='{"rolling_mean_7": NaN, "lag_1": 12000}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert "rolling_mean_7" in response.json()["detail"]

    print("NaN input rejection passed!")


def test_infinity_input_rejected(dynamic_api):
    client, _ = dynamic_api

    response = client.post(
        "/predict",
        content='{"rolling_mean_7": Infinity, "lag_1": 12000}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert "rolling_mean_7" in response.json()["detail"]

    print("Infinity input rejection passed!")


def test_non_numeric_input_rejected(dynamic_api):
    client, _ = dynamic_api

    response = client.post(
        "/predict",
        json={
            "rolling_mean_7": "invalid",
            "lag_1": 12000,
        },
    )

    assert response.status_code == 422

    print("Non-numeric input rejection passed!")



def test_boolean_input_rejected(dynamic_api):
    client, _ = dynamic_api

    response = client.post(
        "/predict",
        json={
            "rolling_mean_7": True,
            "lag_1": 12000,
        },
    )

    assert response.status_code == 422
    assert "rolling_mean_7" in response.json()["detail"]

    print("Boolean input rejection passed!")



class FailingModel:
    def predict(self, dataframe):
        raise RuntimeError("Simulated model failure")


def test_model_prediction_failure(dynamic_api):
    client, _ = dynamic_api

    api.model = FailingModel()

    response = client.post(
        "/predict",
        json={
            "rolling_mean_7": 11800,
            "lag_1": 12000,
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Model prediction failed."

    print("Model prediction failure handling passed!")



def test_extremely_large_positive_input(dynamic_api):
    client, _ = dynamic_api

    response = client.post(
        "/predict",
        json={
            "rolling_mean_7": 1e100,
            "lag_1": 12000,
        },
    )

    assert response.status_code == 200
    assert response.json()["predicted_demand"] == 123.45

    print("Extremely large positive input handled!")


def test_extremely_large_negative_input(dynamic_api):
    client, _ = dynamic_api

    response = client.post(
        "/predict",
        json={
            "rolling_mean_7": -1e100,
            "lag_1": 12000,
        },
    )

    assert response.status_code == 200
    assert response.json()["predicted_demand"] == 123.45

    print("Extremely large negative input handled!")




class NonFiniteModel:
    def predict(self, dataframe):
        return np.array([float("nan")])


def test_non_finite_model_prediction_rejected(dynamic_api):
    client, _ = dynamic_api

    api.model = NonFiniteModel()

    response = client.post(
        "/predict",
        json={
            "rolling_mean_7": 11800,
            "lag_1": 12000,
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Model returned a non-finite prediction."

    print("Non-finite model prediction rejection passed!")
