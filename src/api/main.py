import math
import json
import mlflow
import mlflow.pyfunc
import pandas as pd
import time
import threading
from typing import Any
import mlflow.sklearn

from fastapi import FastAPI, Request, HTTPException
from mlflow.tracking import MlflowClient

from src.api.remediation_history import router as remediation_history_router


MODEL_NAME = "MLPilot-Demand-Forecasting"
MODEL_ALIAS = "production"

FEATURES = [
    "day_of_week",
    "month",
    "day_of_month",
    "week_of_year",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
]

TARGET = "daily_demand"

ROLLING_FEATURES = {
    "rolling_mean_7",
    "rolling_mean_14",
}

app = FastAPI(
    title="MLPilot API",
    description="MLPilot - ML Model Serving API",
    version="2.1.0",
)

app.include_router(remediation_history_router)

model = None
model_version = None
model_schema = None
model_lock = threading.RLock()

request_count = 0
total_latency = 0.0
max_latency = 0.0


def load_model_schema(client, production):
    """
    Load and validate the feature schema from MLflow.
    Fall back to the baseline schema for older model versions.
    """

    artifacts = client.list_artifacts(production.run_id)

    schema_exists = any(
        artifact.path == "feature_schema.json"
        for artifact in artifacts
    )

    # Backward compatibility for older model versions.
    if not schema_exists:
        print(
            f"Version {production.version} has no stored schema. "
            "Using legacy baseline schema."
        )

        return {
            "schema_version": 0,
            "feature_names": FEATURES.copy(),
            "target": TARGET,
            "rolling_windows": {
                "rolling_mean_7": 7,
                "rolling_mean_14": 14,
            },
        }

    artifact_path = client.download_artifacts(
        production.run_id,
        "feature_schema.json",
    )

    with open(artifact_path, "r", encoding="utf-8") as file:
        schema = json.load(file)

    if not isinstance(schema, dict):
        raise ValueError(
            "Invalid feature schema: expected a JSON object."
        )

    # Validate schema version.
    schema_version = schema.get("schema_version")

    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError(
            f"Unsupported schema version: {schema_version}"
        )

    # Validate feature names.
    feature_names = schema.get("feature_names")

    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError(
            "Invalid feature schema: feature_names must be a non-empty list."
        )

    if any(
        not isinstance(name, str) or not name
        for name in feature_names
    ):
        raise ValueError(
            "Invalid feature schema: invalid feature name."
        )

    if len(feature_names) != len(set(feature_names)):
        raise ValueError(
            "Invalid feature schema: duplicate features."
        )

    unsupported_features = [
        name for name in feature_names
        if name not in FEATURES
    ]

    if unsupported_features:
        raise ValueError(
            f"Unsupported features in schema: {unsupported_features}"
        )

    # Validate target.
    if schema.get("target") != TARGET:
        raise ValueError(
            f"Invalid feature schema: expected target '{TARGET}'."
        )

    # Validate rolling windows.
    rolling_windows = schema.get("rolling_windows")

    if not isinstance(rolling_windows, dict):
        raise ValueError(
            "Invalid feature schema: rolling_windows must be a dictionary."
        )

    for feature, window in rolling_windows.items():
        if feature not in ROLLING_FEATURES:
            raise ValueError(
                f"Unsupported rolling feature: {feature}"
            )

        if (
            isinstance(window, bool)
            or not isinstance(window, int)
            or not 2 <= window <= 30
        ):
            raise ValueError(
                f"Invalid rolling window for {feature}: {window}"
            )

        if feature not in feature_names:
            raise ValueError(
                f"Rolling feature {feature} is not in feature_names."
            )

    for feature in feature_names:
        if (
            feature in ROLLING_FEATURES
            and feature not in rolling_windows
        ):
            raise ValueError(
                f"Missing rolling window configuration for {feature}."
            )

    return schema


def load_production_model():
    global model, model_version, model_schema

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    production = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )

    new_version = str(production.version)

    with model_lock:
        if model is not None and model_version == new_version:
            return False

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    new_model = mlflow.sklearn.load_model(model_uri)
    new_model.set_params(n_jobs=1)

    print("Serving model configured with n_jobs=1")

    new_schema = load_model_schema(client, production)

    if new_schema["schema_version"] > 0:
        print("Loaded feature schema from MLflow.")

    # Update model and schema together.
    with model_lock:
        model = new_model
        model_version = new_version
        model_schema = new_schema

    print(f"Loaded production model version {new_version}")
    print("Expected features:", model_schema["feature_names"])

    return True


@app.on_event("startup")
def startup_event():
    load_production_model()


@app.middleware("http")
async def monitor_latency(request: Request, call_next):
    global request_count, total_latency, max_latency

    start = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - start) * 1000

    if request.url.path != "/metrics":
        request_count += 1
        total_latency += latency
        max_latency = max(max_latency, latency)

    response.headers["X-Process-Time-ms"] = str(round(latency, 2))

    return response


@app.get("/")
def home():
    return {
        "message": "MLPilot API is running",
        "model_version": model_version,
    }


@app.get("/health")
def health():
    with model_lock:
        active_model = model
        active_version = model_version
        active_schema = model_schema

    return {
        "status": "healthy" if active_model is not None else "unhealthy",
        "model_loaded": active_model is not None,
        "model_version": active_version,
        "schema_loaded": active_schema is not None,
    }


@app.get("/metrics")
def metrics():
    average = total_latency / request_count if request_count else 0

    return {
        "total_requests": request_count,
        "average_latency_ms": round(average, 2),
        "max_latency_ms": round(max_latency, 2),
    }


@app.post("/admin/reload-model")
def reload_model():
    try:
        changed = load_production_model()

        return {
            "success": True,
            "model_reloaded": changed,
            "model_version": model_version,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {error}",
        )




@app.post("/predict")
def predict(request: dict[str, Any]):
    endpoint_start = time.perf_counter()

    with model_lock:
        if model is None or model_schema is None:
            raise HTTPException(
                status_code=503,
                detail="No model or feature schema loaded.",
            )

        active_model = model
        active_version = model_version
        active_schema = model_schema.copy()

    feature_names = active_schema["feature_names"]

    missing_features = [
        feature for feature in feature_names if feature not in request
    ]
    unexpected_features = [
        feature for feature in request if feature not in feature_names
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing model features: {missing_features}",
        )

    if unexpected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected features: {unexpected_features}",
        )

    invalid_values = [
        feature
        for feature, value in request.items()
        if isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ]

    if invalid_values:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid numerical values: {invalid_values}",
        )

    preprocessing_start = time.perf_counter()

    input_data = pd.DataFrame(
        [[request[feature] for feature in feature_names]],
        columns=feature_names,
    )

    preprocessing_ms = (
        time.perf_counter() - preprocessing_start
    ) * 1000

    inference_start = time.perf_counter()

    try:
        prediction = float(active_model.predict(input_data)[0])
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Model prediction failed.",
        )

    inference_ms = (
        time.perf_counter() - inference_start
    ) * 1000

    if not math.isfinite(prediction):
        raise HTTPException(
            status_code=500,
            detail="Model returned a non-finite prediction.",
        )

    endpoint_ms = (
        time.perf_counter() - endpoint_start
    ) * 1000

    print(
        f"[PREDICTION PROFILE] "
        f"Preprocessing: {preprocessing_ms:.3f} ms | "
        f"Inference: {inference_ms:.3f} ms | "
        f"Endpoint: {endpoint_ms:.3f} ms"
    )

    return {
        "predicted_demand": round(prediction, 2),
        "model_version": active_version,
    }
