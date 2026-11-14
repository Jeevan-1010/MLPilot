
import math
import json
import os
import hmac
import time
import threading
import tempfile
import uuid
from contextlib import contextmanager

from typing import Any

from langgraph.types import Command
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from src.agents.investigator import build_investigator_graph

import boto3
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd

from src.tracking.mlflow_config import configure_mlflow
from src.monitoring.monitoring_pipeline import run_monitoring, prepare_investigator_data
from src.api.remediation_history import router as remediation_history_router

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient


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

MONITORING_BUCKET_ENV = "MLFLOW_ARTIFACTS_BUCKET"
MONITORING_OBJECT_KEY = "monitoring/prepwise_cleaned_dataset.csv"


app = FastAPI(
    title="MLPilot API",
    description="MLPilot - ML Model Serving API",
    version="2.1.0",
)

app.include_router(remediation_history_router)


@app.middleware("http")
async def protect_admin_endpoints(request: Request, call_next):
    if request.url.path.startswith("/admin/"):
        admin_token = os.getenv("MLPILOT_ADMIN_TOKEN")

        if not admin_token:
            return JSONResponse(
                status_code=503,
                content={"detail": "Admin authentication is not configured."},
            )

        supplied_token = request.headers.get("X-Admin-Token", "")

        if not hmac.compare_digest(supplied_token, admin_token):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing admin token."},
            )

    return await call_next(request)


model = None
model_version = None
model_schema = None
checkpoint_pool = None
investigator_graph = None
model_lock = threading.RLock()
investigation_lock = threading.Lock()
active_investigation_thread = None

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

    configure_mlflow()
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




@contextmanager
def production_data_context():
    """Acquire the B2 dataset for one remediation execution."""
    required = [
        MONITORING_BUCKET_ENV,
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"B2 storage configuration is incomplete: {missing}")

    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = os.path.join(temp_dir, "investigator_dataset.csv")
        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name=os.environ["AWS_DEFAULT_REGION"],
        )
        s3.download_file(
            os.environ[MONITORING_BUCKET_ENV],
            MONITORING_OBJECT_KEY,
            data_path,
        )
        yield data_path


def deploy_candidate_model(
    target_version,
    approved=False,
    expected_previous_version=None,
):
    """Deploy within this API process and verify the in-memory serving model."""
    from src.pipeline.deploy_model import deploy_model

    def verify_loaded_version(expected_version):
        with model_lock:
            return (
                model is not None
                and model_schema is not None
                and str(model_version) == str(expected_version)
            )

    return deploy_model(
        target_version,
        approved=approved,
        expected_previous_version=expected_previous_version,
        reload_callback=load_production_model,
        verify_callback=verify_loaded_version,
    )


@app.on_event("startup")
def startup_event():
    global checkpoint_pool, investigator_graph
    load_production_model()

    database_url = os.getenv("MLFLOW_TRACKING_URI")
    if not database_url or not database_url.startswith(("postgresql://", "postgres://")):
        raise RuntimeError(
            "MLFLOW_TRACKING_URI must point to the Neon PostgreSQL database for persistent Investigator checkpoints."
        )

    checkpoint_pool = ConnectionPool(
        conninfo=database_url,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        min_size=1,
        max_size=5,
        open=False,
    )
    checkpoint_pool.open()
    checkpoint_pool.wait()
    checkpointer = PostgresSaver(checkpoint_pool)
    checkpointer.setup()

    investigator_graph = build_investigator_graph(
        checkpointer=checkpointer,
        model_provider=lambda: model,
        data_context_provider=production_data_context,
        deployment_provider=deploy_candidate_model,
    )
    print("AI Investigator initialized with PostgreSQL checkpointing.")


@app.on_event("shutdown")
def shutdown_event():
    global checkpoint_pool
    if checkpoint_pool is not None:
        checkpoint_pool.close()
        print("AI Investigator PostgreSQL connection pool closed.")


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


@app.post("/admin/monitoring/run")
def run_monitoring_endpoint():

    # Capture the currently loaded production model and schema.
    with model_lock:
        if model is None or model_schema is None:
            raise HTTPException(
                status_code=503,
                detail="Production model or feature schema is not loaded.",
            )

        active_model = model
        active_version = model_version
        active_schema = model_schema.copy()

    # Validate B2 configuration.
    required_env = [
        MONITORING_BUCKET_ENV,
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]

    missing_env = [
        key for key in required_env
        if not os.getenv(key)
    ]

    if missing_env:
        raise HTTPException(
            status_code=503,
            detail="B2 storage configuration is incomplete.",
        )

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name=os.environ["AWS_DEFAULT_REGION"],
        )

        bucket = os.environ[MONITORING_BUCKET_ENV]

        # Download and process the dataset temporarily.
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(
                temp_dir,
                "monitoring_dataset.csv",
            )

            s3.download_file(
                bucket,
                MONITORING_OBJECT_KEY,
                data_path,
            )

            report = run_monitoring(
                data_path=data_path,
                model=active_model,
                model_schema=active_schema,
            )

        report["model_version"] = active_version
        report["monitoring_source"] = {
            "storage": "Backblaze B2",
            "object_key": MONITORING_OBJECT_KEY,
        }

        return report

    except Exception as error:
        print(
            f"[MONITORING ERROR] "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Monitoring execution failed.",
        )




@app.post("/admin/investigator/start")
def start_investigation():
    global active_investigation_thread
    if investigator_graph is None:
        raise HTTPException(status_code=503, detail="AI Investigator is not initialized.")

    with model_lock:
        if model is None or model_schema is None:
            raise HTTPException(status_code=503, detail="Production model/schema is unavailable.")
        active_model, active_version, active_schema = model, model_version, model_schema.copy()

    with investigation_lock:
        if active_investigation_thread == "starting":
            return {"status": "already_starting"}
        if active_investigation_thread:
            existing_config = {"configurable": {"thread_id": active_investigation_thread}}
            existing = investigator_graph.get_state(existing_config)
            if existing and existing.next:
                return {
                    "status": "already_pending",
                    "thread_id": active_investigation_thread,
                    "pending_nodes": list(existing.next),
                }
        active_investigation_thread = "starting"

    try:
        with production_data_context() as data_path:
            report = run_monitoring(
                data_path=data_path,
                model=active_model,
                model_schema=active_schema,
            )
            reference_data, current_data, features = prepare_investigator_data(
                data_path, active_schema
            )

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "monitoring_report": report,
            "reference_data": reference_data,
            "current_data": current_data,
            "features": features,
            "selected_tools": [],
            "diagnostic_results": {},
            "root_cause": "",
            "recommended_action": "",
            "remediation_action": {},
            "human_approval": {},
            "remediation_status": "pending",
            "retraining_result": {},
            "test_feature_data": None,
            "deployment_approval": {},
            "deployment_status": "not_started",
            "deployment_result": {},
        }
        investigator_graph.invoke(initial_state, config=config)
        snapshot = investigator_graph.get_state(config)
        pending = set(snapshot.next or [])
        with investigation_lock:
            active_investigation_thread = thread_id if pending else None

        if "human_approval" in pending:
            status = "awaiting_remediation_approval"
        elif "deployment_approval" in pending:
            status = "awaiting_deployment_approval"
        else:
            status = "completed"

        values = snapshot.values or {}
        return {
            "thread_id": thread_id,
            "model_version": active_version,
            "status": status,
            "pending_nodes": list(pending),
            "monitoring_report": report,
            "root_cause": values.get("root_cause"),
            "recommended_action": values.get("recommended_action"),
            "remediation_action": values.get("remediation_action"),
            "remediation_status": values.get("remediation_status"),
        }
    except Exception as error:
        with investigation_lock:
            if active_investigation_thread == "starting":
                active_investigation_thread = None
        print(f"[INVESTIGATOR ERROR] {type(error).__name__}: {error}")
        raise HTTPException(status_code=500, detail="Investigator execution failed.")


@app.get("/admin/investigator/{thread_id}")
def get_investigation(thread_id: str):
    if investigator_graph is None:
        raise HTTPException(status_code=503, detail="AI Investigator is not initialized.")
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = investigator_graph.get_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    values = snapshot.values
    pending = set(snapshot.next or [])
    if "human_approval" in pending:
        status = "awaiting_remediation_approval"
    elif "deployment_approval" in pending:
        status = "awaiting_deployment_approval"
    else:
        status = "completed"
    retraining = values.get("retraining_result") or {}
    return {
        "thread_id": thread_id,
        "status": status,
        "pending_nodes": list(pending),
        "monitoring_report": values.get("monitoring_report"),
        "root_cause": values.get("root_cause"),
        "recommended_action": values.get("recommended_action"),
        "remediation_action": values.get("remediation_action"),
        "remediation_status": values.get("remediation_status"),
        "retraining_result": retraining,
        "deployment_status": values.get("deployment_status"),
        "deployment_result": values.get("deployment_result"),
        "pending_approval": (
            {
                "stage": "deployment",
                "candidate_version": retraining.get("model_version"),
                "candidate_metrics": retraining.get("candidate_metrics"),
                "production_metrics": retraining.get("production_metrics"),
            }
            if "deployment_approval" in pending else None
        ),
    }


@app.post("/admin/investigator/{thread_id}/approve")
def approve_investigation(thread_id: str, request: dict[str, Any]):
    global active_investigation_thread
    if investigator_graph is None:
        raise HTTPException(status_code=503, detail="AI Investigator is not initialized.")
    approved = request.get("approved")
    if not isinstance(approved, bool):
        raise HTTPException(status_code=400, detail="'approved' must be true or false.")

    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = investigator_graph.get_state(config)
        pending = set(snapshot.next or []) if snapshot else set()
        if not pending:
            raise HTTPException(status_code=404, detail="No pending approval for this investigation.")
        if "human_approval" in pending:
            stage = "remediation"
        elif "deployment_approval" in pending:
            stage = "deployment"
        else:
            raise HTTPException(status_code=409, detail="Investigation is not waiting for approval.")

        investigator_graph.invoke(
            Command(resume={"approved": approved, "feedback": request.get("feedback", "")}),
            config=config,
        )
        updated = investigator_graph.get_state(config)
        values = updated.values or {}
        next_nodes = set(updated.next or [])
        if "deployment_approval" in next_nodes:
            status = "awaiting_deployment_approval"
        elif "human_approval" in next_nodes:
            status = "awaiting_remediation_approval"
        else:
            status = "completed"
            with investigation_lock:
                if active_investigation_thread == thread_id:
                    active_investigation_thread = None

        retraining = values.get("retraining_result") or {}
        return {
            "thread_id": thread_id,
            "approval_stage": stage,
            "approved": approved,
            "status": status,
            "remediation_status": values.get("remediation_status"),
            "retraining_result": retraining,
            "deployment_status": values.get("deployment_status"),
            "deployment_result": values.get("deployment_result"),
            "pending_approval": (
                {
                    "stage": "deployment",
                    "candidate_version": retraining.get("model_version"),
                    "candidate_metrics": retraining.get("candidate_metrics"),
                    "production_metrics": retraining.get("production_metrics"),
                }
                if status == "awaiting_deployment_approval" else None
            ),
        }
    except HTTPException:
        raise
    except Exception as error:
        print(f"[INVESTIGATOR APPROVAL ERROR] {type(error).__name__}: {error}")
        raise HTTPException(status_code=500, detail="Investigator approval failed.")


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
