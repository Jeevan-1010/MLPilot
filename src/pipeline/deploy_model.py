import argparse
import os
import time

import requests
import mlflow  # Backward-compatible module attribute used by deployment tests/integrations.
from mlflow.tracking import MlflowClient

from src.tracking.mlflow_config import configure_mlflow
from src.pipeline.promote_model import promote_model

MODEL_NAME = "MLPilot-Demand-Forecasting"


def _api_url():
    return os.getenv("MLPILOT_API_URL", f"http://127.0.0.1:{os.getenv('PORT', '8000')}").rstrip("/")


def reload_api_model():
    token = os.getenv("MLPILOT_ADMIN_TOKEN")
    if not token:
        raise RuntimeError("MLPILOT_ADMIN_TOKEN is required to reload the API model.")
    response = requests.post(
        f"{_api_url()}/admin/reload-model",
        headers={"X-Admin-Token": token},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def verify_deployment(expected_version):
    for _ in range(5):
        try:
            response = requests.get(f"{_api_url()}/health", timeout=10)
            response.raise_for_status()
            health = response.json()
            if (
                health.get("status") == "healthy"
                and health.get("schema_loaded") is True
                and str(health.get("model_version")) == str(expected_version)
            ):
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def deploy_model(
    target_version,
    approved=False,
    expected_previous_version=None,
    reload_callback=None,
    verify_callback=None,
):
    if approved is not True:
        raise PermissionError("Deployment requires explicit human approval.")
    target_version = str(target_version)
    if not target_version.isdigit():
        raise ValueError("Target model version must be numeric.")

    configure_mlflow()
    client = MlflowClient()
    current = client.get_model_version_by_alias(MODEL_NAME, "production")
    previous_version = str(current.version)

    if expected_previous_version is not None and previous_version != str(expected_previous_version):
        raise RuntimeError("Production model changed after evaluation; re-run candidate evaluation.")
    if previous_version == target_version:
        raise RuntimeError("Candidate is already in production.")

    def reload_and_verify(version):
        if reload_callback is None:
            reload_api_model()
        else:
            reload_callback()
        return verify_deployment(version) if verify_callback is None else bool(verify_callback(version))

    try:
        promotion = promote_model(target_version, approved=True)
        if not reload_and_verify(target_version):
            raise RuntimeError("Deployment health validation failed.")
        return {
            "status": "success",
            "previous_version": previous_version,
            "deployed_version": target_version,
            "promotion": promotion,
        }
    except Exception as deployment_error:
        try:
            client.set_registered_model_alias(MODEL_NAME, "production", previous_version)
            restored = client.get_model_version_by_alias(MODEL_NAME, "production")
            if str(restored.version) != previous_version:
                raise RuntimeError("Rollback alias verification failed.")
            if not reload_and_verify(previous_version):
                raise RuntimeError("Previous model API health verification failed.")
            return {
                "status": "rolled_back",
                "failed_version": target_version,
                "restored_version": previous_version,
                "error": str(deployment_error),
            }
        except Exception as rollback_error:
            raise RuntimeError(
                f"Deployment failed ({deployment_error}); rollback verification failed ({rollback_error})."
            ) from rollback_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not args.version.isdigit():
        raise ValueError("Enter a valid model version number.")
    approval = input(f"Approve deployment of version {args.version}? (yes/no): ").strip().lower()
    if approval == "yes":
        print(deploy_model(args.version, approved=True))
    else:
        print("Deployment cancelled.")
