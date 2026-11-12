
import argparse
import time

import requests
import mlflow
from mlflow.tracking import MlflowClient
from src.tracking.mlflow_config import configure_mlflow

from src.pipeline.promote_model import promote_model

MODEL_NAME = "MLPilot-Demand-Forecasting"
API_URL = "http://127.0.0.1:8000"


def reload_api_model():
    response = requests.post(
        f"{API_URL}/admin/reload-model",
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def verify_deployment(expected_version):
    for attempt in range(5):
        try:
            response = requests.get(
                f"{API_URL}/health",
                timeout=10
            )
            response.raise_for_status()

            health = response.json()

            if (
                health.get("status") == "healthy"
                and str(health.get("model_version"))
                == str(expected_version)
            ):
                return True

        except requests.RequestException:
            pass

        time.sleep(2)

    return False


def deploy_model(target_version, approved=False):
    if not approved:
        raise PermissionError(
            "Deployment requires explicit human approval."
        )

    
    configure_mlflow()
    client = MlflowClient()

    current = client.get_model_version_by_alias(
        MODEL_NAME, "production"
    )
    previous_version = str(current.version)

    print(f"\nPrevious production version: v{previous_version}")
    print(f"Deployment candidate: v{target_version}")

    promotion = promote_model(
        target_version,
        approved=True
    )

    try:
        print("\nReloading API model...")
        reload_api_model()

        print("Checking deployment health...")

        if not verify_deployment(target_version):
            raise RuntimeError(
                "Deployment health validation failed."
            )

        print("\nDeployment successful!")

        return {
            "status": "success",
            "previous_version": previous_version,
            "deployed_version": str(target_version)
        }

    except Exception as error:
        print(f"\nDeployment failed: {error}")
        print("Starting automatic rollback...")

        client.set_registered_model_alias(
            MODEL_NAME,
            "production",
            previous_version
        )

        restored = client.get_model_version_by_alias(
            MODEL_NAME, "production"
        )

        if str(restored.version) != previous_version:
            raise RuntimeError(
                "Rollback alias verification failed."
            )

        print("Restoring previous API model...")

        reload_api_model()

        if not verify_deployment(previous_version):
            raise RuntimeError(
                "Rollback completed in MLflow, "
                "but API health verification failed."
            ) from error

        print(
            f"Rollback successful. Production is back "
            f"on version {previous_version}."
        )

        return {
            "status": "rolled_back",
            "failed_version": str(target_version),
            "restored_version": previous_version,
            "error": str(error)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        required=True,
        help="Exact candidate model version to deploy"
    )
    args = parser.parse_args()

    if not args.version.isdigit():
        raise ValueError("Enter a valid model version number.")

    approval = input(
        f"Approve deployment of version {args.version}? (yes/no): "
    ).strip().lower()

    if approval != "yes":
        print("Deployment cancelled.")
    else:
        result = deploy_model(
            args.version,
            approved=True
        )
        print("\nFinal Result:", result)
