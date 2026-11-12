
import mlflow
from mlflow.tracking import MlflowClient
from src.tracking.mlflow_config import configure_mlflow

MODEL_NAME = "MLPilot-Demand-Forecasting"
PRODUCTION_ALIAS = "production"


def rollback_model(target_version):
    
    configure_mlflow()
    client = MlflowClient()

    print("\nStarting Model Rollback...")

    # Get current production model
    try:
        current = client.get_model_version_by_alias(
            MODEL_NAME, PRODUCTION_ALIAS
        )
        print(f"Current Production Version: {current.version}")
    except Exception:
        raise RuntimeError("No production model currently assigned.")

    # Validate target version
    target = client.get_model_version(
        MODEL_NAME, str(target_version)
    )

    if target.status != "READY":
        raise RuntimeError("Target model is not ready.")

    if target.version == current.version:
        print("Target version is already in production.")
        return

    print(f"Rolling back to version {target.version}...")

    # Update production alias
    client.set_registered_model_alias(
        MODEL_NAME,
        PRODUCTION_ALIAS,
        str(target.version)
    )

    # Verify rollback
    updated = client.get_model_version_by_alias(
        MODEL_NAME, PRODUCTION_ALIAS
    )

    print("\nRollback successful!")
    print(f"Previous version: {current.version}")
    print(f"Production version: {updated.version}")


if __name__ == "__main__":
    version = input("Enter the version to roll back to: ").strip()

    if not version.isdigit():
        raise ValueError("Enter a valid model version number.")

    rollback_model(version)

