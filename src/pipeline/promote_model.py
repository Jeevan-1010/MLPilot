
import mlflow
from mlflow.tracking import MlflowClient
from src.tracking.mlflow_config import configure_mlflow

MODEL_NAME = "MLPilot-Demand-Forecasting"


def promote_model(target_version, approved=False):
    if not approved:
        raise PermissionError(
            "Deployment requires explicit human approval."
        )

    
    configure_mlflow()
    client = MlflowClient()

    target_version = str(target_version)

    print("\nChecking deployment candidate...")

    target = client.get_model_version(
        MODEL_NAME, target_version
    )

    if target.status != "READY":
        raise RuntimeError("Candidate model is not READY.")

    run = client.get_run(target.run_id)

    if run.info.status != "FINISHED":
        raise RuntimeError("Candidate run is not FINISHED.")

    metrics = run.data.metrics
    mae = metrics.get("MAE")
    rmse = metrics.get("RMSE")

    if mae is None or rmse is None:
        raise ValueError("Candidate metrics are missing.")

    if mae <= 0 or rmse <= 0:
        raise ValueError("Candidate metrics are invalid.")

    current = client.get_model_version_by_alias(
        MODEL_NAME, "production"
    )

    previous_version = str(current.version)

    if previous_version == target_version:
        raise RuntimeError(
            "Candidate is already in production."
        )

    print(f"Current production: v{previous_version}")
    print(f"Deployment candidate: v{target_version}")
    print(f"Candidate MAE: {mae}")
    print(f"Candidate RMSE: {rmse}")

    client.set_registered_model_alias(
        MODEL_NAME,
        "production",
        target_version
    )

    updated = client.get_model_version_by_alias(
        MODEL_NAME, "production"
    )

    if str(updated.version) != target_version:
        raise RuntimeError(
            "Production alias verification failed."
        )

    print(f"\nVersion {target_version} promoted.")
    print(f"Previous production version: {previous_version}")

    return {
        "status": "promoted",
        "previous_version": previous_version,
        "deployed_version": target_version
    }


if __name__ == "__main__":
    version = input("Candidate version to deploy: ").strip()

    if not version.isdigit():
        raise ValueError("Enter a valid model version.")

    approval = input(
        f"Approve deployment of version {version}? (yes/no): "
    ).strip().lower()

    if approval != "yes":
        print("Deployment cancelled.")
    else:
        promote_model(version, approved=True)
