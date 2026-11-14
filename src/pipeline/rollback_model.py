import argparse

from mlflow.tracking import MlflowClient
from src.tracking.mlflow_config import configure_mlflow
from src.pipeline.deploy_model import deploy_model

MODEL_NAME = "MLPilot-Demand-Forecasting"
PRODUCTION_ALIAS = "production"


def rollback_model(target_version, approved=False):
    if approved is not True:
        raise PermissionError("Rollback requires explicit human approval.")
    configure_mlflow()
    client = MlflowClient()
    current = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    target = client.get_model_version(MODEL_NAME, str(target_version))
    if target.status != "READY":
        raise RuntimeError("Target model is not READY.")
    if str(target.version) == str(current.version):
        return {"status": "unchanged", "production_version": str(current.version)}
    return deploy_model(
        str(target.version),
        approved=True,
        expected_previous_version=str(current.version),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not args.version.isdigit():
        raise ValueError("Enter a valid model version number.")
    approval = input(f"Approve rollback to version {args.version}? (yes/no): ").strip().lower()
    if approval == "yes":
        print(rollback_model(args.version, approved=True))
    else:
        print("Rollback cancelled.")
