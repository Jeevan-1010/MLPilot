
import mlflow
import mlflow.sklearn


MODEL_NAME = "MLPilot-Demand-Forecasting"
EXPERIMENT_NAME = "MLPilot-Demand-Forecasting"


def start_tracking():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_run(
    model,
    metrics: dict,
    params: dict,
    feature_schema: dict = None,
):
    start_tracking()

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if feature_schema is not None:
            mlflow.log_dict(
                feature_schema,
                "feature_schema.json",
            )

        mlflow.sklearn.log_model(
            model,
            name="demand_forecasting_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        print("MLflow run logged successfully!")
        print("Run ID:", run.info.run_id)

        return run.info.run_id


def register_model(model_uri: str):
    start_tracking()

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print("Model registered successfully!")
    print("Model name:", MODEL_NAME)
    print("Model version:", registered_model.version)

    return registered_model
