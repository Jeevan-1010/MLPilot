
import numpy as np
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from src.tracking.mlflow_config import configure_mlflow
from src.data_preparation import prepare_data
from src.pipeline.feature_engineering import (
    create_features,
    DEFAULT_ROLLING_WINDOWS,
)
from src.tracking.mlflow_tracker import log_run, register_model


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "prepwise_cleaned_dataset (3).csv"

MODEL_NAME = "MLPilot-Demand-Forecasting"
PRODUCTION_ALIAS = "production"

FEATURES = [
    "day_of_week", "month", "day_of_month", "week_of_year",
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14",
]

TARGET = "daily_demand"
IMPROVEMENT_TOLERANCE = 0.01

PARAM_DISTRIBUTIONS = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [3, 5, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": [1.0, "sqrt", 0.8],
}


def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def evaluate_config(X, y, params, splitter):
    results = []

    for train_idx, val_idx in splitter.split(X):
        model = RandomForestRegressor(**params)

        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = model.predict(X.iloc[val_idx])

        results.append(
            calculate_metrics(y.iloc[val_idx], predictions)
        )

    return {
        metric: float(np.mean([fold[metric] for fold in results]))
        for metric in ["MAE", "RMSE"]
    }


def get_production_model():
    
    configure_mlflow()
    print("\nLoading current production model...")

    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}"
    )

    print("Production model loaded successfully.")

    return model


def get_production_feature_columns(model):
    """
    Recover the feature schema expected by the production model.
    """

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    expected_count = getattr(model, "n_features_in_", None)

    if expected_count == len(FEATURES):
        return FEATURES.copy()

    raise ValueError(
        "Cannot safely determine the production model's feature schema. "
        "The model does not expose feature_names_in_ and its feature "
        "count does not match the known baseline schema."
    )



def run_pipeline(
    data_path=None,
    feature_data=None,
    candidate_params=None,
    feature_columns=None,
    rolling_windows=None,
    register_candidate=True,
):
    print("\n========== MLPILOT RETRAINING PIPELINE ==========")

    selected_data_path = Path(data_path) if data_path else DATA_PATH

    selected_features = (
        feature_columns.copy()
        if feature_columns is not None
        else FEATURES.copy()
    )

    selected_windows = (
        rolling_windows.copy()
        if rolling_windows is not None
        else DEFAULT_ROLLING_WINDOWS.copy()
    )

    if not selected_features:
        raise ValueError("At least one feature must be selected.")

    if TARGET in selected_features or "date" in selected_features:
        raise ValueError("Invalid feature selection.")

    if len(selected_features) != len(set(selected_features)):
        raise ValueError("Duplicate features are not allowed.")

    if feature_data is not None:
        print("\nUsing features supplied by remediation executor.")
        feature_df = feature_data.copy()
    else:
        print("\nGenerating features from source data.")

        if not selected_data_path.exists():
            raise FileNotFoundError(
                f"Training data not found: {selected_data_path}"
            )

        _, daily_demand = prepare_data(selected_data_path)

        feature_df = create_features(
            daily_demand,
            rolling_windows=selected_windows,
        )

    required_columns = ["date", TARGET] + selected_features

    missing_columns = [
        column for column in required_columns
        if column not in feature_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Training data is missing columns: {missing_columns}"
        )

    feature_df = (
        feature_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    if feature_df[selected_features + [TARGET]].isna().any().any():
        raise ValueError(
            "Training data contains missing feature or target values."
        )

    y = feature_df[TARGET]
    X_candidate = feature_df[selected_features]

    splitter = TimeSeriesSplit(n_splits=5)

    # 1. Load production model and recover its feature schema.
    production_model = get_production_model()

    production_features = get_production_feature_columns(
        production_model
    )

    missing_production_features = [
        column for column in production_features
        if column not in feature_df.columns
    ]

    if missing_production_features:
        raise ValueError(
            "Cannot evaluate production model. Missing its required "
            f"features: {missing_production_features}"
        )

    X_production = feature_df[production_features]

    if X_production.isna().any().any():
        raise ValueError(
            "Production model input contains missing feature values."
        )

    # 2. Evaluate production configuration.
    production_params = production_model.get_params()
    production_params["random_state"] = 42
    production_params["n_jobs"] = -1

    production_metrics = evaluate_config(
        X_production,
        y,
        production_params,
        splitter,
    )

    print("\nProduction Features:")
    print(production_features)

    print("\nCandidate Features:")
    print(selected_features)

    print("\nProduction CV Metrics:")
    print(production_metrics)

    # 3. Select candidate configuration.
    if candidate_params is not None:
        print("\nUsing AI-recommended hyperparameters.")

        allowed_params = set(
            RandomForestRegressor().get_params().keys()
        )

        invalid_params = (
            set(candidate_params.keys()) - allowed_params
        )

        if invalid_params:
            raise ValueError(
                f"Unsupported hyperparameters: {invalid_params}"
            )

        selected_params = candidate_params.copy()
        selected_params.update({
            "random_state": 42,
            "n_jobs": -1,
        })

    else:
        print("\nSearching for candidate hyperparameters...")

        search = RandomizedSearchCV(
            estimator=RandomForestRegressor(
                random_state=42,
                n_jobs=-1,
            ),
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=20,
            scoring="neg_mean_absolute_error",
            cv=TimeSeriesSplit(n_splits=5),
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        search.fit(X_candidate, y)

        selected_params = search.best_params_.copy()
        selected_params.update({
            "random_state": 42,
            "n_jobs": -1,
        })

    print("\nSelected Candidate Parameters:")
    print(selected_params)

    # 4. Evaluate candidate.
    candidate_metrics = evaluate_config(
        X_candidate,
        y,
        selected_params,
        splitter,
    )

    print("\nCandidate CV Metrics:")
    print(candidate_metrics)

    mae_improved = (
        candidate_metrics["MAE"]
        < production_metrics["MAE"] * (1 - IMPROVEMENT_TOLERANCE)
    )

    rmse_not_worse = (
        candidate_metrics["RMSE"]
        <= production_metrics["RMSE"] * (1 + IMPROVEMENT_TOLERANCE)
    )

    validation_passed = mae_improved and rmse_not_worse

    print("\n========== VALIDATION ==========")
    print("MAE improved:", mae_improved)
    print("RMSE within tolerance:", rmse_not_worse)
    print("Validation passed:", validation_passed)

    if not validation_passed:
        print("\nCandidate rejected. Production model unchanged.")

        return {
            "status": "failed",
            "reason": "Candidate did not meet production validation criteria.",
            "production_metrics": production_metrics,
            "candidate_metrics": candidate_metrics,
            "production_features": production_features,
            "candidate_features": selected_features,
        }

    # 5. Evaluation-only mode: do not train, log, or register.
    if not register_candidate:
        print("\nEvaluation-only mode. No model trained or registered.")

        return {
            "status": "evaluation_passed",
            "registered": False,
            "production_metrics": production_metrics,
            "candidate_metrics": candidate_metrics,
            "candidate_params": selected_params,
            "production_features": production_features,
            "features": selected_features,
            "rolling_windows": selected_windows,
        }

    # 6. Train final candidate.
    print("\nCandidate passed validation. Training candidate.")

    final_model = RandomForestRegressor(**selected_params)
    final_model.fit(X_candidate, y)

    # 7. Log candidate.
    feature_schema = {
    "schema_version": 1,
    "feature_names": selected_features,
    "target": TARGET,
    "rolling_windows": selected_windows,
}

    run_id = log_run(
        final_model,
        candidate_metrics,
        selected_params,
        feature_schema=feature_schema,
    )

    # 8. Register candidate without promoting it.
    model_uri = f"runs:/{run_id}/demand_forecasting_model"
    registered_model = register_model(model_uri)

    print("\nCandidate registered successfully.")
    print("Run ID:", run_id)
    print("Model Version:", registered_model.version)

    return {
        "status": "passed",
        "registered": True,
        "run_id": run_id,
        "model_version": str(registered_model.version),
        "production_metrics": production_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_params": selected_params,
        "production_features": production_features,
        "features": selected_features,
        "rolling_windows": selected_windows,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("\nPipeline Result:")
    print(result)
