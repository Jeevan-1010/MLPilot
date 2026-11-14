
from pathlib import Path
import joblib

from src.data_preparation import prepare_daily_demand
from src.pipeline.feature_engineering import create_features

from src.monitoring.data_drift import detect_drift
from src.monitoring.performance_monitor import detect_performance_degradation
from src.monitoring.data_quality import check_data_quality


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "prepwise_cleaned_dataset (3).csv"

MODEL_PATH = BASE_DIR / "models" / "final_demand_forecasting_rf.joblib"

FEATURES_PATH = (
    BASE_DIR / "artifacts" / "demand_forecasting_features.joblib"
)

DEFAULT_ROLLING_WINDOWS = {
    "rolling_mean_7": 7,
    "rolling_mean_14": 14,
}

DRIFT_FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
]


def run_monitoring(
    data_path=DATA_PATH,
    model=None,
    model_schema=None,
):
    print("\nStarting MLPilot Monitoring...\n")

    # Load historical data
    daily_demand = prepare_daily_demand(data_path)

    # Resolve model and feature configuration
    if model_schema is not None:
        features = model_schema["feature_names"]
        rolling_windows = model_schema.get(
            "rolling_windows",
            DEFAULT_ROLLING_WINDOWS,
        )
    else:
        features = joblib.load(FEATURES_PATH)
        rolling_windows = DEFAULT_ROLLING_WINDOWS

    # Preserve standalone execution using the local model
    if model is None:
        model = joblib.load(MODEL_PATH)

    # Generate features using the model's schema
    feature_df = create_features(
        daily_demand,
        rolling_windows=rolling_windows,
    )

    # Validate required feature availability
    missing_features = [
        feature for feature in features
        if feature not in feature_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features in monitoring data: {missing_features}"
        )

    # Split into reference and current windows
    if len(feature_df) < 60:
        raise ValueError(
            "At least 60 feature records are required for monitoring."
        )

    reference_data = feature_df.iloc[-60:-30]
    current_data = feature_df.tail(30)

    # 1. DATA QUALITY MONITORING
    quality_columns = ["daily_demand"] + features

    quality_results = check_data_quality(
        data=current_data,
        required_columns=quality_columns,
        numeric_columns=quality_columns,
        non_negative_columns=["daily_demand"],
    )

    # 2. DATA DRIFT MONITORING
    drift_features = [
        feature for feature in DRIFT_FEATURES
        if feature in features
    ]

    drift_results = detect_drift(
        reference_data,
        current_data,
        drift_features,
    ) if drift_features else {}

    drift_detected = any(
        result.get("drift_detected", False)
        for result in drift_results.values()
    )

    # 3. MODEL PERFORMANCE MONITORING
    performance_results = detect_performance_degradation(
        model,
        reference_data,
        current_data,
        features,
    )

    performance_degraded = performance_results[
        "performance_degraded"
    ]

    # 4. OVERALL MODEL STATUS
    data_quality_issues = (
        quality_results["quality_status"] != "HEALTHY"
    )

    if data_quality_issues or performance_degraded:
        status = "DEGRADED"
    elif drift_detected:
        status = "DRIFT DETECTED - PERFORMANCE NOT DEGRADED"
    else:
        status = "HEALTHY"

    # 5. STRUCTURED MONITORING REPORT
    monitoring_report = {
        "model_name": "MLPilot-Demand-Forecasting",
        "monitoring_status": status,
        "data_quality": quality_results,
        "data_drift": {
            "drift_detected": drift_detected,
            "feature_results": drift_results,
        },
        "model_performance": performance_results,
        "alerts": {
            "data_quality_issues": data_quality_issues,
            "data_drift_detected": drift_detected,
            "performance_degraded": performance_degraded,
        },
    }

    # Print report
    print("\n========== MLPILOT MONITORING REPORT ==========")

    for section, result in monitoring_report.items():
        print(f"\n{section.upper()}:")
        print(result)

    print("\n===============================================")

    return monitoring_report


if __name__ == "__main__":
    run_monitoring()


def prepare_investigator_data(data_path, model_schema=None):
    """Return serializable reference/current windows for the Investigator."""
    daily_demand = prepare_daily_demand(data_path)
    rolling_windows = (
        model_schema.get("rolling_windows", DEFAULT_ROLLING_WINDOWS)
        if model_schema else DEFAULT_ROLLING_WINDOWS
    )
    default_features = [
        "day_of_week", "month", "day_of_month", "week_of_year",
        "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14",
    ]
    feature_names = (
        model_schema.get("feature_names", default_features)
        if model_schema else default_features
    )
    feature_df = create_features(
        daily_demand,
        rolling_windows=rolling_windows,
    )
    missing = [name for name in feature_names if name not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing Investigator features: {missing}")
    if len(feature_df) < 60:
        raise ValueError("At least 60 feature records are required for investigation.")
    reference = feature_df.iloc[-60:-30]
    current = feature_df.tail(30)
    return (
        reference.to_dict(orient="records"),
        current.to_dict(orient="records"),
        list(feature_names),
    )
