
from pathlib import Path
import joblib

from src.data_preparation import prepare_data
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


def run_monitoring():

    print("\nStarting MLPilot Monitoring...\n")

    # Load historical data
    _, daily_demand = prepare_data(DATA_PATH)

    # Generate features
    feature_df = create_features(daily_demand)

    # Load model and feature names
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    # Split into reference and current windows
    reference_data = feature_df.iloc[-60:-30]
    current_data = feature_df.tail(30)

    # -----------------------------
    # 1. DATA QUALITY MONITORING
    # -----------------------------

    quality_results = check_data_quality(
        data=current_data,
        required_columns=["daily_demand"] + features,
        numeric_columns=["daily_demand"] + features,
        non_negative_columns=["daily_demand"]
    )

    # -----------------------------
    # 2. DATA DRIFT MONITORING
    # -----------------------------

    drift_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14"
    ]

    drift_results = detect_drift(
        reference_data,
        current_data,
        drift_features
    )

    drift_detected = any(
        result.get("drift_detected", False)
        for result in drift_results.values()
    )

    # -----------------------------
    # 3. MODEL PERFORMANCE MONITORING
    # -----------------------------

    performance_results = detect_performance_degradation(
        model,
        reference_data,
        current_data,
        features
    )

    performance_degraded = performance_results[
        "performance_degraded"
    ]

    # -----------------------------
    # 4. OVERALL MODEL STATUS
    # -----------------------------

    data_quality_issues = (
        quality_results["quality_status"] != "HEALTHY"
    )

    if data_quality_issues or performance_degraded:
        status = "DEGRADED"
    elif drift_detected:
        status = "DRIFT DETECTED - PERFORMANCE NOT DEGRADED"
    else:
        status = "HEALTHY"

    # -----------------------------
    # 5. STRUCTURED MONITORING REPORT
    # -----------------------------

    monitoring_report = {
        "model_name": "MLPilot-Demand-Forecasting",
        "monitoring_status": status,

        "data_quality": quality_results,

        "data_drift": {
            "drift_detected": drift_detected,
            "feature_results": drift_results
        },

        "model_performance": performance_results,

        "alerts": {
            "data_quality_issues": data_quality_issues,
            "data_drift_detected": drift_detected,
            "performance_degraded": performance_degraded
        }
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
