
import pandas as pd
import joblib


from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.monitoring.data_quality import check_data_quality
from src.monitoring.data_drift import detect_drift


def run_data_quality_check(state):

    current_data = pd.DataFrame(state["current_data"])
    features = state["features"]

    results = check_data_quality(
        data=current_data,
        required_columns=["daily_demand"] + features,
        numeric_columns=["daily_demand"] + features,
        non_negative_columns=["daily_demand"]
    )

    return results


def run_data_drift_check(state):

    reference_data = pd.DataFrame(state["reference_data"])
    current_data = pd.DataFrame(state["current_data"])

    drift_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14"
    ]

    results = detect_drift(
        reference_data=reference_data,
        current_data=current_data,
        features=drift_features
    )

    return results


def run_feature_drift_analysis(state):

    reference_data = pd.DataFrame(state["reference_data"])
    current_data = pd.DataFrame(state["current_data"])

    drift_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14"
    ]

    drift_results = detect_drift(
        reference_data=reference_data,
        current_data=current_data,
        features=drift_features
    )

    ranked_features = sorted(
        [
            {
                "feature": feature,
                "ks_statistic": result.get("ks_statistic", 0),
                "p_value": result.get("p_value"),
                "drift_detected": result.get("drift_detected", False)
            }
            for feature, result in drift_results.items()
        ],
        key=lambda x: x["ks_statistic"],
        reverse=True
    )

    return {
        "ranked_features": ranked_features,
        "strongest_drift_feature": (
            ranked_features[0]["feature"]
            if ranked_features else None
        )
    }


def run_prediction_error_analysis(state, model=None):

    current_data = pd.DataFrame(state["current_data"])
    if model is None:
        model = joblib.load("models/final_demand_forecasting_rf.joblib")
    features = state["features"]

    X = current_data[features]
    actual = current_data["daily_demand"]

    predictions = model.predict(X)
    prediction_series = pd.Series(predictions, index=current_data.index)

    errors = (actual - prediction_series).abs()

    results = {
        "mae": round(float(mean_absolute_error(actual, predictions)), 2),
        "rmse": round(float(mean_squared_error(actual, predictions) ** 0.5), 2),
        "mean_absolute_error_per_record": round(float(errors.mean()), 2),
        "worst_prediction_errors": []
    }

    worst_indices = errors.nlargest(5).index

    for index in worst_indices:
        results["worst_prediction_errors"].append({
            "actual": round(float(actual.loc[index]), 2),
            "predicted": round(float(prediction_series.loc[index]), 2),
            "absolute_error": round(float(errors.loc[index]), 2)
        })

    return results
