
import numpy as np
import pandas as pd

from src.data_preparation import prepare_data
from src.pipeline.feature_engineering import (
    create_features,
    DEFAULT_ROLLING_WINDOWS,
)


FEATURES = [
    "day_of_week",
    "month",
    "day_of_month",
    "week_of_year",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
]

ROLLING_FEATURES = DEFAULT_ROLLING_WINDOWS.copy()

ALLOWED_HYPERPARAMETERS = {
    "n_estimators": (50, 1000),
    "max_depth": (1, 50),
    "min_samples_split": (2, 50),
    "min_samples_leaf": (1, 20),
    "max_features": [1.0, "sqrt", "log2", 0.8],
}


def _verify_and_rebuild(
    daily_demand,
    feature_data=None,
    rolling_windows=None,
):
    issues = []

    windows = (
        rolling_windows.copy()
        if rolling_windows is not None
        else DEFAULT_ROLLING_WINDOWS.copy()
    )

    daily = daily_demand.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["daily_demand"] = pd.to_numeric(
        daily["daily_demand"], errors="coerce"
    )

    if daily["date"].duplicated().any():
        issues.append("Duplicate dates detected")

    if daily["daily_demand"].isna().any():
        issues.append("Invalid demand values detected")

    if (daily["daily_demand"].dropna() < 0).any():
        issues.append("Negative demand detected")

    daily = (
        daily.dropna(subset=["date", "daily_demand"])
        .groupby("date", as_index=False)["daily_demand"]
        .sum()
        .sort_values("date")
    )

    if daily.empty:
        raise ValueError("No valid daily-demand data available")

    all_dates = pd.date_range(
        daily["date"].min(),
        daily["date"].max(),
        freq="D",
    )

    daily = (
        daily.set_index("date")
        .reindex(all_dates, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    expected_features = create_features(
        daily,
        rolling_windows=windows,
    )

    feature_results = {}

    for feature, window in windows.items():

        expected = expected_features[feature].to_numpy()

        if feature_data is None:
            actual = expected
            missing_column = False

        elif feature not in feature_data.columns:
            actual = np.array([])
            missing_column = True

        else:
            actual = feature_data[feature].to_numpy()
            missing_column = False

        same_length = len(actual) == len(expected)

        matches = (
            np.isclose(
                actual,
                expected,
                atol=1e-6,
                rtol=0,
                equal_nan=True,
            )
            if same_length
            else np.array([], dtype=bool)
        )

        mismatch_count = (
            int((~matches).sum())
            if same_length
            else abs(len(actual) - len(expected))
        )

        mismatch = (
            missing_column
            or not same_length
            or mismatch_count > 0
        )

        feature_results[feature] = {
            "window": window,
            "mismatched_rows": mismatch_count,
            "missing_column": missing_column,
            "issue_detected": bool(mismatch),
        }

        if mismatch:
            issues.append(f"{feature} mismatch detected")

    correction_applied = bool(issues)

    return {
        "status": "corrected" if correction_applied else "healthy",
        "correction_applied": correction_applied,
        "issues": issues,
        "feature_results": feature_results,
        "rolling_windows": windows,
        "corrected_data": expected_features,
    }


def verify_and_correct_rolling_features(
    data_path: str,
    feature_data=None,
    rolling_windows=None,
) -> dict:

    try:
        if feature_data is not None and not isinstance(feature_data, pd.DataFrame):
            feature_data = pd.DataFrame(feature_data)
        _, daily_demand = prepare_data(data_path)

        return _verify_and_rebuild(
            daily_demand=daily_demand,
            feature_data=feature_data,
            rolling_windows=rolling_windows,
        )

    except Exception as error:
        return {
            "status": "failed",
            "reason": str(error),
            "correction_applied": False,
            "issues": [],
            "corrected_data": None,
        }


def execute_remediation_action(
    action_type: str,
    parameters: dict,
) -> dict:
    """
    Validate AI-recommended remediation actions.

    Only predefined, whitelisted actions are supported.
    Arbitrary AI-generated code is never executed.
    """

    if not isinstance(parameters, dict):
        raise ValueError("Action parameters must be a dictionary.")

    # ACTION 1: Hyperparameter modification
    if action_type == "update_hyperparameters":

        candidate_params = {}

        for name, value in parameters.items():

            if name not in ALLOWED_HYPERPARAMETERS:
                raise ValueError(
                    f"Unsupported hyperparameter: {name}"
                )

            allowed = ALLOWED_HYPERPARAMETERS[name]

            if isinstance(allowed, tuple):

                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not allowed[0] <= value <= allowed[1]
                ):
                    raise ValueError(
                        f"Invalid value for {name}: {value}"
                    )

            elif value not in allowed:
                raise ValueError(
                    f"Invalid value for {name}: {value}"
                )

            candidate_params[name] = value

        if not candidate_params:
            raise ValueError(
                "At least one hyperparameter must be supplied."
            )

        return {
            "status": "validated",
            "action_type": action_type,
            "candidate_params": candidate_params,
        }

    # ACTION 2: Feature removal
    if action_type == "remove_feature":

        feature = parameters.get("feature")

        if feature not in FEATURES:
            raise ValueError(
                f"Cannot remove unknown feature: {feature}"
            )

        remaining_features = [
            name for name in FEATURES
            if name != feature
        ]

        if not remaining_features:
            raise ValueError(
                "Cannot remove every model feature."
            )

        return {
            "status": "validated",
            "action_type": action_type,
            "removed_feature": feature,
            "feature_columns": remaining_features,
        }

    # ACTION 3: Rolling-window modification
    if action_type == "change_rolling_window":

        feature = parameters.get("feature")
        window = parameters.get("window")

        if feature not in ROLLING_FEATURES:
            raise ValueError(
                f"Unsupported rolling feature: {feature}"
            )

        if (
            isinstance(window, bool)
            or not isinstance(window, int)
            or not 2 <= window <= 30
        ):
            raise ValueError(
                "Rolling window must be an integer between 2 and 30."
            )

        updated_windows = ROLLING_FEATURES.copy()
        updated_windows[feature] = window

        return {
            "status": "validated",
            "action_type": action_type,
            "feature": feature,
            "window": window,
            "rolling_windows": updated_windows,
        }

    raise ValueError(
        f"Unsupported remediation action: {action_type}"
    )
