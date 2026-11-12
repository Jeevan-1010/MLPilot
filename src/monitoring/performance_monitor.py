
import pandas as pd
from sklearn.metrics import mean_absolute_error


def detect_performance_degradation(
    model,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: list,
    threshold: float = 0.20
) -> dict:

    # Generate predictions
    reference_predictions = model.predict(
        reference_data[features]
    )

    current_predictions = model.predict(
        current_data[features]
    )

    # Calculate prediction errors
    reference_mae = mean_absolute_error(
        reference_data["daily_demand"],
        reference_predictions
    )

    current_mae = mean_absolute_error(
        current_data["daily_demand"],
        current_predictions
    )

    # Calculate percentage increase in error
    if reference_mae == 0:
        error_increase = 0.0 if current_mae == 0 else float("inf")
    else:
        error_increase = (
            (current_mae - reference_mae) / reference_mae
        ) * 100

    # Determine performance degradation
    degraded = current_mae > reference_mae * (1 + threshold)

    return {
        "reference_mae": round(float(reference_mae), 2),
        "current_mae": round(float(current_mae), 2),
        "error_increase_percentage": round(float(error_increase), 2),
        "performance_degraded": bool(degraded)
    }
