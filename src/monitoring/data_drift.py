
import pandas as pd
from scipy.stats import ks_2samp


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: list,
    significance_level: float = 0.05
) -> dict:

    drift_results = {}

    for feature in features:
        reference_values = reference_data[feature].dropna()
        current_values = current_data[feature].dropna()

        if len(reference_values) == 0 or len(current_values) == 0:
            drift_results[feature] = {
                "drift_detected": False,
                "error": "Insufficient data"
            }
            continue

        statistic, p_value = ks_2samp(
            reference_values,
            current_values
        )

        drift_results[feature] = {
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": bool(
                p_value < significance_level
            )
        }

    return drift_results
