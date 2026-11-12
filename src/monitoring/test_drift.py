import pandas as pd

from src.monitoring.data_drift import detect_drift


def test_detect_drift_identifies_distribution_shift():
    reference_data = pd.DataFrame({
        "lag_1": [100, 110, 120, 130, 140, 150, 160, 170],
        "rolling_mean_7": [105, 115, 125, 135, 145, 155, 165, 175],
    })

    current_data = pd.DataFrame({
        "lag_1": [500, 510, 520, 530, 540, 550, 560, 570],
        "rolling_mean_7": [505, 515, 525, 535, 545, 555, 565, 575],
    })

    features = ["lag_1", "rolling_mean_7"]

    results = detect_drift(
        reference_data,
        current_data,
        features,
    )

    assert set(results.keys()) == set(features)

    for feature in features:
        assert results[feature]["drift_detected"] is True
        assert results[feature]["p_value"] < 0.05