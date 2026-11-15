"""Deterministic, offline evaluation of MLPilot monitoring detectors.

These tests use labelled synthetic fixtures; they do not call Render, B2,
MLflow, Groq, or mutate any production model/deployment.
"""
import numpy as np
import pandas as pd

from src.monitoring.data_drift import detect_drift
from src.monitoring.data_quality import check_data_quality


def _monitoring_case(reference, current):
    result = detect_drift(reference, current, ["signal"])
    return bool(result["signal"]["drift_detected"])


def test_monitoring_drift_detection_metrics():
    rng = np.random.default_rng(20261115)
    cases = []

    # Labelled stable distributions (negative cases).
    for _ in range(10):
        reference = pd.DataFrame({"signal": rng.normal(0, 1, 200)})
        current = pd.DataFrame({"signal": rng.normal(0, 1, 200)})
        cases.append((False, _monitoring_case(reference, current)))

    # Labelled shifted distributions (positive cases).
    for _ in range(10):
        reference = pd.DataFrame({"signal": rng.normal(0, 1, 200)})
        current = pd.DataFrame({"signal": rng.normal(1.5, 1, 200)})
        cases.append((True, _monitoring_case(reference, current)))

    tp = sum(label and prediction for label, prediction in cases)
    tn = sum(not label and not prediction for label, prediction in cases)
    fp = sum(not label and prediction for label, prediction in cases)
    fn = sum(label and not prediction for label, prediction in cases)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(cases)

    print(
        "\nMONITORING DRIFT EVALUATION "
        f"cases={len(cases)} accuracy={accuracy:.3f} "
        f"precision={precision:.3f} recall={recall:.3f} "
        f"false_positive_rate={false_positive_rate:.3f}"
    )

    assert len(cases) == 20
    assert recall >= 0.90
    assert false_positive_rate <= 0.20


def test_data_quality_detection_metrics():
    required = ["daily_demand", "signal"]
    clean = pd.DataFrame({"daily_demand": [10, 11, 12], "signal": [1.0, 2.0, 3.0]})
    cases = [
        (False, clean.copy()),
        (True, pd.DataFrame({"daily_demand": [10, None, 12], "signal": [1.0, 2.0, 3.0]})),
        (True, pd.DataFrame({"daily_demand": [10, -1, 12], "signal": [1.0, 2.0, 3.0]})),
        (True, pd.DataFrame({"daily_demand": [10, 11, 12], "signal": [1.0, np.inf, 3.0]})),
        (True, pd.concat([clean, clean.iloc[[0]]], ignore_index=True)),
    ]

    predictions = []
    for expected_issue, frame in cases:
        result = check_data_quality(
            frame,
            required_columns=required,
            numeric_columns=required,
            non_negative_columns=["daily_demand"],
        )
        predictions.append(result["quality_status"] != "HEALTHY")
        assert predictions[-1] is expected_issue

    tp = sum(expected and predicted for (expected, _), predicted in zip(cases, predictions))
    tn = sum(not expected and not predicted for (expected, _), predicted in zip(cases, predictions))
    fp = sum(not expected and predicted for (expected, _), predicted in zip(cases, predictions))
    fn = sum(expected and not predicted for (expected, _), predicted in zip(cases, predictions))
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(cases)

    print(
        "\nDATA QUALITY EVALUATION "
        f"cases={len(cases)} accuracy={accuracy:.3f} "
        f"recall={recall:.3f} false_positive_rate={false_positive_rate:.3f}"
    )
    assert accuracy == 1.0
    assert recall == 1.0
    assert false_positive_rate == 0.0
