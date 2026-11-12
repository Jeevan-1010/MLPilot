
import time
import statistics

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd


MODEL_NAME = "MLPilot-Demand-Forecasting"
MODEL_URI = f"models:/{MODEL_NAME}@production"

INPUT = {
    "day_of_week": 2,
    "month": 11,
    "day_of_month": 11,
    "week_of_year": 46,
    "lag_1": 25000,
    "lag_7": 23000,
    "lag_14": 22000,
    "rolling_mean_7": 24000,
    "rolling_mean_14": 23500,
}

FEATURES = list(INPUT.keys())
input_data = pd.DataFrame([INPUT], columns=FEATURES)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

print("Loading production model...")

pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)
sklearn_model = mlflow.sklearn.load_model(MODEL_URI)

def benchmark(model, name):
    # Warm-up
    for _ in range(10):
        model.predict(input_data)

    timings = []

    for _ in range(100):
        start = time.perf_counter()
        model.predict(input_data)
        timings.append((time.perf_counter() - start) * 1000)

    timings.sort()

    print(f"\n{name}")
    print(f"Average: {statistics.mean(timings):.3f} ms")
    print(f"P50:     {statistics.median(timings):.3f} ms")
    print(f"P95:     {timings[94]:.3f} ms")


benchmark(pyfunc_model, "MLflow PyFunc")
benchmark(sklearn_model, "Direct Scikit-learn")
