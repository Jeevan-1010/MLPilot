
import time
import statistics

import mlflow
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

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model(MODEL_URI)

for batch_size in [1, 10, 50, 100, 500]:
    data = pd.DataFrame([INPUT] * batch_size)

    # Warm-up
    model.predict(data)

    timings = []

    for _ in range(30):
        start = time.perf_counter()
        model.predict(data)
        timings.append((time.perf_counter() - start) * 1000)

    avg_ms = statistics.mean(timings)

    print(
        f"Batch: {batch_size:3d} | "
        f"Total: {avg_ms:.3f} ms | "
        f"Per prediction: {avg_ms / batch_size:.3f} ms | "
        f"Throughput: {batch_size / (avg_ms / 1000):.2f} predictions/sec"
    )
