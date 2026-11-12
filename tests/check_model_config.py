
import time
import statistics

import mlflow
import mlflow.sklearn
import pandas as pd
from threadpoolctl import threadpool_limits


MODEL_NAME = "MLPilot-Demand-Forecasting"

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

model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}@production"
)

input_data = pd.DataFrame([INPUT])

print("Testing Random Forest thread configurations")

for jobs in [1, 2, 4, 8, -1]:
    model.set_params(n_jobs=jobs)

    with threadpool_limits(limits=1):
        for _ in range(10):
            model.predict(input_data)

        timings = []

        for _ in range(100):
            start = time.perf_counter()
            model.predict(input_data)
            timings.append(
                (time.perf_counter() - start) * 1000
            )

    print(
        f"n_jobs={jobs:2} | "
        f"Average={statistics.mean(timings):.3f} ms | "
        f"P50={statistics.median(timings):.3f} ms | "
        f"P95={sorted(timings)[94]:.3f} ms"
    )
