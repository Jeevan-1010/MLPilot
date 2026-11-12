
import time
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://127.0.0.1:8000/predict"

PAYLOAD = {
    "day_of_week": 2,
    "month": 5,
    "day_of_month": 15,
    "week_of_year": 20,
    "lag_1": 12000,
    "lag_7": 11500,
    "lag_14": 11000,
    "rolling_mean_7": 11800,
    "rolling_mean_14": 11600,
}

TOTAL_REQUESTS = 100
CONCURRENT_WORKERS = 10


def send_request():
    start = time.perf_counter()

    response = requests.post(
        API_URL,
        json=PAYLOAD,
        timeout=30,
    )

    latency = (time.perf_counter() - start) * 1000
    response.raise_for_status()

    return latency


# Check API
health = requests.get(
    "http://127.0.0.1:8000/health",
    timeout=10,
)
health.raise_for_status()

print("API is healthy. Starting concurrent benchmark...")

# Warm-up
for _ in range(10):
    send_request()

latencies = []
errors = []

benchmark_start = time.perf_counter()

with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
    futures = [
        executor.submit(send_request)
        for _ in range(TOTAL_REQUESTS)
    ]

    for future in as_completed(futures):
        try:
            latencies.append(future.result())
        except Exception as error:
            errors.append(str(error))

total_time = time.perf_counter() - benchmark_start

if latencies:
    sorted_latencies = sorted(latencies)

    average_latency = statistics.mean(latencies)
    p50_latency = statistics.median(latencies)
    p95_latency = sorted_latencies[
        int(0.95 * (len(sorted_latencies) - 1))
    ]

    throughput = len(latencies) / total_time

    print("\n======= CONCURRENT PERFORMANCE =======")
    print(f"Total requests:       {TOTAL_REQUESTS}")
    print(f"Successful requests:  {len(latencies)}")
    print(f"Failed requests:      {len(errors)}")
    print(f"Concurrent workers:  {CONCURRENT_WORKERS}")
    print(f"Total time:           {total_time:.2f} seconds")
    print(f"Average latency:      {average_latency:.2f} ms")
    print(f"P50 latency:          {p50_latency:.2f} ms")
    print(f"P95 latency:          {p95_latency:.2f} ms")
    print(f"Throughput:           {throughput:.2f} req/sec")
    print("======================================")

if errors:
    print("\nFirst errors:")
    for error in errors[:5]:
        print(error)

