
import time
import requests
import statistics

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

WARMUP_REQUESTS = 10
TOTAL_REQUESTS = 100

# Check API availability
health = requests.get(
    "http://127.0.0.1:8000/health",
    timeout=10,
)
health.raise_for_status()

print("API is healthy. Starting benchmark...")

# Warm-up
for _ in range(WARMUP_REQUESTS):
    response = requests.post(API_URL, json=PAYLOAD, timeout=30)
    response.raise_for_status()

latencies = []

# Sequential benchmark
for _ in range(TOTAL_REQUESTS):
    start = time.perf_counter()

    response = requests.post(
        API_URL,
        json=PAYLOAD,
        timeout=30,
    )

    elapsed = time.perf_counter() - start
    response.raise_for_status()

    latencies.append(elapsed * 1000)

# Calculate metrics
average_latency = statistics.mean(latencies)
p50_latency = statistics.median(latencies)
p95_latency = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
total_time = sum(latencies) / 1000
throughput = TOTAL_REQUESTS / total_time

print("\n========== API PERFORMANCE ==========")
print(f"Requests:          {TOTAL_REQUESTS}")
print(f"Average latency:   {average_latency:.2f} ms")
print(f"P50 latency:       {p50_latency:.2f} ms")
print(f"P95 latency:       {p95_latency:.2f} ms")
print(f"Throughput:        {throughput:.2f} requests/sec")
print("=====================================")
