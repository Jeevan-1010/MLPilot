
import sqlite3

run_ids = [
    "58586b3ad8dc4a999e278670aeaf156d",
    "2e351080f26a4eb089145ebf5f5a7e5c",
    "97699a45cda7428d84883a1fb3dc913d",
    "0b25b373a6d54b58b63cd5344040ca2c",
    "e39a63ccf96342acbc72019c5595a06b",
]

connection = sqlite3.connect("mlflow.db")

placeholders = ",".join("?" for _ in run_ids)

query = f"""
SELECT run_uuid, artifact_uri
FROM runs
WHERE run_uuid IN ({placeholders})
"""

for row in connection.execute(query, run_ids):
    print(row)

connection.close()
