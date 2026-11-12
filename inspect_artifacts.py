
import sqlite3

connection = sqlite3.connect("mlflow.db")

print("EXPERIMENTS:")
for row in connection.execute(
    "SELECT experiment_id, name, artifact_location FROM experiments"
):
    print(row)

print("\nMLPILOT MODEL RUNS:")
for row in connection.execute("""
    SELECT run_uuid, artifact_uri
    FROM runs
    WHERE experiment_id = '1'
"""):
    print(row)

connection.close()
