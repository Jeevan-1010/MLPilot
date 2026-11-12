
import sqlite3

connection = sqlite3.connect("mlflow.db")

print("REGISTERED MODEL VERSIONS:")

rows = connection.execute("""
    SELECT name, version, source, storage_location, run_id
    FROM model_versions
    WHERE name = 'MLPilot-Demand-Forecasting'
    ORDER BY CAST(version AS INTEGER)
""").fetchall()

for row in rows:
    print(row)

print("\nLOGGED MODEL ARTIFACT LOCATIONS:")

rows = connection.execute("""
    SELECT model_id, artifact_location, source_run_id
    FROM logged_models
    WHERE model_id IN (
        'm-efd37c0a73284674be8355c78aba7201',
        'm-e03cec7b2d3e46739cec364d33782761',
        'm-d71601eac32c4f46b228a37e78834558',
        'm-9eb107bf50af46978010ae2c4e09f1d5',
        'm-edfc8030d68f4d108fe8a1feba258f58'
    )
""").fetchall()

for row in rows:
    print(row)

connection.close()
