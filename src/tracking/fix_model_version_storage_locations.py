
import os

import psycopg
from dotenv import load_dotenv


MODEL_NAME = "MLPilot-Demand-Forecasting"
BUCKET = "mlpilot-artifacts"
PREFIX = "mlflow-artifacts/1"
WINDOWS_ROOT = "file:///C:/Users/jeeva/Desktop/MLpilot/mlruns/1/models"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    load_dotenv(".env", override=True)
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    with psycopg.connect(tracking_uri) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT version, source, storage_location
                    FROM model_versions
                    WHERE name = %s
                    ORDER BY version
                """, (MODEL_NAME,))

                versions = cur.fetchall()

                if len(versions) != 5:
                    raise RuntimeError(
                        f"Expected 5 versions, found {len(versions)}."
                    )

                updates = []

                for version, source, old_location in versions:
                    prefix = "models:/"

                    if not source.startswith(prefix):
                        raise RuntimeError(
                            f"Unexpected source for version {version}: {source}"
                        )

                    model_id = source[len(prefix):]

                    expected_old = (
                        f"{WINDOWS_ROOT}/{model_id}/artifacts"
                    )
                    expected_new = (
                        f"s3://{BUCKET}/{PREFIX}/models/"
                        f"{model_id}/artifacts"
                    )

                    if old_location != expected_old:
                        raise RuntimeError(
                            f"Unexpected old location for version {version}: "
                            f"{old_location}"
                        )

                    cur.execute("""
                        SELECT artifact_location
                        FROM logged_models
                        WHERE model_id = %s
                    """, (model_id,))

                    row = cur.fetchone()

                    if row is None or row[0] != expected_new:
                        raise RuntimeError(
                            f"B2 metadata mismatch for model {model_id}."
                        )

                    updates.append(
                        (version, model_id, expected_old, expected_new)
                    )

                print("Preflight passed.")
                print(f"Model versions verified: {len(updates)}")

                for version, model_id, old, new in updates:
                    print(f"Version {version}: {old} -> {new}")

                if not args.execute:
                    print("\nDRY RUN ONLY. No changes made.")
                    return

                for version, model_id, old, new in updates:
                    cur.execute("""
                        UPDATE model_versions
                        SET storage_location = %s
                        WHERE name = %s
                          AND version = %s
                          AND storage_location = %s
                    """, (new, MODEL_NAME, version, old))

                    if cur.rowcount != 1:
                        raise RuntimeError(
                            f"Unexpected update count for version {version}."
                        )

                cur.execute("""
                    SELECT COUNT(*)
                    FROM model_versions
                    WHERE name = %s
                      AND storage_location LIKE 's3://%%'
                """, (MODEL_NAME,))

                if cur.fetchone()[0] != 5:
                    raise RuntimeError(
                        "Post-update verification failed."
                    )

                print("\nMODEL VERSION LOCATIONS UPDATED AND VERIFIED.")


if __name__ == "__main__":
    main()
