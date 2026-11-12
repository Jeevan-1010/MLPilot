
import argparse
import os

import boto3
import psycopg
from dotenv import load_dotenv


PREFIX = "mlflow-artifacts"
MISSING_MODEL_ID = "m-b469613cdc91495590dcf87a7a55a16d"

EXPECTED_RUNS = 8
EXPECTED_MODELS_TO_UPDATE = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Commit artifact-location updates to Neon.",
    )
    args = parser.parse_args()

    load_dotenv(".env", override=True)

    required = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_ARTIFACTS_BUCKET",
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]

    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {missing}")

    bucket = os.environ["MLFLOW_ARTIFACTS_BUCKET"]
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )

    # Verify all uploaded objects before changing metadata.
    expected_objects = []

    for experiment_id in (1,):
        run_prefix = f"{PREFIX}/{experiment_id}/"

        # Seven model directories, each containing five files.
        model_ids = [
            "m-072553ab767f4125b45e5b0ac4a870e7",
            "m-53fe9ab44a714e76a5236ff41e76b45c",
            "m-9eb107bf50af46978010ae2c4e09f1d5",
            "m-d71601eac32c4f46b228a37e78834558",
            "m-e03cec7b2d3e46739cec364d33782761",
            "m-edfc8030d68f4d108fe8a1feba258f58",
            "m-efd37c0a73284674be8355c78aba7201",
        ]

        filenames = [
            "MLmodel",
            "conda.yaml",
            "model.skops",
            "python_env.yaml",
            "requirements.txt",
        ]

        for model_id in model_ids:
            for filename in filenames:
                expected_objects.append(
                    f"{run_prefix}models/{model_id}/artifacts/{filename}"
                )

        expected_objects.append(
            f"{run_prefix}2569663b1e9247ee8e6c13e4dd7a6f2c/"
            "artifacts/feature_schema.json"
        )

    for key in expected_objects:
        response = s3.head_object(Bucket=bucket, Key=key)
        if response["ContentLength"] <= 0:
            raise RuntimeError(f"Empty B2 object: {key}")

    print(f"B2 objects verified: {len(expected_objects)}")

    new_experiment_locations = {
        0: f"s3://{bucket}/{PREFIX}/0",
        1: f"s3://{bucket}/{PREFIX}/1",
    }

    with psycopg.connect(tracking_uri) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                # Verify experiment records and their current locations.
                cur.execute("""
                    SELECT experiment_id, artifact_location
                    FROM experiments
                    WHERE experiment_id IN (0, 1, 2)
                    ORDER BY experiment_id
                """)
                experiments = dict(cur.fetchall())

                expected_experiments = {
                    0: "file:///C:/Users/jeeva/Desktop/MLpilot/mlruns/0",
                    1: "file:///C:/Users/jeeva/Desktop/MLpilot/mlruns/1",
                    2: "mlflow-artifacts:/2",
                }

                if experiments != expected_experiments:
                    raise RuntimeError(
                        "Experiment locations differ from expected values. "
                        "No changes made."
                    )

                # Verify experiment 1 run records.
                cur.execute("""
                    SELECT run_uuid, artifact_uri
                    FROM runs
                    WHERE experiment_id = 1
                    ORDER BY run_uuid
                """)
                runs = cur.fetchall()

                if len(runs) != EXPECTED_RUNS:
                    raise RuntimeError(
                        f"Expected {EXPECTED_RUNS} runs; found {len(runs)}."
                    )

                for run_id, artifact_uri in runs:
                    expected = (
                        "file:///C:/Users/jeeva/Desktop/MLpilot/"
                        f"mlruns/1/{run_id}/artifacts"
                    )
                    if artifact_uri != expected:
                        raise RuntimeError(
                            f"Unexpected artifact URI for run {run_id}: "
                            f"{artifact_uri}"
                        )

                # Verify seven available logged-model records.
                cur.execute("""
                    SELECT model_id, artifact_location
                    FROM logged_models
                    WHERE experiment_id = 1
                    ORDER BY model_id
                """)
                models = dict(cur.fetchall())

                if len(models) != 8:
                    raise RuntimeError(
                        f"Expected 8 logged models; found {len(models)}."
                    )

                if MISSING_MODEL_ID not in models:
                    raise RuntimeError(
                        "Expected missing model record was not found."
                    )

                if models[MISSING_MODEL_ID] != (
                    "file:///C:/Users/jeeva/Desktop/MLpilot/"
                    f"mlruns/1/models/{MISSING_MODEL_ID}/artifacts"
                ):
                    raise RuntimeError(
                        "Missing model record has an unexpected location."
                    )

                available_model_ids = set(models) - {MISSING_MODEL_ID}

                if len(available_model_ids) != EXPECTED_MODELS_TO_UPDATE:
                    raise RuntimeError(
                        "Unexpected set of available logged models."
                    )

                for model_id in available_model_ids:
                    expected = (
                        "file:///C:/Users/jeeva/Desktop/MLpilot/"
                        f"mlruns/1/models/{model_id}/artifacts"
                    )
                    if models[model_id] != expected:
                        raise RuntimeError(
                            f"Unexpected location for model {model_id}."
                        )

                print("Neon preflight passed.")
                print("Experiment locations to update: 2")
                print(f"Run locations to update: {len(runs)}")
                print(f"Logged-model locations to update: {len(available_model_ids)}")
                print("Experiment 2 will remain unchanged.")
                print("Missing logged-model record will remain unchanged.")

                if not args.execute:
                    print("\nDRY RUN ONLY. Neon was not modified.")
                    print("Run with --execute to commit these updates.")
                    return

                # Update only the verified records.
                for experiment_id, location in new_experiment_locations.items():
                    cur.execute("""
                        UPDATE experiments
                        SET artifact_location = %s
                        WHERE experiment_id = %s
                    """, (location, experiment_id))

                    if cur.rowcount != 1:
                        raise RuntimeError(
                            f"Unexpected update count for experiment {experiment_id}."
                        )

                for run_id, _ in runs:
                    location = (
                        f"s3://{bucket}/{PREFIX}/1/{run_id}/artifacts"
                    )
                    cur.execute("""
                        UPDATE runs
                        SET artifact_uri = %s
                        WHERE run_uuid = %s AND experiment_id = 1
                    """, (location, run_id))

                    if cur.rowcount != 1:
                        raise RuntimeError(
                            f"Unexpected update count for run {run_id}."
                        )

                for model_id in available_model_ids:
                    location = (
                        f"s3://{bucket}/{PREFIX}/1/models/"
                        f"{model_id}/artifacts"
                    )
                    cur.execute("""
                        UPDATE logged_models
                        SET artifact_location = %s
                        WHERE model_id = %s AND experiment_id = 1
                    """, (location, model_id))

                    if cur.rowcount != 1:
                        raise RuntimeError(
                            f"Unexpected update count for model {model_id}."
                        )

                # Verify the new values before transaction commit.
                cur.execute("""
                    SELECT COUNT(*)
                    FROM experiments
                    WHERE experiment_id IN (0, 1)
                      AND artifact_location LIKE 's3://%%'
                """)
                if cur.fetchone()[0] != 2:
                    raise RuntimeError("Experiment verification failed.")

                cur.execute("""
                    SELECT COUNT(*)
                    FROM runs
                    WHERE experiment_id = 1
                      AND artifact_uri LIKE 's3://%%'
                """)
                if cur.fetchone()[0] != EXPECTED_RUNS:
                    raise RuntimeError("Run verification failed.")

                cur.execute("""
                    SELECT COUNT(*)
                    FROM logged_models
                    WHERE experiment_id = 1
                      AND model_id <> %s
                      AND artifact_location LIKE 's3://%%'
                """, (MISSING_MODEL_ID,))
                if cur.fetchone()[0] != EXPECTED_MODELS_TO_UPDATE:
                    raise RuntimeError("Logged-model verification failed.")

                cur.execute("""
                    SELECT artifact_location
                    FROM logged_models
                    WHERE model_id = %s
                """, (MISSING_MODEL_ID,))
                if cur.fetchone()[0] != models[MISSING_MODEL_ID]:
                    raise RuntimeError(
                        "Missing model record was unexpectedly changed."
                    )

                cur.execute("""
                    SELECT artifact_location
                    FROM experiments
                    WHERE experiment_id = 2
                """)
                if cur.fetchone()[0] != "mlflow-artifacts:/2":
                    raise RuntimeError(
                        "Experiment 2 was unexpectedly changed."
                    )

        print("\nNEON ARTIFACT LOCATION UPDATE COMMITTED.")


if __name__ == "__main__":
    main()
