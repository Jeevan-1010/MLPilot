import argparse
import hashlib
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv


ROOT = Path("mlruns")
PREFIX = "mlflow-artifacts/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Upload files to Backblaze B2.",
    )
    args = parser.parse_args()

    load_dotenv(".env", override=True)

    required = [
        "MLFLOW_ARTIFACTS_BUCKET",
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]

    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise RuntimeError(f"Missing environment variables: {missing}")

    if not ROOT.is_dir():
        raise FileNotFoundError(f"Directory not found: {ROOT}")

    bucket = os.environ["MLFLOW_ARTIFACTS_BUCKET"]

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )

    files = sorted(path for path in ROOT.rglob("*") if path.is_file())

    if len(files) != 36:
        raise RuntimeError(
            f"Expected 36 files, found {len(files)}. Aborting."
        )

    upload_plan = []

    for path in files:
        key = PREFIX + path.relative_to(ROOT).as_posix()

        try:
            s3.head_object(Bucket=bucket, Key=key)
            raise RuntimeError(
                f"Target already exists; refusing to overwrite: {key}"
            )
        except s3.exceptions.ClientError as exc:
            status = exc.response["ResponseMetadata"]["HTTPStatusCode"]
            if status != 404:
                raise

        upload_plan.append((path, key, sha256_file(path)))

    print(f"Preflight passed: {len(upload_plan)} files")
    print(f"Total bytes: {sum(p.stat().st_size for p, _, _ in upload_plan):,}")

    if not args.execute:
        print("DRY RUN ONLY. No files uploaded.")
        print("Run with --execute to upload.")
        return

    for index, (path, key, digest) in enumerate(upload_plan, start=1):
        s3.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={
                "Metadata": {
                    "sha256": digest,
                }
            },
        )
        print(f"Uploaded {index}/{len(upload_plan)}: {key}")

    print("\nVerifying uploaded objects...")

    for path, key, digest in upload_plan:
        response = s3.head_object(Bucket=bucket, Key=key)

        if response["ContentLength"] != path.stat().st_size:
            raise RuntimeError(f"Size mismatch: {key}")

        if response.get("Metadata", {}).get("sha256") != digest:
            raise RuntimeError(f"SHA-256 metadata mismatch: {key}")

    print(f"\nUPLOAD AND VERIFICATION SUCCESSFUL: {len(upload_plan)} files")
    print("Original local mlruns files were not modified.")


if __name__ == "__main__":
    main()