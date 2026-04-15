#!/usr/bin/env python3
"""
GemSpot — Export Training Data from Iceberg / MinIO
====================================================
Reads the latest transformed Parquet dataset from MinIO (produced by the
batch pipeline), splits by the 'split' column, and writes train.csv / val.csv
locally or back to MinIO for the training pipeline to consume.

This bridges the gap between the data pipeline (Iceberg/Parquet) and the
training pipeline (CSV-based).

Usage:
    python scripts/export_training_data.py \\
        --minio-endpoint http://minio:9000 \\
        --output-dir /data/training_sets
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BUCKET = "agent-datalake"


def get_latest_transformed_key(s3, bucket: str) -> str | None:
    """Find the most recent transformed/ parquet file in MinIO."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix="transformed/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])

    if not keys:
        # Also check metadata for the latest pipeline run
        return None

    # Sort by key (date-based prefix) to get latest
    keys.sort(reverse=True)
    return keys[0]


def get_latest_metadata(s3, bucket: str) -> dict | None:
    """Read the latest pipeline metadata to find the Iceberg snapshot info."""
    import json
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix="metadata/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                keys.append(obj["Key"])

    if not keys:
        return None

    keys.sort(reverse=True)
    obj = s3.get_object(Bucket=bucket, Key=keys[0])
    return json.loads(obj["Body"].read().decode())


def read_parquet_from_s3(s3, bucket: str, key: str) -> pd.DataFrame:
    """Read a Parquet file from S3/MinIO into a DataFrame."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")


def try_read_from_iceberg(
    iceberg_db_url: str,
    minio_endpoint: str,
    aws_key: str,
    aws_secret: str,
) -> pd.DataFrame | None:
    """Try reading directly from the Iceberg table."""
    try:
        from pyiceberg.catalog import load_catalog

        catalog = load_catalog(
            "adventurelog",
            type="sql",
            uri=iceberg_db_url,
            warehouse=f"s3://{BUCKET}/iceberg",
            **{
                "s3.endpoint": minio_endpoint,
                "s3.access-key-id": aws_key,
                "s3.secret-access-key": aws_secret,
                "s3.path-style-access": "true",
            },
        )
        table = catalog.load_table("adventurelog.training_data")
        scan = table.scan()
        df = scan.to_pandas()
        if not df.empty:
            logger.info("Read %d rows from Iceberg table", len(df))
            return df
    except Exception as e:
        logger.warning("Iceberg read failed: %s", e)
    return None


def export(
    minio_endpoint: str,
    aws_key: str,
    aws_secret: str,
    output_dir: str,
    iceberg_db_url: str | None = None,
    upload_to_s3: bool = False,
):
    """Main export logic."""
    s3 = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
    )

    df = None

    # Strategy 1: Read from Iceberg table
    if iceberg_db_url:
        df = try_read_from_iceberg(iceberg_db_url, minio_endpoint, aws_key, aws_secret)

    # Strategy 2: Read latest transformed parquet from MinIO
    if df is None:
        key = get_latest_transformed_key(s3, BUCKET)
        if key:
            logger.info("Reading from s3://%s/%s", BUCKET, key)
            df = read_parquet_from_s3(s3, BUCKET, key)
        else:
            logger.error("No transformed data found in MinIO bucket '%s'", BUCKET)
            sys.exit(1)

    if df.empty:
        logger.error("Dataset is empty — nothing to export")
        sys.exit(1)

    logger.info("Total rows: %d | Columns: %s", len(df), list(df.columns))

    # Split by 'split' column
    if "split" not in df.columns:
        logger.error("Dataset missing 'split' column")
        sys.exit(1)

    train_df = df[df["split"] == "train"].drop(columns=["split", "pipeline_run_date", "snapshot_id"], errors="ignore")
    val_df = df[df["split"] == "eval"].drop(columns=["split", "pipeline_run_date", "snapshot_id"], errors="ignore")

    logger.info("Train: %d rows | Val: %d rows", len(train_df), len(val_df))

    # Write locally
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train.csv"
    val_path = out / "val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    logger.info("Wrote %s (%d rows)", train_path, len(train_df))
    logger.info("Wrote %s (%d rows)", val_path, len(val_df))

    # Optionally upload to MinIO
    if upload_to_s3:
        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for name, local_path in [("train.csv", train_path), ("val.csv", val_path)]:
            key = f"training_sets/{run_date}/{name}"
            s3.upload_file(str(local_path), BUCKET, key)
            logger.info("Uploaded s3://%s/%s", BUCKET, key)


def main():
    parser = argparse.ArgumentParser(description="Export training data from Iceberg/MinIO")
    parser.add_argument("--minio-endpoint", default=os.getenv("ENDPOINT_URL", "http://minio:9000"))
    parser.add_argument("--aws-key", default=os.getenv("AWS_ACCESS_KEY_ID", "admin"))
    parser.add_argument("--aws-secret", default=os.getenv("AWS_SECRET_ACCESS_KEY", "password"))
    parser.add_argument("--iceberg-db-url", default=os.getenv("ICEBERG_DB_URL"))
    parser.add_argument("--output-dir", default="/data/training_sets")
    parser.add_argument("--upload-to-s3", action="store_true")
    args = parser.parse_args()

    export(
        minio_endpoint=args.minio_endpoint,
        aws_key=args.aws_key,
        aws_secret=args.aws_secret,
        output_dir=args.output_dir,
        iceberg_db_url=args.iceberg_db_url,
        upload_to_s3=args.upload_to_s3,
    )


if __name__ == "__main__":
    main()
