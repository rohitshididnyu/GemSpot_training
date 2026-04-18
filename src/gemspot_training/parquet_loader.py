"""Parquet dataset loader for GemSpot.

Backs the MinIO/Iceberg table:
    agent-datalake/iceberg/adventurelog.db/training_data/data

and also works with parquet files placed locally in `data/demo/`.

The `batch_pipeline` upstream appends each week's new data onto the
previous training set and re-writes a single parquet file. The file
already contains:
  - All accumulated rows (old + new),
  - A `split` column with values {"train", "eval"} computed
    chronologically by the pipeline, preventing data leakage.

This module:
  1. Resolves a parquet source (local path, local directory, or S3/MinIO
     URL) to the LATEST file.
  2. Reads it into a pandas DataFrame.
  3. Applies column renames + canonical schema enforcement.
  4. Splits it into train / eval based on the `split` column.

No dependencies on boto3 are required for local use — only when pulling
from MinIO / S3.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ParquetSplitBundle:
    """Train + Eval split of a single parquet snapshot (after schema canonicalisation)."""
    train_frame: pd.DataFrame
    eval_frame: pd.DataFrame
    pipeline_run_date: Optional[str] = None
    source_path: str = ""


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

def _list_local_parquets(directory: Path) -> list[Path]:
    return sorted([p for p in directory.glob("*.parquet") if p.is_file()])


def resolve_parquet_source(source: str) -> str:
    """Resolve a user-supplied source string to a concrete parquet file path.

    Accepts:
      - /path/to/file.parquet         → used directly
      - /path/to/directory/           → picks the most recently modified .parquet
      - s3://bucket/prefix/           → (MinIO) picks latest object with .parquet suffix
      - s3://bucket/prefix/file.parquet → direct S3 key

    Returns either a local file path (string) or an s3:// URL. For s3:// URLs
    the caller is expected to use `load_parquet_from_source` which handles
    the download transparently.
    """
    if source.startswith("s3://"):
        return _resolve_s3_source(source)

    path = Path(source)
    if path.is_file():
        return str(path)
    if path.is_dir():
        candidates = _list_local_parquets(path)
        if not candidates:
            raise FileNotFoundError(
                f"No .parquet files found in directory: {path}"
            )
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"  [parquet] Resolved latest parquet: {latest.name}", flush=True)
        return str(latest)

    raise FileNotFoundError(f"Parquet source does not exist: {source}")


def _resolve_s3_source(s3_url: str) -> str:
    """For an s3:// URL pointing to a prefix, find the newest .parquet object.

    Requires boto3 + MinIO/S3 credentials via environment:
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL (for MinIO)
    """
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required to read from s3://… URLs. "
            "Install it with: pip install boto3"
        ) from exc

    bucket, _, key = s3_url[len("s3://"):].partition("/")
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    s3 = boto3.client("s3", endpoint_url=endpoint_url)

    # If the URL already ends in .parquet, return it unchanged.
    if key.endswith(".parquet"):
        return s3_url

    # Otherwise treat `key` as a prefix and find the newest parquet under it.
    print(f"  [parquet] Listing s3://{bucket}/{key}...", flush=True)
    paginator = s3.get_paginator("list_objects_v2")
    newest = None
    newest_ts = None
    for page in paginator.paginate(Bucket=bucket, Prefix=key):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith(".parquet"):
                continue
            if newest_ts is None or obj["LastModified"] > newest_ts:
                newest = obj["Key"]
                newest_ts = obj["LastModified"]
    if newest is None:
        raise FileNotFoundError(f"No .parquet objects found under s3://{bucket}/{key}")
    resolved = f"s3://{bucket}/{newest}"
    print(f"  [parquet] Resolved latest S3 object: {resolved}", flush=True)
    return resolved


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_parquet_from_source(resolved_source: str) -> pd.DataFrame:
    """Read a parquet file into a DataFrame. Supports local paths and s3:// URLs.

    For s3:// URLs, requires `s3fs` (or boto3+pyarrow configured for MinIO).
    """
    if resolved_source.startswith("s3://"):
        # pandas delegates to s3fs / pyarrow which respect AWS_* env vars.
        storage_options = {}
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        if endpoint:
            storage_options["client_kwargs"] = {"endpoint_url": endpoint}
        return pd.read_parquet(resolved_source, storage_options=storage_options or None)
    return pd.read_parquet(resolved_source)


# ---------------------------------------------------------------------------
# Train/Eval split using the `split` column
# ---------------------------------------------------------------------------

def load_and_split_parquet(
    source: str,
    config: dict,
) -> ParquetSplitBundle:
    """Resolve a parquet source, load it, rename/enforce schema, then split.

    Uses the parquet's `split` column (produced by batch_pipeline) as
    the ground truth partition. This avoids recomputing splits on
    every run and guarantees consistency with the upstream producer.

    If `split_column` is configured but absent from the data, raises.
    """
    from .data import apply_column_rename, enforce_canonical_schema

    resolved = resolve_parquet_source(source)
    print(f"  [parquet] Reading {resolved}...", flush=True)
    df = load_parquet_from_source(resolved)
    print(f"  [parquet] Loaded {len(df):,} rows × {len(df.columns)} cols", flush=True)

    dataset_cfg = config["dataset"]
    split_col = dataset_cfg.get("split_column", "split")
    train_value = dataset_cfg.get("split_train_value", "train")
    eval_value = dataset_cfg.get("split_eval_value", "eval")

    # Capture pipeline_run_date for logging BEFORE canonicalization removes it.
    run_date = None
    if "pipeline_run_date" in df.columns and len(df) > 0:
        run_date = str(df["pipeline_run_date"].iloc[0])
        print(f"  [parquet] pipeline_run_date = {run_date}", flush=True)

    # Apply renames early so downstream code sees the canonical names.
    rename_map = dataset_cfg.get("column_rename", {})
    df = apply_column_rename(df, rename_map)

    # Split BEFORE canonical enforcement (since `split` is not canonical).
    if split_col not in df.columns:
        raise ValueError(
            f"Expected `{split_col}` column in parquet but it's missing. "
            f"Available: {list(df.columns)}"
        )
    train_mask = df[split_col] == train_value
    eval_mask = df[split_col] == eval_value
    train_df = df[train_mask].drop(columns=[split_col])
    eval_df = df[eval_mask].drop(columns=[split_col])

    print(
        f"  [parquet] Split '{split_col}': "
        f"train={len(train_df):,}  eval={len(eval_df):,}",
        flush=True,
    )

    return ParquetSplitBundle(
        train_frame=train_df,
        eval_frame=eval_df,
        pipeline_run_date=run_date,
        source_path=resolved,
    )


def filter_latest_pipeline_date(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the rows belonging to the most recent pipeline_run_date.

    Useful when you want to retrain on the DELTA only (just this week's
    additions), not the full accumulative set.
    """
    if "pipeline_run_date" not in df.columns:
        return df
    latest = df["pipeline_run_date"].max()
    print(f"  [parquet] Keeping only rows with pipeline_run_date == {latest}", flush=True)
    return df[df["pipeline_run_date"] == latest].copy()
