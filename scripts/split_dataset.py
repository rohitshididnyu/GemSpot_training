"""Split initial_training_set.csv into time-based train / validation CSVs.

Uses a temporal cutoff so the model trains on older data and validates
on newer data, which is how a recommender system works in production.

Supports --max-train-rows and --max-val-rows to downsample for memory-
constrained environments (e.g., 4 GB Chameleon VMs).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-based split of GemSpot dataset.")
    parser.add_argument("--input", default="data/demo/initial_training_set.csv", help="Path to full dataset.")
    parser.add_argument("--output-dir", default="data/demo", help="Directory for output CSVs.")
    parser.add_argument(
        "--cutoff",
        default="2021-05-01",
        help="Date cutoff (YYYY-MM-DD). Train = before cutoff, Val = on or after cutoff.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Max training rows (random sample, preserving class ratio). Use for low-memory VMs.",
    )
    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=None,
        help="Max validation rows (random sample, preserving class ratio).",
    )
    args = parser.parse_args()

    print(f"Loading {args.input}...", flush=True)
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df):,} rows", flush=True)

    df["_datetime"] = pd.to_datetime(df["time"], unit="ms")
    cutoff = pd.Timestamp(args.cutoff)

    train_df = df[df["_datetime"] < cutoff].drop(columns=["_datetime"])
    val_df = df[df["_datetime"] >= cutoff].drop(columns=["_datetime"])

    print(f"\nFull split: Train={len(train_df):,}  Val={len(val_df):,}", flush=True)

    # Downsample if requested (stratified to preserve class ratio)
    if args.max_train_rows and len(train_df) > args.max_train_rows:
        train_df = train_df.groupby("will_visit", group_keys=False).apply(
            lambda g: g.sample(
                n=int(args.max_train_rows * len(g) / len(train_df)),
                random_state=42,
            )
        ).reset_index(drop=True)
        print(f"Downsampled train -> {len(train_df):,} rows", flush=True)

    if args.max_val_rows and len(val_df) > args.max_val_rows:
        val_df = val_df.groupby("will_visit", group_keys=False).apply(
            lambda g: g.sample(
                n=int(args.max_val_rows * len(g) / len(val_df)),
                random_state=42,
            )
        ).reset_index(drop=True)
        print(f"Downsampled val   -> {len(val_df):,} rows", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "gemspot_train.csv"
    val_path = out / "gemspot_val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\nTrain: {len(train_df):,} rows -> {train_path}")
    print(f"  will_visit=1: {train_df['will_visit'].sum():,}  "
          f"=0: {(train_df['will_visit'] == 0).sum():,}  "
          f"({train_df['will_visit'].mean():.1%} positive)")
    print()
    print(f"Val:   {len(val_df):,} rows -> {val_path}")
    print(f"  will_visit=1: {val_df['will_visit'].sum():,}  "
          f"=0: {(val_df['will_visit'] == 0).sum():,}  "
          f"({val_df['will_visit'].mean():.1%} positive)")


if __name__ == "__main__":
    main()
