"""Split initial_training_set.csv into time-based train / validation CSVs.

Uses a temporal cutoff so the model trains on older data and validates
on newer data, which is how a recommender system works in production.
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
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["_datetime"] = pd.to_datetime(df["time"], unit="ms")

    cutoff = pd.Timestamp(args.cutoff)

    train_df = df[df["_datetime"] < cutoff].drop(columns=["_datetime"])
    val_df = df[df["_datetime"] >= cutoff].drop(columns=["_datetime"])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "gemspot_train.csv"
    val_path = out / "gemspot_val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Loaded {len(df):,} rows from {args.input}")
    print(f"Time range: {df['_datetime'].min()} to {df['_datetime'].max()}")
    print(f"Cutoff:     {cutoff}")
    print()
    print(f"Train: {len(train_df):,} rows (before {args.cutoff}) -> {train_path}")
    print(f"  will_visit=1: {train_df['will_visit'].sum():,}  =0: {(train_df['will_visit'] == 0).sum():,}  ({train_df['will_visit'].mean():.1%} positive)")
    print()
    print(f"Val:   {len(val_df):,} rows (on or after {args.cutoff}) -> {val_path}")
    print(f"  will_visit=1: {val_df['will_visit'].sum():,}  =0: {(val_df['will_visit'] == 0).sum():,}  ({val_df['will_visit'].mean():.1%} positive)")


if __name__ == "__main__":
    main()
