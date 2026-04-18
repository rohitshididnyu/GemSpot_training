"""Split a GemSpot CSV into train / val (/ test) sets.

Supports THREE split modes:

  1. --mode time         Time-based: rows before --cutoff go to train; after
                         go to val/test (production-like).

  2. --mode percent      Random, stratified on target (preserves class ratio).
                         Controlled by --train-pct.

  3. --mode sequential   Linear positional split: the FIRST train_pct of
                         rows (as they appear in the CSV) → train;
                         remaining → val/test. No shuffling, no date math.

All modes support an optional THIRD split via --test-pct:
  - --test-pct 0   (default) → 2-way train/val
  - --test-pct 0.1            → 3-way train/val/test where train=80%,
                                 val=10%, test=10% (when --train-pct=0.8)

Outputs
-------
  gemspot_train.csv   (always)
  gemspot_val.csv     (always)
  gemspot_test.csv    (only when --test-pct > 0)

Examples
--------
# Time split (production default):
python3 scripts/split_dataset.py \\
    --input data/demo/initial_training_set.csv \\
    --mode time --cutoff 2021-05-01

# Random 80/20 stratified:
python3 scripts/split_dataset.py \\
    --input data/demo/initial_training_set.csv \\
    --mode percent --train-pct 0.80

# Linear (sequential) 80/20 — first 80% of rows are train:
python3 scripts/split_dataset.py \\
    --input data/demo/initial_training_set.csv \\
    --mode sequential --train-pct 0.80

# 3-way: sequential 70/15/15 train/val/test:
python3 scripts/split_dataset.py \\
    --input data/demo/initial_training_set.csv \\
    --mode sequential --train-pct 0.70 --test-pct 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Split strategies (each returns a list of DataFrames: [train, val] or
# [train, val, test] depending on test_pct)
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    cutoff: str,
    test_pct: float,
    seed: int,
) -> list[pd.DataFrame]:
    """Time-based split. Optionally carves a test tail from the POST-cutoff
    portion where test = the latest rows (simulating 'most-recent' test set).
    """
    df = df.copy()
    df["_datetime"] = pd.to_datetime(df["time"], unit="ms")
    cutoff_ts = pd.Timestamp(cutoff)
    train_df = df[df["_datetime"] < cutoff_ts].drop(columns=["_datetime"])
    post_df = df[df["_datetime"] >= cutoff_ts].drop(columns=["_datetime"])

    if test_pct > 0:
        # Sort post-cutoff by time, then take the last test_pct * TOTAL rows as test
        post_df = post_df.sort_values("time").reset_index(drop=True)
        total_n = len(train_df) + len(post_df)
        n_test = min(int(round(test_pct * total_n)), len(post_df))
        test_df = post_df.iloc[len(post_df) - n_test :].reset_index(drop=True)
        val_df = post_df.iloc[: len(post_df) - n_test].reset_index(drop=True)
        return [train_df, val_df, test_df]

    return [train_df, post_df]


def percent_split(
    df: pd.DataFrame,
    train_pct: float,
    test_pct: float,
    target_col: str,
    seed: int,
) -> list[pd.DataFrame]:
    """Stratified random split on target_col."""
    _validate_fractions(train_pct, test_pct)

    train_parts, val_parts, test_parts = [], [], []
    for _, group in df.groupby(target_col, group_keys=False):
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        n_train = int(n * train_pct)
        n_test = int(n * test_pct) if test_pct > 0 else 0
        n_val = n - n_train - n_test

        train_parts.append(shuffled.iloc[:n_train])
        val_parts.append(shuffled.iloc[n_train : n_train + n_val])
        if n_test > 0:
            test_parts.append(shuffled.iloc[n_train + n_val :])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if test_pct > 0:
        test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return [train_df, val_df, test_df]
    return [train_df, val_df]


def sequential_split(
    df: pd.DataFrame,
    train_pct: float,
    test_pct: float,
) -> list[pd.DataFrame]:
    """Linear positional split — NO shuffle, NO sorting.

    Row order is preserved exactly as it appears in the input CSV.
      - first train_pct * N rows → train
      - next  val_pct  * N rows → val
      - last  test_pct * N rows → test  (if test_pct > 0)
    """
    _validate_fractions(train_pct, test_pct)

    n = len(df)
    n_train = int(n * train_pct)
    n_test = int(n * test_pct) if test_pct > 0 else 0
    n_val = n - n_train - n_test

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)

    if test_pct > 0:
        test_df = df.iloc[n_train + n_val :].reset_index(drop=True)
        return [train_df, val_df, test_df]
    return [train_df, val_df]


def _validate_fractions(train_pct: float, test_pct: float) -> None:
    if not 0.0 < train_pct < 1.0:
        raise ValueError(f"--train-pct must be in (0, 1); got {train_pct}")
    if test_pct < 0 or test_pct >= 1:
        raise ValueError(f"--test-pct must be in [0, 1); got {test_pct}")
    if train_pct + test_pct >= 1.0:
        raise ValueError(
            f"--train-pct + --test-pct must be < 1 (got "
            f"{train_pct} + {test_pct} = {train_pct + test_pct}); "
            f"val gets whatever's left."
        )


# ---------------------------------------------------------------------------
# Downsampling helper
# ---------------------------------------------------------------------------

def stratified_downsample(
    df: pd.DataFrame,
    target_col: str,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.groupby(target_col, group_keys=False).apply(
        lambda g: g.sample(
            n=max(1, int(max_rows * len(g) / len(df))),
            random_state=seed,
        )
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Split a GemSpot CSV into train/val(/test).")
    parser.add_argument("--input", default="data/demo/initial_training_set.csv")
    parser.add_argument("--output-dir", default="data/demo")
    parser.add_argument(
        "--mode",
        choices=["time", "percent", "sequential"],
        default="time",
        help="Split mode.",
    )
    parser.add_argument("--cutoff", default="2021-05-01",
                        help="[time mode] Date cutoff (YYYY-MM-DD).")
    parser.add_argument("--train-pct", type=float, default=0.85,
                        help="[percent/sequential] Training fraction (default 0.85).")
    parser.add_argument("--test-pct", type=float, default=0.0,
                        help="Test fraction for 3-way split (default 0 = no test set).")
    parser.add_argument("--target-col", default="will_visit",
                        help="Target column (used for stratification in percent mode).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--train-out-name", default="gemspot_train.csv")
    parser.add_argument("--val-out-name", default="gemspot_val.csv")
    parser.add_argument("--test-out-name", default="gemspot_test.csv")
    args = parser.parse_args()

    print(f"Loading {args.input}...", flush=True)
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns", flush=True)

    if args.target_col not in df.columns:
        raise ValueError(
            f"Target column '{args.target_col}' not in CSV. "
            f"Found: {list(df.columns)}"
        )

    # ---- Split ----
    if args.mode == "time":
        print(f"Using TIME split at {args.cutoff}"
              f"{' + test tail' if args.test_pct > 0 else ''}")
        splits = time_split(df, args.cutoff, args.test_pct, args.seed)
    elif args.mode == "percent":
        print(f"Using PERCENT split (train={args.train_pct:.0%}, "
              f"val={1 - args.train_pct - args.test_pct:.0%}, "
              f"test={args.test_pct:.0%}), stratified on '{args.target_col}'")
        splits = percent_split(
            df, args.train_pct, args.test_pct, args.target_col, args.seed
        )
    else:  # sequential
        print(f"Using SEQUENTIAL (linear) split — first {args.train_pct:.0%} of rows → train, "
              f"{1 - args.train_pct - args.test_pct:.0%} → val"
              f"{f', last {args.test_pct:.0%} → test' if args.test_pct > 0 else ''}")
        splits = sequential_split(df, args.train_pct, args.test_pct)

    # splits is a list of [train, val] or [train, val, test]
    names = ["Train", "Val", "Test"][: len(splits)]
    out_names = [args.train_out_name, args.val_out_name, args.test_out_name][: len(splits)]
    caps = [args.max_train_rows, args.max_val_rows, args.max_test_rows][: len(splits)]

    print(f"\nPre-downsample sizes: "
          + "  ".join(f"{n}={len(d):,}" for n, d in zip(names, splits)),
          flush=True)

    # ---- Downsample each split ----
    for i, (name, cap) in enumerate(zip(names, caps)):
        if cap is not None:
            splits[i] = stratified_downsample(splits[i], args.target_col, cap, args.seed)
            print(f"Downsampled {name.lower()} -> {len(splits[i]):,} rows", flush=True)

    # ---- Write + Report ----
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, out_name, df_split in zip(names, out_names, splits):
        path = out / out_name
        df_split.to_csv(path, index=False)
        t = args.target_col
        print(f"\n{name}: {len(df_split):,} rows -> {path}")
        if t in df_split.columns and len(df_split) > 0:
            print(
                f"  {t}=1: {int(df_split[t].sum()):,}  "
                f"=0: {int((df_split[t] == 0).sum()):,}  "
                f"({df_split[t].mean():.1%} positive)"
            )


if __name__ == "__main__":
    main()
