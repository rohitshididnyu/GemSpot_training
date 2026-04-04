"""Data loading and preprocessing for GemSpot real dataset.

The real CSV contains list-encoded columns stored as strings:
  - category_encoded:           21-element multi-hot int vector
  - destination_vibe_tag:       20-element binary int vector
  - user_personal_preferences:  20-element float vector

This module parses those strings into individual numeric columns so that
every feature fed to the model is a plain number.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public data container
# ---------------------------------------------------------------------------

@dataclass
class DatasetBundle:
    train_features: pd.DataFrame
    train_target: pd.Series
    val_features: pd.DataFrame
    val_target: pd.Series


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _safe_parse_list(value: object) -> list:
    """Parse a string like '[0, 1, 0.5]' into a Python list."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def explode_list_column(
    frame: pd.DataFrame,
    column: str,
    expected_length: int,
    prefix: str,
) -> pd.DataFrame:
    """Expand a list-encoded string column into N individual numeric columns.

    For example, a column containing '[0, 1, 0]' with expected_length=3 and
    prefix='cat' produces columns cat_0, cat_1, cat_2.
    """
    parsed = frame[column].apply(_safe_parse_list)

    # Pad or truncate to expected_length
    def _normalise(lst: list) -> list:
        if len(lst) >= expected_length:
            return lst[:expected_length]
        return lst + [0] * (expected_length - len(lst))

    matrix = np.array(parsed.apply(_normalise).tolist(), dtype=np.float64)
    col_names = [f"{prefix}_{i}" for i in range(expected_length)]
    return pd.DataFrame(matrix, columns=col_names, index=frame.index)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_schema(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Main preparation
# ---------------------------------------------------------------------------

def prepare_frame(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Transform a raw CSV frame into a model-ready DataFrame.

    Steps:
      1. Drop ID/redundant columns
      2. Explode list-encoded columns into individual numeric columns
      3. Create interaction features (user_pref * destination_vibe)
      4. Keep scalar numeric features as-is
      5. Return clean frame with target column intact
    """
    dataset_cfg = config["dataset"]
    target = dataset_cfg["target_column"]
    drop_cols = dataset_cfg.get("drop_columns", [])
    scalar_features = dataset_cfg.get("scalar_numeric_features", [])
    list_features = dataset_cfg.get("list_encoded_features", {})
    interaction_cfg = dataset_cfg.get("interaction_features", [])

    # Validate minimum required columns
    validate_schema(frame, [target])

    prepared = frame.copy()

    # Drop unwanted columns
    cols_to_drop = [c for c in drop_cols if c in prepared.columns]
    prepared = prepared.drop(columns=cols_to_drop)

    # Explode each list-encoded column
    expanded_parts: list[pd.DataFrame] = []
    list_col_names_to_drop: list[str] = []

    for col_name, col_cfg in list_features.items():
        if col_name not in prepared.columns:
            continue
        length = col_cfg["length"]
        prefix = col_cfg["prefix"]
        expanded = explode_list_column(prepared, col_name, length, prefix)
        expanded_parts.append(expanded)
        list_col_names_to_drop.append(col_name)

    # Drop original list-string columns and concat expanded versions
    prepared = prepared.drop(columns=list_col_names_to_drop)
    for part in expanded_parts:
        prepared = pd.concat([prepared, part], axis=1)

    # Create interaction features (element-wise multiply)
    for interaction in interaction_cfg:
        src_a_prefix = interaction["a"]
        src_b_prefix = interaction["b"]
        out_prefix = interaction["prefix"]
        # Find the columns matching each prefix
        a_cols = sorted([c for c in prepared.columns if c.startswith(f"{src_a_prefix}_")])
        b_cols = sorted([c for c in prepared.columns if c.startswith(f"{src_b_prefix}_")])
        n = min(len(a_cols), len(b_cols))
        for i in range(n):
            prepared[f"{out_prefix}_{i}"] = prepared[a_cols[i]] * prepared[b_cols[i]]

    # Ensure target is int
    prepared[target] = prepared[target].astype(int)

    return prepared


# ---------------------------------------------------------------------------
# Bundle builder
# ---------------------------------------------------------------------------

def make_dataset_bundle(
    train_csv: str | Path,
    val_csv: str | Path,
    config: dict,
) -> DatasetBundle:
    train_frame = prepare_frame(load_csv(train_csv), config)
    val_frame = prepare_frame(load_csv(val_csv), config)

    target = config["dataset"]["target_column"]

    # Also drop any remaining non-feature columns that slipped through
    non_feature_cols = [target]
    for col in train_frame.columns:
        if train_frame[col].dtype == object:
            non_feature_cols.append(col)

    return DatasetBundle(
        train_features=train_frame.drop(columns=non_feature_cols, errors="ignore"),
        train_target=train_frame[target],
        val_features=val_frame.drop(columns=non_feature_cols, errors="ignore"),
        val_target=val_frame[target],
    )
