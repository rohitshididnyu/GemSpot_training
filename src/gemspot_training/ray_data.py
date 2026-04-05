"""Ray-compatible data helpers for GemSpot.

Thin wrapper around gemspot_training.data that provides the XGBoostFrameBundle
and label-appended DataFrames needed by train_ray_tune.py and
train_ray_xgboost.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gemspot_training.data import make_dataset_bundle


@dataclass
class XGBoostFrameBundle:
    train_features: pd.DataFrame
    train_target: pd.Series
    val_features: pd.DataFrame
    val_target: pd.Series


def make_xgboost_frame_bundle(
    train_csv: str | Path,
    val_csv: str | Path,
    config: dict,
) -> XGBoostFrameBundle:
    """Load CSVs and return preprocessed feature/target arrays."""
    bundle = make_dataset_bundle(train_csv, val_csv, config)
    return XGBoostFrameBundle(
        train_features=bundle.train_features,
        train_target=bundle.train_target,
        val_features=bundle.val_features,
        val_target=bundle.val_target,
    )


def make_xgboost_training_frames(
    train_csv: str | Path,
    val_csv: str | Path,
    config: dict,
    label_column: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df) with label column appended (for Ray Train)."""
    bundle = make_xgboost_frame_bundle(train_csv, val_csv, config)

    train_frame = bundle.train_features.copy()
    train_frame[label_column] = bundle.train_target.values

    val_frame = bundle.val_features.copy()
    val_frame[label_column] = bundle.val_target.values

    return train_frame, val_frame
