from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class DatasetBundle:
    train_features: pd.DataFrame
    train_target: pd.Series
    val_features: pd.DataFrame
    val_target: pd.Series


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _stringify(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _combine_text_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return pd.Series([""] * len(frame), index=frame.index, name="combined_text")
    return frame[existing].apply(
        lambda row: " ".join(_stringify(value) for value in row if _stringify(value)),
        axis=1,
    )


def validate_schema(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def prepare_frame(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    target_column = dataset_cfg["target_column"]
    numeric_features = dataset_cfg["numeric_features"]
    categorical_features = dataset_cfg["categorical_features"]
    text_features = dataset_cfg["text_features"]

    required_columns = [target_column, *numeric_features, *categorical_features, *text_features]
    validate_schema(frame, required_columns)

    prepared = frame.copy()
    prepared["combined_text"] = _combine_text_columns(prepared, text_features)
    prepared[target_column] = prepared[target_column].astype(int)
    return prepared


def make_dataset_bundle(train_csv: str | Path, val_csv: str | Path, config: dict) -> DatasetBundle:
    train_frame = prepare_frame(load_csv(train_csv), config)
    val_frame = prepare_frame(load_csv(val_csv), config)

    target_column = config["dataset"]["target_column"]

    return DatasetBundle(
        train_features=train_frame.drop(columns=[target_column]),
        train_target=train_frame[target_column],
        val_features=val_frame.drop(columns=[target_column]),
        val_target=val_frame[target_column],
    )
