from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gemspot_training.data import load_csv, prepare_frame


TEXT_DERIVED_COLUMNS = [
    "combined_text_char_len",
    "combined_text_word_count",
]


@dataclass
class XGBoostFrameBundle:
    train_features: pd.DataFrame
    train_target: pd.Series
    val_features: pd.DataFrame
    val_target: pd.Series


def _add_text_summary_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    text = enriched["combined_text"].fillna("").astype(str)
    enriched["combined_text_char_len"] = text.str.len().astype(float)
    enriched["combined_text_word_count"] = text.str.split().map(len).astype(float)
    return enriched


def make_xgboost_frame_bundle(train_csv: str | Path, val_csv: str | Path, config: dict) -> XGBoostFrameBundle:
    dataset_cfg = config["dataset"]
    target_column = dataset_cfg["target_column"]
    numeric_features = list(dataset_cfg["numeric_features"])
    categorical_features = list(dataset_cfg["categorical_features"])

    train_frame = _add_text_summary_features(prepare_frame(load_csv(train_csv), config))
    val_frame = _add_text_summary_features(prepare_frame(load_csv(val_csv), config))

    model_features = numeric_features + categorical_features + TEXT_DERIVED_COLUMNS

    combined = pd.concat(
        [train_frame[model_features], val_frame[model_features]],
        axis=0,
        ignore_index=True,
    )
    combined = pd.get_dummies(combined, columns=categorical_features, dtype=float)

    split_at = len(train_frame)
    train_features = combined.iloc[:split_at].reset_index(drop=True)
    val_features = combined.iloc[split_at:].reset_index(drop=True)

    return XGBoostFrameBundle(
        train_features=train_features,
        train_target=train_frame[target_column].astype(int).reset_index(drop=True),
        val_features=val_features,
        val_target=val_frame[target_column].astype(int).reset_index(drop=True),
    )


def make_xgboost_training_frames(
    train_csv: str | Path,
    val_csv: str | Path,
    config: dict,
    label_column: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = make_xgboost_frame_bundle(train_csv, val_csv, config)

    train_frame = bundle.train_features.copy()
    train_frame[label_column] = bundle.train_target.values

    val_frame = bundle.val_features.copy()
    val_frame[label_column] = bundle.val_target.values

    return train_frame, val_frame
