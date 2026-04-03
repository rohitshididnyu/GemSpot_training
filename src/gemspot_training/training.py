from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _text_to_series(values: pd.DataFrame | np.ndarray | list[str]) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0].fillna("").astype(str)
    if isinstance(values, np.ndarray):
        flattened = values.reshape(-1)
        return pd.Series(flattened).fillna("").astype(str)
    return pd.Series(values).fillna("").astype(str)


def build_preprocessor(dataset_cfg: dict) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    text_pipeline = Pipeline(
        steps=[
            ("to_series", FunctionTransformer(_text_to_series, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=250, ngram_range=(1, 2))),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, dataset_cfg["numeric_features"]),
            ("categorical", categorical_pipeline, dataset_cfg["categorical_features"]),
            ("text", text_pipeline, ["combined_text"]),
        ],
        sparse_threshold=0.0,
    )


def build_estimator(kind: str, params: dict[str, Any]) -> Any:
    if kind == "dummy":
        return DummyClassifier(**params)
    if kind == "logistic_regression":
        return LogisticRegression(**params)
    if kind == "random_forest":
        return RandomForestClassifier(**params)
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params)
    if kind == "xgboost":
        return XGBClassifier(**params)
    raise ValueError(f"Unsupported model kind: {kind}")


def build_pipeline(candidate_cfg: dict, dataset_cfg: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(dataset_cfg)),
            ("model", build_estimator(candidate_cfg["kind"], candidate_cfg.get("params", {}))),
        ]
    )


def compute_binary_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    unique_labels = set(pd.Series(y_true).astype(int).unique().tolist())
    if len(unique_labels) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    return metrics
