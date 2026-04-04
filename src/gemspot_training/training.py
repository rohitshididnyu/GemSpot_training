"""Model building, preprocessing pipeline, and metric computation for GemSpot.

After data.py explodes the list-encoded columns, every feature is numeric.
The preprocessor is therefore a simple impute -> scale pipeline applied to
all columns uniformly.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor(feature_columns: list[str]) -> Pipeline:
    """Build a simple numeric preprocessor: impute missing -> scale."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_pipeline(candidate_cfg: dict, feature_columns: list[str]) -> Pipeline:
    """Build preprocessor + estimator pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_columns)),
            ("model", build_estimator(candidate_cfg["kind"], candidate_cfg.get("params", {}))),
        ]
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_binary_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
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
