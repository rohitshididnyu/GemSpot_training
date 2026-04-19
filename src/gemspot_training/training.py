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


# ---------------------------------------------------------------------------
# Quality gates for first-time training
# ---------------------------------------------------------------------------
#
# A candidate is "registered" (saved to artifacts/models/<name>.joblib and
# logged via mlflow.sklearn.log_model) ONLY if it passes every gate below.
# Otherwise it is written to artifacts/models/_rejected/ with a JSON sidecar
# explaining the failure, and NOT logged as a model artifact.
#
# The philosophy:
#   - Hard minimums catch pathologies (near-random rankers, broken
#     preprocessing that produced constant predictions, schema flips).
#   - `beat_baseline` guarantees every served model strictly improves on
#     the dumb `DummyClassifier` reference — a model that can't beat
#     "always predict majority class" must not be deployed.
#   - `exempt` lets the baseline itself skip the gates (it IS the
#     reference, not a product).

def evaluate_quality_gates(
    candidate_name: str,
    candidate_metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None,
    gates_cfg: dict,
) -> dict:
    """Return a gate decision dict.

    Result shape:
        {
          "passed":   bool,       # overall decision
          "exempt":   bool,       # True if candidate is on the exempt list
          "failures": list[str],  # human-readable reasons (empty if passed)
          "baseline_comparison": dict | None,
        }
    """
    exempt = gates_cfg.get("exempt", []) or []
    if candidate_name in exempt:
        return {
            "passed": True,
            "exempt": True,
            "failures": [],
            "baseline_comparison": None,
        }

    failures: list[str] = []

    # --- Gate 1: hard minimum thresholds -----------------------------------
    hard_minimums = gates_cfg.get("hard_minimums", {}) or {}
    for metric_name, threshold in hard_minimums.items():
        value = candidate_metrics.get(metric_name)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            failures.append(f"{metric_name} is missing or NaN (threshold {threshold})")
        elif value < threshold:
            failures.append(
                f"{metric_name}={value:.4f} below hard minimum {threshold}"
            )

    # --- Gate 2: must beat the dumb baseline -------------------------------
    baseline_comparison = None
    beat_cfg = gates_cfg.get("beat_baseline", {}) or {}
    if beat_cfg and baseline_metrics:
        metric_name = beat_cfg.get("metric", "roc_auc")
        min_delta = float(beat_cfg.get("min_delta", 0.0))
        cand_score = candidate_metrics.get(metric_name)
        base_score = baseline_metrics.get(metric_name)
        if (
            cand_score is not None
            and base_score is not None
            and not (isinstance(cand_score, float) and np.isnan(cand_score))
            and not (isinstance(base_score, float) and np.isnan(base_score))
        ):
            delta = cand_score - base_score
            baseline_comparison = {
                "metric": metric_name,
                "candidate": cand_score,
                "baseline": base_score,
                "delta": delta,
                "min_delta": min_delta,
            }
            if delta < min_delta:
                failures.append(
                    f"{metric_name} delta vs baseline {delta:+.4f} < "
                    f"required min_delta={min_delta}"
                )
    elif beat_cfg and not baseline_metrics:
        failures.append(
            "beat_baseline gate is configured but no baseline metrics were "
            "captured — train the baseline candidate before other candidates."
        )

    return {
        "passed": len(failures) == 0,
        "exempt": False,
        "failures": failures,
        "baseline_comparison": baseline_comparison,
    }
