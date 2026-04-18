"""Incremental retraining script for GemSpot models.

Workflow:
  1. Load the previously-trained model from artifacts/models/<name>.joblib
  2. Evaluate it on the validation set → baseline_metrics
  3. Continue training on NEW data (initial_training_set_new.csv by default)
     - For XGBoost: uses xgb_model=old_booster to resume boosting
     - For sklearn models without incremental support: refits on NEW+OLD data
  4. Evaluate the retrained model on the same validation set → new_metrics
  5. Compare ROC-AUC (primary metric) between old and new
  6. SAVE the new model ONLY if it improved; otherwise keep the old model
  7. Log everything to MLflow as a separate "retrain" experiment

Usage:
    python3 src/retrain.py \\
        --config configs/candidates.yaml \\
        --candidate xgboost_v2 \\
        --new-data-csv data/demo/initial_training_set_new.csv \\
        --val-csv data/demo/gemspot_val.csv \\
        --old-model artifacts/models/xgboost_v2.joblib \\
        --artifact-dir artifacts/models \\
        --improvement-threshold 0.001
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from gemspot_training.data import load_csv, prepare_frame
from gemspot_training.training import build_pipeline, compute_binary_metrics
from gemspot_training.utils import (
    collect_environment_info,
    ensure_dir,
    flatten_dict,
    get_git_sha,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally retrain a GemSpot model on new data.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--candidate",
        required=True,
        help="Name of candidate from the config (e.g., xgboost_v2) to retrain.",
    )
    parser.add_argument(
        "--new-data-csv",
        required=True,
        help="Path to the NEW training CSV (e.g. initial_training_set_new.csv).",
    )
    parser.add_argument(
        "--val-csv",
        required=True,
        help="Validation CSV for evaluating old vs new model (same used during training).",
    )
    parser.add_argument(
        "--old-model",
        required=True,
        help="Path to previously-saved .joblib model (e.g. artifacts/models/xgboost_v2.joblib).",
    )
    parser.add_argument(
        "--old-train-csv",
        default=None,
        help="Optional: original training CSV. Used to refit preprocessor "
             "(and for non-XGBoost models) on the combined OLD+NEW data.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/models",
        help="Directory to save the retrained model.",
    )
    parser.add_argument(
        "--backup-dir",
        default="artifacts/models/_backup",
        help="Directory to back up the replaced model (if improvement wins).",
    )
    parser.add_argument(
        "--rejected-dir",
        default="artifacts/models/_rejected",
        help="Directory to store rejected models (when retrain fails to improve).",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.001,
        help="Minimum ROC-AUC improvement to keep the new model (default: 0.001 = 0.1%%).",
    )
    parser.add_argument(
        "--primary-metric",
        default="roc_auc",
        choices=["roc_auc", "f1", "accuracy", "average_precision"],
        help="Metric to compare old vs new model on.",
    )
    parser.add_argument(
        "--additional-rounds",
        type=int,
        default=50,
        help="Extra boosting rounds for XGBoost incremental training (default: 50).",
    )
    parser.add_argument(
        "--experiment-name",
        default="GemSpot-Retrain",
        help="MLflow experiment name for retraining runs.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI override.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_candidate(config: dict, name: str) -> dict:
    for cand in config["candidates"]:
        if cand["name"] == name:
            return cand
    raise ValueError(f"Candidate '{name}' not found in config.")


def safe_metric(value: float) -> float | None:
    if isinstance(value, float) and np.isnan(value):
        return None
    return float(value)


def evaluate_pipeline(pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Run pipeline on validation set and return binary metrics."""
    preds = pipeline.predict(X_val)
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(X_val)[:, 1]
    else:
        scores = preds.astype(float)
    return compute_binary_metrics(y_val, preds, scores)


def align_features(features: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Ensure `features` has exactly `expected_columns` in the right order.

    Missing columns are filled with 0.0; extra columns are dropped. This is
    critical when retraining on NEW data whose schema may drift slightly
    (e.g. _new.csv has `price` but not `category`).
    """
    aligned = features.copy()
    for col in expected_columns:
        if col not in aligned.columns:
            aligned[col] = 0.0
    extra = [c for c in aligned.columns if c not in expected_columns]
    if extra:
        aligned = aligned.drop(columns=extra)
    return aligned[expected_columns]


def get_expected_columns_from_pipeline(pipeline: Pipeline) -> list[str] | None:
    """Extract the feature column order the old pipeline was trained on."""
    try:
        pre = pipeline.named_steps.get("preprocessor")
        if pre is not None and hasattr(pre, "feature_names_in_"):
            return list(pre.feature_names_in_)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Incremental training strategies
# ---------------------------------------------------------------------------

def _is_xgboost_pipeline(pipeline: Pipeline) -> bool:
    from xgboost import XGBClassifier
    return isinstance(pipeline.named_steps.get("model"), XGBClassifier)


def incremental_retrain_xgboost(
    old_pipeline: Pipeline,
    X_new: pd.DataFrame,
    y_new: pd.Series,
    additional_rounds: int,
) -> Pipeline:
    """Continue XGBoost boosting from the old model's trees on NEW data.

    Uses xgb_model parameter to `.fit()` which resumes from the existing booster.
    The preprocessor is REUSED (not refit) — this preserves the scale/impute
    statistics learned on the original training data, matching what the old
    booster expects.
    """
    from xgboost import XGBClassifier

    preprocessor = old_pipeline.named_steps["preprocessor"]
    old_model: XGBClassifier = old_pipeline.named_steps["model"]

    # Transform with the OLD preprocessor (do NOT refit — that would shift scales).
    X_new_transformed = preprocessor.transform(X_new)

    # Build a new XGBClassifier with additional boosting rounds.
    old_params = old_model.get_params()
    total_rounds = (old_params.get("n_estimators") or 100) + additional_rounds
    new_params = {**old_params, "n_estimators": total_rounds}
    new_model = XGBClassifier(**new_params)

    # Continue boosting from the old model's trees.
    new_model.fit(X_new_transformed, y_new, xgb_model=old_model.get_booster())

    # Reassemble the pipeline with the SAME preprocessor + new booster.
    new_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", new_model),
    ])
    return new_pipeline


def full_retrain_combined(
    candidate_cfg: dict,
    feature_columns: list[str],
    X_combined: pd.DataFrame,
    y_combined: pd.Series,
) -> Pipeline:
    """Fallback: build a fresh pipeline and fit on OLD+NEW combined data.

    Used for non-XGBoost candidates (dummy, hist_gradient_boosting) which
    don't support warm-starting from a saved model.
    """
    pipeline = build_pipeline(candidate_cfg, feature_columns)
    pipeline.fit(X_combined, y_combined)
    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    candidate_cfg = find_candidate(config, args.candidate)

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    artifact_dir = ensure_dir(args.artifact_dir)
    backup_dir = ensure_dir(args.backup_dir)
    rejected_dir = ensure_dir(args.rejected_dir)

    # ---- Load old model ----
    old_model_path = Path(args.old_model)
    if not old_model_path.exists():
        raise FileNotFoundError(f"Old model not found at {old_model_path}. "
                                f"Run `src/train.py` first to create it.")
    print(f"Loading old model from {old_model_path}...", flush=True)
    old_pipeline: Pipeline = joblib.load(old_model_path)

    # ---- Load validation set ----
    print(f"Loading validation CSV: {args.val_csv}", flush=True)
    val_frame = prepare_frame(load_csv(args.val_csv), config)
    target = config["dataset"]["target_column"]
    val_target = val_frame[target].astype(int)
    val_features_raw = val_frame.drop(columns=[target], errors="ignore")
    # Drop any object cols leftover
    obj_cols = [c for c in val_features_raw.columns if val_features_raw[c].dtype == object]
    val_features_raw = val_features_raw.drop(columns=obj_cols, errors="ignore")

    # ---- Load new training data ----
    print(f"Loading NEW training CSV: {args.new_data_csv}", flush=True)
    new_frame = prepare_frame(load_csv(args.new_data_csv), config)
    new_target = new_frame[target].astype(int)
    new_features_raw = new_frame.drop(columns=[target], errors="ignore")
    obj_cols = [c for c in new_features_raw.columns if new_features_raw[c].dtype == object]
    new_features_raw = new_features_raw.drop(columns=obj_cols, errors="ignore")

    # ---- Align schemas to old pipeline's feature_names_in_ ----
    expected = get_expected_columns_from_pipeline(old_pipeline)
    if expected is None:
        raise RuntimeError(
            "Could not extract feature column order from old pipeline. "
            "The pipeline must have a ColumnTransformer/preprocessor with "
            "feature_names_in_ attribute."
        )
    print(f"Aligning features to old model's {len(expected)} columns...", flush=True)
    val_features = align_features(val_features_raw, expected)
    new_features = align_features(new_features_raw, expected)

    # ---- Evaluate OLD model ----
    print("\n=== Evaluating OLD model on validation set ===", flush=True)
    old_metrics = evaluate_pipeline(old_pipeline, val_features, val_target)
    print(json.dumps({k: safe_metric(v) for k, v in old_metrics.items()}, indent=2))

    # ---- Incremental retrain ----
    print(f"\n=== Retraining {args.candidate} on NEW data ({len(new_features):,} rows) ===", flush=True)
    retrain_start = time.perf_counter()

    if _is_xgboost_pipeline(old_pipeline):
        print(f"Using incremental XGBoost continuation "
              f"(+{args.additional_rounds} boosting rounds).", flush=True)
        new_pipeline = incremental_retrain_xgboost(
            old_pipeline=old_pipeline,
            X_new=new_features,
            y_new=new_target,
            additional_rounds=args.additional_rounds,
        )
    else:
        print("Non-XGBoost candidate → refitting fresh on OLD+NEW combined data.", flush=True)
        if args.old_train_csv is None:
            raise ValueError(
                "Non-XGBoost models need --old-train-csv to combine with NEW data "
                "for a fair full refit."
            )
        old_frame = prepare_frame(load_csv(args.old_train_csv), config)
        old_target = old_frame[target].astype(int)
        old_features_raw = old_frame.drop(columns=[target], errors="ignore")
        obj_cols = [c for c in old_features_raw.columns if old_features_raw[c].dtype == object]
        old_features_raw = old_features_raw.drop(columns=obj_cols, errors="ignore")
        old_features = align_features(old_features_raw, expected)

        X_combined = pd.concat([old_features, new_features], ignore_index=True)
        y_combined = pd.concat([old_target, new_target], ignore_index=True)
        new_pipeline = full_retrain_combined(
            candidate_cfg=candidate_cfg,
            feature_columns=expected,
            X_combined=X_combined,
            y_combined=y_combined,
        )

    retrain_seconds = time.perf_counter() - retrain_start
    print(f"Retrain took {retrain_seconds:.1f}s", flush=True)

    # ---- Evaluate NEW model ----
    print("\n=== Evaluating RETRAINED model on validation set ===", flush=True)
    new_metrics = evaluate_pipeline(new_pipeline, val_features, val_target)
    print(json.dumps({k: safe_metric(v) for k, v in new_metrics.items()}, indent=2))

    # ---- Decide: keep or reject ----
    metric = args.primary_metric
    old_score = old_metrics.get(metric, float("nan"))
    new_score = new_metrics.get(metric, float("nan"))
    delta = new_score - old_score
    improved = delta >= args.improvement_threshold

    print("\n" + "=" * 60)
    print(f"COMPARISON on {metric}:")
    print(f"  OLD model:       {old_score:.6f}")
    print(f"  RETRAINED model: {new_score:.6f}")
    print(f"  Delta:           {delta:+.6f}   (threshold: +{args.improvement_threshold})")
    print(f"  Decision:        {'KEEP new model ✅' if improved else 'REJECT (keep old) ❌'}")
    print("=" * 60 + "\n")

    # ---- MLflow logging ----
    run_prefix = config.get("tracking", {}).get("run_name_prefix", "gemspot")
    run_name = f"{run_prefix}-retrain-{args.candidate}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "project": "GemSpot",
            "task": "retrain",
            "candidate_name": args.candidate,
            "candidate_kind": candidate_cfg["kind"],
            "decision": "keep" if improved else "reject",
            "code_version": get_git_sha(),
        })
        mlflow.log_params(flatten_dict({"candidate": candidate_cfg}))
        mlflow.log_param("retrain.new_data_csv", args.new_data_csv)
        mlflow.log_param("retrain.old_model_path", str(old_model_path))
        mlflow.log_param("retrain.additional_rounds", args.additional_rounds)
        mlflow.log_param("retrain.improvement_threshold", args.improvement_threshold)
        mlflow.log_param("retrain.primary_metric", metric)
        mlflow.log_param("retrain.new_rows", len(new_features))
        mlflow.log_param("retrain.val_rows", len(val_features))
        for key, value in collect_environment_info().items():
            mlflow.log_param(f"environment.{key}", value)

        # Log metrics with prefixes to distinguish
        for k, v in old_metrics.items():
            sv = safe_metric(v)
            if sv is not None:
                mlflow.log_metric(f"old.{k}", sv)
        for k, v in new_metrics.items():
            sv = safe_metric(v)
            if sv is not None:
                mlflow.log_metric(f"new.{k}", sv)
        mlflow.log_metric("delta.primary_metric", float(delta))
        mlflow.log_metric("retrain.seconds", retrain_seconds)
        mlflow.log_metric("decision.improved", 1.0 if improved else 0.0)

        # ---- Save or reject ----
        decision_summary = {
            "candidate": args.candidate,
            "primary_metric": metric,
            "old_score": safe_metric(old_score),
            "new_score": safe_metric(new_score),
            "delta": safe_metric(delta),
            "threshold": args.improvement_threshold,
            "decision": "keep" if improved else "reject",
            "retrain_seconds": retrain_seconds,
            "new_data_rows": len(new_features),
        }
        mlflow.log_text(json.dumps(decision_summary, indent=2), "retrain_decision.json")

        if improved:
            # Back up old model → replace with new
            backup_path = backup_dir / f"{args.candidate}_prev_{int(time.time())}.joblib"
            shutil.copy2(old_model_path, backup_path)
            print(f"Backed up old model to: {backup_path}", flush=True)

            new_path = artifact_dir / f"{args.candidate}.joblib"
            joblib.dump(new_pipeline, new_path)
            print(f"Saved new model to: {new_path}", flush=True)
            mlflow.log_artifact(str(new_path), artifact_path="exported_models")
            mlflow.sklearn.log_model(new_pipeline, artifact_path="model")
        else:
            print("New model rejected — old model preserved on disk.", flush=True)
            # Save the rejected candidate to the dedicated rejected folder
            # (separate from artifacts/models/ so it doesn't get picked up
            # accidentally on the next retrain, but preserved for inspection).
            ts = int(time.time())
            rejected_path = rejected_dir / f"{args.candidate}_rejected_{ts}.joblib"
            joblib.dump(new_pipeline, rejected_path)
            print(f"Saved rejected model to: {rejected_path}", flush=True)

            # Also save a sidecar JSON with the reason for rejection
            reject_info_path = rejected_dir / f"{args.candidate}_rejected_{ts}.json"
            with open(reject_info_path, "w", encoding="utf-8") as handle:
                json.dump({
                    **decision_summary,
                    "timestamp": ts,
                    "new_data_csv": args.new_data_csv,
                    "val_csv": args.val_csv,
                    "old_model_path": str(old_model_path),
                    "reason": (
                        f"ROC-AUC did not improve by the required "
                        f"{args.improvement_threshold} threshold "
                        f"(delta={delta:+.6f})."
                    ),
                }, handle, indent=2)
            print(f"Saved rejection reason to: {reject_info_path}", flush=True)

            mlflow.log_artifact(str(rejected_path), artifact_path="rejected_models")
            mlflow.log_artifact(str(reject_info_path), artifact_path="rejected_models")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
