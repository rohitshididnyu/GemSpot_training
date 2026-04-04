from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml

from gemspot_training.data import make_dataset_bundle
from gemspot_training.training import build_pipeline, compute_binary_metrics
from gemspot_training.utils import (
    collect_environment_info,
    ensure_dir,
    flatten_dict,
    get_command_output,
    get_git_sha,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GemSpot recommendation models.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--train-csv", required=True, help="Path to training CSV.")
    parser.add_argument("--val-csv", required=True, help="Path to validation CSV.")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional MLflow experiment name override.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI override. If omitted, MLflow uses the environment.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/models",
        help="Directory for exported model artifacts.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def safe_metric(value: float) -> float | None:
    if isinstance(value, float) and np.isnan(value):
        return None
    return float(value)


def train_candidate(candidate_cfg: dict, config: dict, dataset_bundle, artifact_dir: Path) -> None:
    tracking_cfg = config.get("tracking", {})
    run_prefix = tracking_cfg.get("run_name_prefix", "gemspot")

    feature_columns = list(dataset_bundle.train_features.columns)
    pipeline = build_pipeline(candidate_cfg, feature_columns)

    run_name = f"{run_prefix}-{candidate_cfg['name']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "project": "GemSpot",
                "task": "will_visit_classification",
                "candidate_name": candidate_cfg["name"],
                "candidate_kind": candidate_cfg["kind"],
                "code_version": get_git_sha(),
            }
        )

        mlflow.log_params(flatten_dict({"candidate": candidate_cfg}))
        mlflow.log_param("dataset.target_column", config["dataset"]["target_column"])
        mlflow.log_param("dataset.num_features", len(feature_columns))

        for key, value in collect_environment_info().items():
            mlflow.log_param(f"environment.{key}", value)

        nvidia_smi_output = get_command_output(["nvidia-smi"])
        if nvidia_smi_output:
            mlflow.log_text(nvidia_smi_output, "environment/nvidia_smi.txt")

        train_start = time.perf_counter()
        pipeline.fit(dataset_bundle.train_features, dataset_bundle.train_target)
        train_seconds = time.perf_counter() - train_start

        predictions = pipeline.predict(dataset_bundle.val_features)
        if hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(dataset_bundle.val_features)[:, 1]
        else:
            scores = predictions.astype(float)

        metrics = compute_binary_metrics(dataset_bundle.val_target, predictions, scores)
        metrics["train_seconds"] = train_seconds
        metrics["train_rows"] = float(len(dataset_bundle.train_features))
        metrics["val_rows"] = float(len(dataset_bundle.val_features))
        metrics["rows_per_second"] = float(len(dataset_bundle.train_features) / max(train_seconds, 1e-9))

        clean_metrics = {
            key: value for key, value in ((name, safe_metric(metric)) for name, metric in metrics.items()) if value is not None
        }
        mlflow.log_metrics(clean_metrics)

        artifact_path = artifact_dir / f"{candidate_cfg['name']}.joblib"
        joblib.dump(pipeline, artifact_path)
        mlflow.log_artifact(str(artifact_path), artifact_path="exported_models")
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        summary = {
            "candidate": candidate_cfg["name"],
            "kind": candidate_cfg["kind"],
            "notes": candidate_cfg.get("notes", ""),
            "metrics": clean_metrics,
            "artifact_path": str(artifact_path),
        }
        mlflow.log_text(json.dumps(summary, indent=2), "run_summary.json")

        print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    experiment_name = args.experiment_name or config.get("experiment_name", "GemSpot-WillVisit")
    mlflow.set_experiment(experiment_name)

    os.environ.setdefault("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "1")
    os.environ.setdefault("MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING", "1")
    mlflow.enable_system_metrics_logging()

    artifact_dir = ensure_dir(args.artifact_dir)
    dataset_bundle = make_dataset_bundle(args.train_csv, args.val_csv, config)

    for candidate_cfg in config["candidates"]:
        train_candidate(candidate_cfg, config, dataset_bundle, artifact_dir)


if __name__ == "__main__":
    main()
