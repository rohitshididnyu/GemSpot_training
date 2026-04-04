from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import ray
import ray.data
import xgboost
import yaml
from ray import train
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.xgboost import RayTrainReportCallback, XGBoostTrainer

from gemspot_training.ray_data import make_xgboost_frame_bundle, make_xgboost_training_frames
from gemspot_training.training import compute_binary_metrics
from gemspot_training.utils import ensure_dir, flatten_dict, get_git_sha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GemSpot bonus training with Ray Train + XGBoost.")
    parser.add_argument("--config", default="configs/ray_bonus.yaml", help="Path to Ray bonus YAML config.")
    parser.add_argument("--train-csv", required=True, help="Training CSV path.")
    parser.add_argument("--val-csv", required=True, help="Validation CSV path.")
    parser.add_argument("--ray-address", default=None, help="Optional Ray address. Defaults to RAY_ADDRESS or auto.")
    parser.add_argument("--storage-path", default=None, help="Optional Ray storage path override.")
    parser.add_argument("--tracking-uri", default=None, help="Optional MLflow tracking URI override.")
    parser.add_argument("--experiment-name", default=None, help="Optional MLflow experiment override.")
    parser.add_argument("--run-name", default="GemSpot-RayTrain-XGBoost", help="Logical run name.")
    parser.add_argument("--artifact-dir", default="artifacts/ray", help="Artifact export directory.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override worker count.")
    parser.add_argument("--cpu-per-worker", type=float, default=None, help="Override CPUs per worker.")
    parser.add_argument("--gpu-per-worker", type=float, default=None, help="Override GPUs per worker.")
    parser.add_argument("--use-gpu", action="store_true", help="Request GPUs for Ray workers.")
    parser.add_argument("--max-failures", type=int, default=None, help="Override FailureConfig max_failures.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_loop_per_worker(loop_config: dict) -> None:
    label_column = loop_config["label_column"]
    params = dict(loop_config["xgboost_params"])
    checkpoint_frequency = int(loop_config["checkpoint_frequency"])
    total_rounds = int(loop_config["num_boost_round"])

    checkpoint = train.get_checkpoint()
    starting_model = None
    completed_rounds = 0
    if checkpoint:
        starting_model = RayTrainReportCallback.get_model(checkpoint)
        try:
            completed_rounds = int(starting_model.num_boosted_rounds())
        except Exception:
            completed_rounds = 0

    train_frame = train.get_dataset_shard("train").materialize().to_pandas()
    val_frame = train.get_dataset_shard("validation").materialize().to_pandas()

    train_x = train_frame.drop(columns=[label_column])
    train_y = train_frame[label_column]
    val_x = val_frame.drop(columns=[label_column])
    val_y = val_frame[label_column]

    dtrain = xgboost.DMatrix(train_x, label=train_y)
    dval = xgboost.DMatrix(val_x, label=val_y)

    remaining_rounds = max(1, total_rounds - completed_rounds)

    xgboost.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dval, "validation")],
        num_boost_round=remaining_rounds,
        xgb_model=starting_model,
        verbose_eval=False,
        callbacks=[
            RayTrainReportCallback(
                metrics={
                    "train_logloss": "train-logloss",
                    "validation_logloss": "validation-logloss",
                    "validation_auc": "validation-auc",
                    "validation_aucpr": "validation-aucpr",
                },
                frequency=checkpoint_frequency,
                checkpoint_at_end=True,
            )
        ],
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    ray_address = args.ray_address or None
    if ray_address is None:
        ray_address = "auto"

    ray.init(address=ray_address, ignore_reinit_error=True)

    bundle = make_xgboost_frame_bundle(args.train_csv, args.val_csv, config)
    train_frame, val_frame = make_xgboost_training_frames(args.train_csv, args.val_csv, config, label_column="label")

    ray_train_cfg = dict(config["ray_train"])
    num_workers = args.num_workers if args.num_workers is not None else int(ray_train_cfg["num_workers"])
    cpu_per_worker = args.cpu_per_worker if args.cpu_per_worker is not None else float(ray_train_cfg["cpu_per_worker"])
    gpu_per_worker = args.gpu_per_worker if args.gpu_per_worker is not None else float(ray_train_cfg["gpu_per_worker"])
    use_gpu = bool(args.use_gpu or ray_train_cfg.get("use_gpu", False))
    max_failures = args.max_failures if args.max_failures is not None else int(ray_train_cfg["max_failures"])
    storage_path = args.storage_path or ray_train_cfg["storage_path"]

    trainer = XGBoostTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "label_column": "label",
            "num_boost_round": int(config["base_params"]["num_boost_round"]),
            "checkpoint_frequency": int(ray_train_cfg["checkpoint_frequency"]),
            "xgboost_params": {
                key: value
                for key, value in config["base_params"].items()
                if key != "num_boost_round"
            },
        },
        datasets={
            "train": ray.data.from_pandas(train_frame),
            "validation": ray.data.from_pandas(val_frame),
        },
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            resources_per_worker={
                "CPU": cpu_per_worker,
                "GPU": gpu_per_worker,
            }
            if use_gpu
            else {"CPU": cpu_per_worker},
        ),
        run_config=RunConfig(
            name=args.run_name,
            storage_path=storage_path,
            failure_config=FailureConfig(max_failures=max_failures),
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=int(ray_train_cfg["checkpoint_frequency"]),
                checkpoint_at_end=True,
                num_to_keep=2,
                checkpoint_score_attribute="validation_aucpr",
                checkpoint_score_order="max",
            ),
        ),
    )

    result = trainer.fit()
    booster = RayTrainReportCallback.get_model(result.checkpoint)

    dval = xgboost.DMatrix(bundle.val_features, label=bundle.val_target)
    scores = booster.predict(dval)
    predictions = (scores >= 0.5).astype(int)
    final_metrics = compute_binary_metrics(bundle.val_target, predictions, scores)

    artifact_dir = ensure_dir(args.artifact_dir)
    model_path = artifact_dir / "gemspot_ray_train_model.ubj"
    booster.save_model(model_path)

    summary = {
        "run_name": args.run_name,
        "code_version": get_git_sha(),
        "ray_result_metrics": result.metrics,
        "validation_metrics": final_metrics,
        "storage_path": storage_path,
        "checkpoint_path": result.checkpoint.path if result.checkpoint else "",
        "artifact_path": str(model_path),
    }

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    if args.tracking_uri or args.experiment_name:
        experiment_name = args.experiment_name or config.get("ray_experiment_name", "GemSpot-WillVisit-RayTrain")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=args.run_name, log_system_metrics=True):
            mlflow.set_tags(
                {
                    "project": "GemSpot",
                    "workflow": "ray-train-xgboost",
                    "code_version": get_git_sha(),
                }
            )
            mlflow.log_params(flatten_dict({"base_params": config["base_params"]}))
            mlflow.log_params(
                {
                    "ray.num_workers": num_workers,
                    "ray.cpu_per_worker": cpu_per_worker,
                    "ray.gpu_per_worker": gpu_per_worker,
                    "ray.use_gpu": use_gpu,
                    "ray.storage_path": storage_path,
                    "ray.max_failures": max_failures,
                }
            )
            mlflow.log_metrics({key: float(value) for key, value in final_metrics.items()})
            mlflow.log_artifact(str(model_path), artifact_path="exported_models")
            mlflow.log_text(json.dumps(summary, indent=2), "ray_train_summary.json")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
