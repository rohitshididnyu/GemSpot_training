"""Ray Train distributed XGBoost training for GemSpot.

Demonstrates fault tolerance (FailureConfig) and checkpointing.
"""
from __future__ import annotations

import argparse
import json

import mlflow
import ray
import ray.data
import xgboost
import yaml
from ray import train
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.xgboost import RayTrainReportCallback, XGBoostTrainer

from gemspot_training.ray_data import make_xgboost_frame_bundle
from gemspot_training.training import compute_binary_metrics
from gemspot_training.utils import ensure_dir, flatten_dict, get_git_sha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GemSpot bonus: Ray Train + XGBoost.")
    parser.add_argument("--config", default="configs/ray_bonus.yaml")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--storage-path", default=None)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default="GemSpot-RayTrain-XGBoost")
    parser.add_argument("--artifact-dir", default="artifacts/ray")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cpu-per-worker", type=float, default=None)
    parser.add_argument("--gpu-per-worker", type=float, default=None)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--max-failures", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_loop_per_worker(loop_config: dict) -> None:
    """Training loop executed by each Ray worker."""
    label_column = loop_config["label_column"]
    params = dict(loop_config["xgboost_params"])
    checkpoint_freq = int(loop_config["checkpoint_frequency"])
    total_rounds = int(loop_config["num_boost_round"])

    # Resume from checkpoint
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

    remaining = max(1, total_rounds - completed_rounds)

    xgboost.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dval, "validation")],
        num_boost_round=remaining,
        xgb_model=starting_model,
        verbose_eval=25,
        callbacks=[
            RayTrainReportCallback(
                metrics={
                    "train_logloss": "train-logloss",
                    "validation_logloss": "validation-logloss",
                },
                frequency=checkpoint_freq,
                checkpoint_at_end=True,
            )
        ],
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    ray.init(address=args.ray_address or "auto", ignore_reinit_error=True)

    # Prepare data
    bundle = make_xgboost_frame_bundle(args.train_csv, args.val_csv, config)
    train_frame = bundle.train_features.copy()
    train_frame["label"] = bundle.train_target.values
    val_frame = bundle.val_features.copy()
    val_frame["label"] = bundle.val_target.values

    ray_cfg = config["ray_train"]
    num_workers = args.num_workers or int(ray_cfg["num_workers"])
    cpu_per_worker = args.cpu_per_worker or float(ray_cfg["cpu_per_worker"])
    gpu_per_worker = args.gpu_per_worker or float(ray_cfg["gpu_per_worker"])
    use_gpu = bool(args.use_gpu or ray_cfg.get("use_gpu", False))
    max_failures = args.max_failures or int(ray_cfg["max_failures"])
    storage_path = args.storage_path or ray_cfg["storage_path"]

    xgb_params = {k: v for k, v in config["base_params"].items() if k != "num_boost_round"}

    trainer = XGBoostTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "label_column": "label",
            "num_boost_round": int(config["base_params"]["num_boost_round"]),
            "checkpoint_frequency": int(ray_cfg["checkpoint_frequency"]),
            "xgboost_params": xgb_params,
        },
        datasets={
            "train": ray.data.from_pandas(train_frame),
            "validation": ray.data.from_pandas(val_frame),
        },
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            resources_per_worker=(
                {"CPU": cpu_per_worker, "GPU": gpu_per_worker}
                if use_gpu
                else {"CPU": cpu_per_worker}
            ),
        ),
        run_config=RunConfig(
            name=args.run_name,
            storage_path=storage_path,
            failure_config=FailureConfig(max_failures=max_failures),
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="validation_logloss",
                checkpoint_score_order="min",
            ),
        ),
    )

    print("Starting Ray Train...")
    result = trainer.fit()
    booster = RayTrainReportCallback.get_model(result.checkpoint)

    # Final evaluation
    dval = xgboost.DMatrix(bundle.val_features, label=bundle.val_target)
    scores = booster.predict(dval)
    predictions = (scores >= 0.5).astype(int)
    final_metrics = compute_binary_metrics(bundle.val_target, predictions, scores)

    # Export model
    artifact_dir = ensure_dir(args.artifact_dir)
    model_path = artifact_dir / "gemspot_ray_train_model.ubj"
    booster.save_model(str(model_path))

    summary = {
        "run_name": args.run_name,
        "code_version": get_git_sha(),
        "validation_metrics": final_metrics,
        "storage_path": storage_path,
        "num_workers": num_workers,
        "max_failures": max_failures,
    }

    # Log to MLflow
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
        exp_name = args.experiment_name or config.get("ray_experiment_name", "GemSpot-WillVisit-RayTrain")
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name=args.run_name, log_system_metrics=True):
            mlflow.set_tags({
                "project": "GemSpot",
                "workflow": "ray-train-xgboost",
                "code_version": get_git_sha(),
            })
            mlflow.log_params(flatten_dict({"base_params": config["base_params"]}))
            mlflow.log_params({
                "ray.num_workers": num_workers,
                "ray.cpu_per_worker": cpu_per_worker,
                "ray.max_failures": max_failures,
            })
            mlflow.log_metrics({k: float(v) for k, v in final_metrics.items()})
            mlflow.log_artifact(str(model_path), artifact_path="exported_models")
            mlflow.log_text(json.dumps(summary, indent=2), "ray_train_summary.json")

    print("\n=== Ray Train Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
