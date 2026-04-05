"""Ray Tune hyperparameter search for GemSpot XGBoost.

Runs ASHA early-stopping over an XGBoost search space, checkpoints to
S3 (MinIO), and logs the best trial to MLflow.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from functools import partial
from pathlib import Path

import mlflow
import ray
import xgboost
import yaml

# Fix Ray Tune verbose output bug in some versions
os.environ.setdefault("RAY_AIR_NEW_OUTPUT", "0")

from ray import tune  # noqa: E402
from ray.train import CheckpointConfig, FailureConfig, RunConfig  # noqa: E402
from ray.tune.schedulers import ASHAScheduler

from gemspot_training.ray_data import make_xgboost_frame_bundle
from gemspot_training.training import compute_binary_metrics
from gemspot_training.utils import ensure_dir, flatten_dict, get_git_sha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GemSpot bonus: Ray Tune + XGBoost.")
    parser.add_argument("--config", default="configs/ray_bonus.yaml")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--storage-path", default=None)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default="GemSpot-RayTune-XGBoost")
    parser.add_argument("--artifact-dir", default="artifacts/ray_tune")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-concurrent-trials", type=int, default=None)
    parser.add_argument("--cpu-per-trial", type=float, default=None)
    parser.add_argument("--gpu-per-trial", type=float, default=None)
    parser.add_argument("--max-failures", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_search_space(search_space_cfg: dict) -> dict:
    space = {}
    for name, spec in search_space_cfg.items():
        t = spec["type"]
        if t == "choice":
            space[name] = tune.choice(spec["values"])
        elif t == "uniform":
            space[name] = tune.uniform(spec["lower"], spec["upper"])
        elif t == "loguniform":
            space[name] = tune.loguniform(spec["lower"], spec["upper"])
        elif t == "randint":
            space[name] = tune.randint(spec["lower"], spec["upper"])
        else:
            raise ValueError(f"Unknown search space type: {t}")
    return space


def train_trial(sampled_params: dict, static_cfg: dict) -> None:
    """One Ray Tune trial: train XGBoost, report metrics, checkpoint."""
    bundle = make_xgboost_frame_bundle(
        static_cfg["train_csv"], static_cfg["val_csv"], static_cfg["config"]
    )

    dtrain = xgboost.DMatrix(bundle.train_features, label=bundle.train_target)
    dval = xgboost.DMatrix(bundle.val_features, label=bundle.val_target)

    params = dict(static_cfg["base_params"])
    params.update(sampled_params)
    total_rounds = int(static_cfg["num_boost_round"])
    report_every = int(static_cfg["report_every"])

    # Resume from checkpoint if available
    checkpoint = ray.train.get_checkpoint()
    booster = None
    completed_rounds = 0
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            model_path = Path(ckpt_dir) / "model.ubj"
            meta_path = Path(ckpt_dir) / "metadata.json"
            if model_path.exists():
                booster = xgboost.Booster()
                booster.load_model(str(model_path))
            if meta_path.exists():
                completed_rounds = int(
                    json.loads(meta_path.read_text())["completed_rounds"]
                )

    trial_id = os.getenv("TUNE_TRIAL_ID", "unknown")

    # Optional MLflow logging per trial
    if static_cfg["tracking_uri"]:
        mlflow.set_tracking_uri(static_cfg["tracking_uri"])
        mlflow.set_experiment(static_cfg["experiment_name"])
        mlflow.start_run(run_name=f"ray-tune-trial-{trial_id}", log_system_metrics=True)
        mlflow.set_tags({
            "project": "GemSpot",
            "workflow": "ray-tune-xgboost",
            "trial_id": trial_id,
            "code_version": get_git_sha(),
        })
        mlflow.log_params(flatten_dict({"sampled": sampled_params}))

    try:
        while completed_rounds < total_rounds:
            chunk = min(report_every, total_rounds - completed_rounds)
            evals_result: dict = {}
            booster = xgboost.train(
                params=params,
                dtrain=dtrain,
                evals=[(dtrain, "train"), (dval, "validation")],
                num_boost_round=chunk,
                xgb_model=booster,
                evals_result=evals_result,
                verbose_eval=False,
            )
            completed_rounds += chunk

            scores = booster.predict(dval)
            predictions = (scores >= 0.5).astype(int)
            metrics = compute_binary_metrics(bundle.val_target, predictions, scores)
            metrics.update({
                "training_iteration": completed_rounds,
                "train_logloss": float(evals_result["train"]["logloss"][-1]),
                "validation_logloss": float(evals_result["validation"]["logloss"][-1]),
            })

            if static_cfg["tracking_uri"]:
                mlflow.log_metrics(
                    {k: float(v) for k, v in metrics.items() if isinstance(v, (float, int))},
                    step=completed_rounds,
                )

            # Save checkpoint and report
            with tempfile.TemporaryDirectory() as ckpt_dir:
                booster.save_model(str(Path(ckpt_dir) / "model.ubj"))
                Path(ckpt_dir, "metadata.json").write_text(
                    json.dumps({"completed_rounds": completed_rounds})
                )
                checkpoint = ray.train.Checkpoint.from_directory(ckpt_dir)
                ray.train.report(metrics, checkpoint=checkpoint)
    finally:
        if static_cfg["tracking_uri"] and mlflow.active_run():
            mlflow.end_run()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    ray.init(address=args.ray_address or "auto", ignore_reinit_error=True)

    tune_cfg = config["ray_tune"]
    num_samples = args.num_samples or int(tune_cfg["num_samples"])
    max_concurrent = args.max_concurrent_trials or int(tune_cfg["max_concurrent_trials"])
    cpu_per_trial = args.cpu_per_trial or float(tune_cfg["cpu_per_trial"])
    gpu_per_trial = args.gpu_per_trial or float(tune_cfg["gpu_per_trial"])
    max_failures = args.max_failures or int(config["ray_train"]["max_failures"])
    storage_path = args.storage_path or config["ray_train"]["storage_path"]

    base_params = {k: v for k, v in config["base_params"].items() if k != "num_boost_round"}
    search_space = build_search_space(config["search_space"])

    metric = tune_cfg["metric"]
    mode = tune_cfg["mode"]

    trainable = tune.with_resources(
        partial(
            train_trial,
            static_cfg={
                "config": config,
                "train_csv": args.train_csv,
                "val_csv": args.val_csv,
                "base_params": base_params,
                "num_boost_round": int(config["base_params"]["num_boost_round"]),
                "report_every": int(tune_cfg["report_every"]),
                "tracking_uri": args.tracking_uri,
                "experiment_name": args.experiment_name
                    or config.get("ray_tune_experiment_name", "GemSpot-WillVisit-RayTune"),
            },
        ),
        resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
    )

    scheduler = ASHAScheduler(
        max_t=int(config["base_params"]["num_boost_round"]),
        grace_period=int(tune_cfg["grace_period"]),
        reduction_factor=int(tune_cfg["reduction_factor"]),
    )

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent,
        ),
        run_config=RunConfig(
            name=args.run_name,
            storage_path=storage_path,
            verbose=1,
            failure_config=FailureConfig(max_failures=max_failures),
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute=metric,
                checkpoint_score_order=mode,
            ),
        ),
    )

    result_grid = tuner.fit()
    best = result_grid.get_best_result(metric=metric, mode=mode)

    # Export best model
    artifact_dir = ensure_dir(args.artifact_dir)
    best_model_path = artifact_dir / "best_tuned_model.ubj"
    best_summary_path = artifact_dir / "best_tuned_result.json"

    with best.checkpoint.as_directory() as ckpt_dir:
        src = Path(ckpt_dir) / "model.ubj"
        if src.exists():
            best_model_path.write_bytes(src.read_bytes())

    summary = {
        "run_name": args.run_name,
        "code_version": get_git_sha(),
        "best_config": best.config,
        "best_metrics": {k: v for k, v in best.metrics.items() if isinstance(v, (float, int))},
        "storage_path": storage_path,
        "num_samples": num_samples,
    }
    best_summary_path.write_text(json.dumps(summary, indent=2))

    # Log best result to MLflow
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
        exp_name = args.experiment_name or config.get("ray_tune_experiment_name", "GemSpot-WillVisit-RayTune")
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name=f"{args.run_name}-best-summary", log_system_metrics=True):
            mlflow.set_tags({
                "project": "GemSpot",
                "workflow": "ray-tune-best-summary",
                "code_version": get_git_sha(),
            })
            mlflow.log_params(flatten_dict({"best_config": best.config}))
            mlflow.log_metrics(
                {k: float(v) for k, v in best.metrics.items() if isinstance(v, (float, int))}
            )
            if best_model_path.exists():
                mlflow.log_artifact(str(best_model_path), artifact_path="exported_models")
            mlflow.log_text(json.dumps(summary, indent=2), "ray_tune_summary.json")

    print("\n=== Best Trial Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
