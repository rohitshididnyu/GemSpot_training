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
from ray import tune
from ray.train import CheckpointConfig, FailureConfig, RunConfig
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from gemspot_training.ray_data import make_xgboost_frame_bundle
from gemspot_training.training import compute_binary_metrics
from gemspot_training.utils import ensure_dir, flatten_dict, get_git_sha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GemSpot bonus tuning with Ray Tune + XGBoost.")
    parser.add_argument("--config", default="configs/ray_bonus.yaml", help="Path to Ray bonus YAML config.")
    parser.add_argument("--train-csv", required=True, help="Training CSV path.")
    parser.add_argument("--val-csv", required=True, help="Validation CSV path.")
    parser.add_argument("--ray-address", default=None, help="Optional Ray address. Defaults to RAY_ADDRESS or auto.")
    parser.add_argument("--storage-path", default=None, help="Optional Ray storage path override.")
    parser.add_argument("--tracking-uri", default=None, help="Optional MLflow tracking URI override.")
    parser.add_argument("--experiment-name", default=None, help="Optional MLflow experiment override.")
    parser.add_argument("--run-name", default="GemSpot-RayTune-XGBoost", help="Logical Tune experiment name.")
    parser.add_argument("--artifact-dir", default="artifacts/ray_tune", help="Artifact export directory.")
    parser.add_argument("--num-samples", type=int, default=None, help="Override Tune sample count.")
    parser.add_argument("--max-concurrent-trials", type=int, default=None, help="Override max concurrent trials.")
    parser.add_argument("--cpu-per-trial", type=float, default=None, help="Override CPUs per trial.")
    parser.add_argument("--gpu-per-trial", type=float, default=None, help="Override GPUs per trial.")
    parser.add_argument("--max-failures", type=int, default=None, help="Override failure retries.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_search_space(search_space_cfg: dict) -> dict:
    search_space = {}
    for name, spec in search_space_cfg.items():
        spec_type = spec["type"]
        if spec_type == "choice":
            search_space[name] = tune.choice(spec["values"])
        elif spec_type == "uniform":
            search_space[name] = tune.uniform(spec["lower"], spec["upper"])
        elif spec_type == "loguniform":
            search_space[name] = tune.loguniform(spec["lower"], spec["upper"])
        elif spec_type == "randint":
            search_space[name] = tune.randint(spec["lower"], spec["upper"])
        else:
            raise ValueError(f"Unsupported search space type: {spec_type}")
    return search_space


def train_trial(sampled_params: dict, static_cfg: dict) -> None:
    bundle = make_xgboost_frame_bundle(static_cfg["train_csv"], static_cfg["val_csv"], static_cfg["config"])

    dtrain = xgboost.DMatrix(bundle.train_features, label=bundle.train_target)
    dval = xgboost.DMatrix(bundle.val_features, label=bundle.val_target)

    params = dict(static_cfg["base_params"])
    params.update(sampled_params)
    total_rounds = int(static_cfg["num_boost_round"])
    report_every = int(static_cfg["report_every"])

    checkpoint = tune.get_checkpoint()
    booster = None
    completed_rounds = 0
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_path = Path(checkpoint_dir) / "model.ubj"
            metadata_path = Path(checkpoint_dir) / "metadata.json"
            if model_path.exists():
                booster = xgboost.Booster()
                booster.load_model(model_path.as_posix())
            if metadata_path.exists():
                completed_rounds = int(json.loads(metadata_path.read_text(encoding="utf-8"))["completed_rounds"])

    trial_id = os.getenv("TUNE_TRIAL_ID", "unknown")
    if static_cfg["tracking_uri"]:
        mlflow.set_tracking_uri(static_cfg["tracking_uri"])
        mlflow.set_experiment(static_cfg["experiment_name"])
        mlflow.start_run(run_name=f"ray-tune-trial-{trial_id}", log_system_metrics=True)
        mlflow.set_tags(
            {
                "project": "GemSpot",
                "workflow": "ray-tune-xgboost",
                "trial_id": trial_id,
                "code_version": get_git_sha(),
            }
        )
        mlflow.log_params(flatten_dict({"sampled_params": sampled_params}))

    try:
        while completed_rounds < total_rounds:
            chunk = min(report_every, total_rounds - completed_rounds)
            evals_result: dict[str, dict[str, list[float]]] = {}
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
            metrics.update(
                {
                    "training_iteration": completed_rounds,
                    "train_logloss": float(evals_result["train"]["logloss"][-1]),
                    "validation_logloss": float(evals_result["validation"]["logloss"][-1]),
                    "validation_auc": float(evals_result["validation"]["auc"][-1]),
                    "validation_aucpr": float(evals_result["validation"]["aucpr"][-1]),
                }
            )

            if static_cfg["tracking_uri"]:
                mlflow.log_metrics(
                    {key: float(value) for key, value in metrics.items() if isinstance(value, (float, int))},
                    step=completed_rounds,
                )

            with tempfile.TemporaryDirectory() as checkpoint_dir:
                model_path = Path(checkpoint_dir) / "model.ubj"
                metadata_path = Path(checkpoint_dir) / "metadata.json"
                booster.save_model(model_path.as_posix())
                metadata_path.write_text(
                    json.dumps({"completed_rounds": completed_rounds}, indent=2),
                    encoding="utf-8",
                )
                tune.report(metrics, checkpoint=Checkpoint.from_directory(checkpoint_dir))
    finally:
        if static_cfg["tracking_uri"] and mlflow.active_run():
            mlflow.end_run()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    ray_address = args.ray_address or "auto"
    ray.init(address=ray_address, ignore_reinit_error=True)

    tune_cfg = dict(config["ray_tune"])
    storage_path = args.storage_path or config["ray_train"]["storage_path"]
    num_samples = args.num_samples if args.num_samples is not None else int(tune_cfg["num_samples"])
    max_concurrent_trials = (
        args.max_concurrent_trials
        if args.max_concurrent_trials is not None
        else int(tune_cfg["max_concurrent_trials"])
    )
    cpu_per_trial = args.cpu_per_trial if args.cpu_per_trial is not None else float(tune_cfg["cpu_per_trial"])
    gpu_per_trial = args.gpu_per_trial if args.gpu_per_trial is not None else float(tune_cfg["gpu_per_trial"])
    max_failures = args.max_failures if args.max_failures is not None else int(config["ray_train"]["max_failures"])

    base_params = {
        key: value
        for key, value in config["base_params"].items()
        if key != "num_boost_round"
    }
    search_space = build_search_space(config["search_space"])

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
                "experiment_name": args.experiment_name or config.get("ray_tune_experiment_name", "GemSpot-WillVisit-RayTune"),
            },
        ),
        resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
    )

    metric = tune_cfg["metric"]
    mode = tune_cfg["mode"]
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
            max_concurrent_trials=max_concurrent_trials,
        ),
        run_config=RunConfig(
            name=args.run_name,
            storage_path=storage_path,
            failure_config=FailureConfig(max_failures=max_failures),
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute=metric,
                checkpoint_score_order=mode,
            ),
        ),
    )

    result_grid = tuner.fit()
    best_result = result_grid.get_best_result(metric=metric, mode=mode)

    artifact_dir = ensure_dir(args.artifact_dir)
    best_model_local = artifact_dir / "best_tuned_model.ubj"
    best_summary_local = artifact_dir / "best_tuned_result.json"

    with best_result.checkpoint.as_directory() as checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        source_model = checkpoint_path / "model.ubj"
        if source_model.exists():
            best_model_local.write_bytes(source_model.read_bytes())

    summary = {
        "run_name": args.run_name,
        "code_version": get_git_sha(),
        "best_config": best_result.config,
        "best_metrics": best_result.metrics,
        "best_checkpoint_path": best_result.checkpoint.path if best_result.checkpoint else "",
        "storage_path": storage_path,
        "num_samples": num_samples,
        "max_concurrent_trials": max_concurrent_trials,
    }
    best_summary_local.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name or config.get("ray_tune_experiment_name", "GemSpot-WillVisit-RayTune"))
        with mlflow.start_run(run_name=f"{args.run_name}-best-summary", log_system_metrics=True):
            mlflow.set_tags(
                {
                    "project": "GemSpot",
                    "workflow": "ray-tune-xgboost-summary",
                    "code_version": get_git_sha(),
                }
            )
            mlflow.log_params(flatten_dict({"best_config": best_result.config}))
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in best_result.metrics.items()
                    if isinstance(value, (float, int))
                }
            )
            if best_model_local.exists():
                mlflow.log_artifact(str(best_model_local), artifact_path="exported_models")
            mlflow.log_text(json.dumps(summary, indent=2), "ray_tune_summary.json")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
