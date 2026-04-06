"""Ray Tune hyperparameter search for GemSpot XGBoost.

Runs ASHA early-stopping over an XGBoost search space, checkpoints to
S3 (MinIO), and logs the best trial to MLflow.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from functools import partial
from pathlib import Path

import mlflow
import ray
import xgboost
import yaml

# Fix Ray Tune verbose output bug in some versions
os.environ.setdefault("RAY_AIR_NEW_OUTPUT", "0")
os.environ.setdefault("RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS", "0")

from ray import tune  # noqa: E402
from ray.tune import RunConfig  # noqa: E402
from ray.tune import FailureConfig  # noqa: E402
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


def restore_trial_state() -> tuple[xgboost.Booster | None, int]:
    """Restore the latest Tune checkpoint for this trial if one exists."""
    checkpoint = tune.get_checkpoint()
    if checkpoint is None:
        return None, 0

    with checkpoint.as_directory() as checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        model_path = checkpoint_path / "model.ubj"
        state_path = checkpoint_path / "state.json"

        booster: xgboost.Booster | None = None
        if model_path.exists():
            booster = xgboost.Booster()
            booster.load_model(str(model_path))

        completed_rounds = 0
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            completed_rounds = int(state.get("completed_rounds", 0))

    return booster, completed_rounds


def report_trial_progress(
    booster: xgboost.Booster,
    metrics: dict,
    sampled_params: dict,
    completed_rounds: int,
) -> None:
    """Persist a Tune checkpoint and report the latest metrics."""
    with tempfile.TemporaryDirectory(prefix="gemspot-ray-tune-") as checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        booster.save_model(str(checkpoint_path / "model.ubj"))
        (checkpoint_path / "state.json").write_text(
            json.dumps(
                {
                    "completed_rounds": completed_rounds,
                    "sampled_params": sampled_params,
                    "metrics": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tune.report(metrics, checkpoint=tune.Checkpoint.from_directory(checkpoint_dir))


def train_trial(sampled_params: dict, static_cfg: dict) -> None:
    """One Ray Tune trial: train XGBoost, report metrics, checkpoint."""
    import gc
    import traceback
    import warnings

    # Suppress Ray v2 deprecation warnings in worker processes
    # (driver-side env vars don't propagate to workers)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")
    os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"

    print(f"[trial] Starting trial with params: {sampled_params}", flush=True)

    try:
        bundle = make_xgboost_frame_bundle(
            static_cfg["train_csv"], static_cfg["val_csv"], static_cfg["config"]
        )
        print(f"[trial] Data loaded: train={len(bundle.train_features)}, val={len(bundle.val_features)}", flush=True)
    except Exception as e:
        print(f"[trial] FAILED to load data: {e}", flush=True)
        traceback.print_exc()
        raise

    try:
        dtrain = xgboost.DMatrix(bundle.train_features, label=bundle.train_target)
        dval = xgboost.DMatrix(bundle.val_features, label=bundle.val_target)
        # Free pandas frames immediately to reduce memory
        del bundle.train_features, bundle.val_features
        gc.collect()
        print("[trial] DMatrix created, pandas frames freed", flush=True)
    except Exception as e:
        print(f"[trial] FAILED to create DMatrix: {e}", flush=True)
        traceback.print_exc()
        raise

    params = dict(static_cfg["base_params"])
    params.update(sampled_params)
    total_rounds = int(static_cfg["num_boost_round"])
    report_every = int(static_cfg["report_every"])

    booster, completed_rounds = restore_trial_state()
    if completed_rounds:
        print(
            f"[trial] Resuming from checkpoint at round {completed_rounds}/{total_rounds}",
            flush=True,
        )

    trial_context = tune.get_context()
    trial_id = os.getenv("TUNE_TRIAL_ID") or trial_context.get_trial_id() or "unknown"
    trial_name = trial_context.get_trial_name() or f"trial-{trial_id}"

    # Optional MLflow logging per trial (wrapped in try/except so failures don't kill trial)
    mlflow_active = False
    if static_cfg.get("tracking_uri"):
        try:
            mlflow.set_tracking_uri(static_cfg["tracking_uri"])
            mlflow.set_experiment(static_cfg["experiment_name"])
            mlflow.start_run(run_name=trial_name, log_system_metrics=True)
            mlflow.set_tags({
                "project": "GemSpot",
                "workflow": "ray-tune-xgboost",
                "trial_id": trial_id,
                "code_version": get_git_sha(),
            })
            mlflow.log_params(flatten_dict({"sampled": sampled_params}))
            mlflow_active = True
            print(f"[trial] MLflow logging started for trial {trial_id}", flush=True)
        except Exception as e:
            print(f"[trial] WARNING: MLflow setup failed (non-fatal): {e}", flush=True)
            mlflow_active = False

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

            print(f"[trial] Round {completed_rounds}/{total_rounds} "
                  f"roc_auc={float(metrics.get('roc_auc', float('nan'))):.4f} "
                  f"val_logloss={float(metrics.get('validation_logloss', float('nan'))):.4f}",
                  flush=True)

            if mlflow_active:
                try:
                    mlflow.log_metrics(
                        {k: float(v) for k, v in metrics.items() if isinstance(v, (float, int))},
                        step=completed_rounds,
                    )
                except Exception as e:
                    print(f"[trial] WARNING: MLflow log_metrics failed: {e}", flush=True)

            # Report metrics to Ray Tune
            report_trial_progress(
                booster=booster,
                metrics=metrics,
                sampled_params=sampled_params,
                completed_rounds=completed_rounds,
            )
    except Exception as e:
        print(f"[trial] FAILED during training: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        if mlflow_active and mlflow.active_run():
            try:
                mlflow.end_run()
            except Exception:
                pass


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
            failure_config=FailureConfig(max_failures=max_failures),
        ),
    )

    result_grid = tuner.fit()
    try:
        best = result_grid.get_best_result(metric=metric, mode=mode)
    except RuntimeError as exc:
        raise RuntimeError(
            f"No successful Ray Tune trial reported metric '{metric}'. "
            "Check Ray trial logs under the run artifacts for the root cause."
        ) from exc

    # Export best result summary
    artifact_dir = ensure_dir(args.artifact_dir)
    best_summary_path = artifact_dir / "best_tuned_result.json"
    best_model_path = artifact_dir / "best_tuned_model.ubj"

    if best.checkpoint:
        with best.checkpoint.as_directory() as checkpoint_dir:
            checkpoint_model_path = Path(checkpoint_dir) / "model.ubj"
            if checkpoint_model_path.exists():
                shutil.copy2(checkpoint_model_path, best_model_path)

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
