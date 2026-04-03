from __future__ import annotations

import argparse

from mlflow.tracking import MlflowClient


def format_params(run) -> str:
    keys = ["candidate.kind", "candidate.params.C", "candidate.params.max_depth", "candidate.params.n_estimators"]
    parts = []
    for key in keys:
        if key in run.data.params:
            parts.append(f"{key.split('.')[-1]}={run.data.params[key]}")
    return ", ".join(parts) or "-"


def format_metrics(run) -> tuple[str, str]:
    quality_keys = ["accuracy", "f1", "roc_auc", "average_precision"]
    cost_keys = ["train_seconds", "rows_per_second"]

    quality = ", ".join(
        f"{key}={run.data.metrics[key]:.4f}" for key in quality_keys if key in run.data.metrics
    ) or "-"
    cost = ", ".join(
        f"{key}={run.data.metrics[key]:.2f}" for key in cost_keys if key in run.data.metrics
    ) or "-"
    return quality, cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Markdown run table from MLflow.")
    parser.add_argument("--experiment-name", required=True, help="MLflow experiment name.")
    parser.add_argument("--tracking-uri", required=True, help="MLflow tracking URI.")
    parser.add_argument(
        "--ui-base-url",
        default="",
        help="Optional public MLflow base URL, for example http://1.2.3.4:5000",
    )
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=args.tracking_uri)
    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise SystemExit(f"Experiment not found: {args.experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],
    )

    print("| Candidate | MLflow run link | Code version | Key hyperparams | Key model metrics | Key training cost metrics | Notes |")
    print("| --- | --- | --- | --- | --- | --- | --- |")

    for run in runs:
        run_link = "-"
        if args.ui_base_url:
            run_link = f"[{run.info.run_id[:8]}]({args.ui_base_url}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id})"

        code_version = run.data.tags.get("code_version", "-")
        quality, cost = format_metrics(run)
        notes = run.data.params.get("candidate.notes", "")

        print(
            f"| {run.data.tags.get('candidate_name', '-')} | {run_link} | {code_version} | "
            f"{format_params(run)} | {quality} | {cost} | {notes} |"
        )


if __name__ == "__main__":
    main()
