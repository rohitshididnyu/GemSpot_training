#!/usr/bin/env python3
"""
GemSpot Model Quality Gate
===========================
Evaluates the latest training runs against quality thresholds and registers
passing models in the MLflow Model Registry.

Quality gates (all must pass):
  - F1 >= 0.60          (meaningfully above majority-class baseline)
  - ROC-AUC >= 0.70     (ranking quality well above random 0.50)
  - Precision >= 0.55   (users shouldn't see irrelevant places)
  - Recall >= 0.50      (must surface at least half of relevant places)
  - F1 > baseline + 0.05 (no-regression vs dummy classifier)

If a candidate passes all gates AND is the best among candidates in the run,
it is registered in MLflow Model Registry as stage="Staging".

Usage:
    python scripts/quality_gate.py --tracking-uri http://mlflow:5000 \\
        --experiment-name GemSpot-WillVisit \\
        --model-name GemSpotWillVisit
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Quality thresholds ───────────────────────────────────────────────────────
GATES = {
    "f1_minimum":        {"metric": "f1",        "threshold": 0.60, "op": ">="},
    "auc_minimum":       {"metric": "roc_auc",   "threshold": 0.70, "op": ">="},
    "precision_minimum": {"metric": "precision",  "threshold": 0.55, "op": ">="},
    "recall_minimum":    {"metric": "recall",     "threshold": 0.50, "op": ">="},
}

BASELINE_F1_MARGIN = 0.05  # Must beat baseline F1 by this margin


def evaluate_gates(metrics: dict, baseline_f1: float | None = None) -> dict[str, bool]:
    """Evaluate all quality gates. Returns {gate_name: passed}."""
    results = {}
    for gate_name, gate in GATES.items():
        value = metrics.get(gate["metric"], 0.0)
        if gate["op"] == ">=":
            results[gate_name] = value >= gate["threshold"]
        elif gate["op"] == ">":
            results[gate_name] = value > gate["threshold"]

    # Baseline comparison gate
    if baseline_f1 is not None:
        results["beats_baseline"] = metrics.get("f1", 0.0) > baseline_f1 + BASELINE_F1_MARGIN
    else:
        # If no baseline available, skip this gate
        results["beats_baseline"] = True

    return results


def find_baseline_f1(client: MlflowClient, experiment_id: str) -> float | None:
    """Find the F1 of the most recent baseline (dummy) run."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.candidate_kind = 'dummy'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs:
        return runs[0].data.metrics.get("f1")
    return None


def find_production_f1(client: MlflowClient, model_name: str) -> float | None:
    """Find the F1 of the current production model."""
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            run = client.get_run(versions[0].run_id)
            return run.data.metrics.get("f1")
    except Exception:
        pass
    return None


def run_quality_gate(
    tracking_uri: str,
    experiment_name: str,
    model_name: str,
    max_runs: int = 10,
) -> bool:
    """
    Main quality gate logic:
    1. Find recent non-baseline runs from the experiment
    2. Evaluate each against quality gates
    3. Register the best passing model in MLflow
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error("Experiment '%s' not found", experiment_name)
        return False

    # Find baseline F1
    baseline_f1 = find_baseline_f1(client, experiment.experiment_id)
    logger.info("Baseline F1: %s", f"{baseline_f1:.4f}" if baseline_f1 is not None else "N/A")

    # Find current production F1
    production_f1 = find_production_f1(client, model_name)
    logger.info("Production F1: %s", f"{production_f1:.4f}" if production_f1 is not None else "N/A")

    # Get recent non-baseline runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.candidate_kind != 'dummy'",
        order_by=["start_time DESC"],
        max_results=max_runs,
    )
    if not runs:
        logger.warning("No non-baseline runs found in experiment '%s'", experiment_name)
        return False

    # Evaluate each run
    best_run = None
    best_f1 = -1.0
    best_gates = {}

    for run in runs:
        metrics = run.data.metrics
        gates = evaluate_gates(metrics, baseline_f1)
        all_passed = all(gates.values())

        candidate_name = run.data.tags.get("candidate_name", "unknown")
        f1 = metrics.get("f1", 0.0)
        auc = metrics.get("roc_auc", 0.0)

        # Tag run with gate results
        client.set_tag(run.info.run_id, "quality_gate.passed", str(all_passed))
        client.set_tag(run.info.run_id, "quality_gate.details", json.dumps(gates))

        status = "PASS" if all_passed else "FAIL"
        logger.info(
            "  %s: %s (F1=%.4f, AUC=%.4f) | gates=%s",
            candidate_name, status, f1, auc, gates,
        )

        if all_passed and f1 > best_f1:
            best_run = run
            best_f1 = f1
            best_gates = gates

    if best_run is None:
        logger.warning("No runs passed all quality gates")
        return False

    # Check production regression
    if production_f1 is not None:
        regression_margin = 0.02
        if best_f1 < production_f1 - regression_margin:
            logger.warning(
                "Best candidate F1=%.4f is below production F1=%.4f (margin=%.2f) — not registering",
                best_f1, production_f1, regression_margin,
            )
            return False

    # Register the best model
    candidate_name = best_run.data.tags.get("candidate_name", "unknown")
    logger.info(
        "Registering best model: %s (F1=%.4f, AUC=%.4f)",
        candidate_name,
        best_f1,
        best_run.data.metrics.get("roc_auc", 0.0),
    )

    model_uri = f"runs:/{best_run.info.run_id}/model"
    try:
        mv = mlflow.register_model(model_uri, model_name)
        logger.info("Registered model version %s", mv.version)

        # Transition to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Staging",
        )
        logger.info("Transitioned version %s to Staging", mv.version)

        # Tag the registered version
        client.set_model_version_tag(model_name, mv.version, "quality_gate.passed", "True")
        client.set_model_version_tag(model_name, mv.version, "quality_gate.f1", str(best_f1))
        client.set_model_version_tag(model_name, mv.version, "source_candidate", candidate_name)

        return True

    except Exception as e:
        logger.error("Failed to register model: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="GemSpot Model Quality Gate")
    parser.add_argument("--tracking-uri", default="http://mlflow:5000")
    parser.add_argument("--experiment-name", default="GemSpot-WillVisit")
    parser.add_argument("--model-name", default="GemSpotWillVisit")
    parser.add_argument("--max-runs", type=int, default=10)
    args = parser.parse_args()

    passed = run_quality_gate(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        max_runs=args.max_runs,
    )

    if passed:
        logger.info("Quality gate PASSED — model registered to Staging")
    else:
        logger.warning("Quality gate FAILED — no model registered")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
