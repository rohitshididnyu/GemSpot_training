#!/usr/bin/env python3
"""
GemSpot — End-to-End Retraining Pipeline Orchestrator
======================================================
Chains the full retraining workflow:
  1. Run batch pipeline (PostgreSQL → Iceberg)
  2. Export training data (Iceberg → CSV)
  3. Train all model candidates
  4. Run quality gate (evaluate + register best model)
  5. Deploy to canary (optional)

Designed to run as a Kubernetes CronJob (weekly) or manual trigger.

Usage:
    python scripts/retrain_pipeline.py \\
        --tracking-uri http://mlflow:5000 \\
        --minio-endpoint http://minio:9000
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrain-pipeline")


def run_step(name: str, cmd: list[str], env: dict | None = None) -> bool:
    """Execute a pipeline step as a subprocess."""
    logger.info("=" * 60)
    logger.info("STEP: %s", name)
    logger.info("CMD:  %s", " ".join(cmd))
    logger.info("=" * 60)

    run_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, env=run_env, capture_output=False)

    if result.returncode != 0:
        logger.error("STEP FAILED: %s (exit code %d)", name, result.returncode)
        return False

    logger.info("STEP PASSED: %s", name)
    return True


def main():
    parser = argparse.ArgumentParser(description="GemSpot Retraining Pipeline")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    parser.add_argument("--minio-endpoint", default=os.getenv("ENDPOINT_URL", "http://minio:9000"))
    parser.add_argument("--aws-key", default=os.getenv("AWS_ACCESS_KEY_ID", "admin"))
    parser.add_argument("--aws-secret", default=os.getenv("AWS_SECRET_ACCESS_KEY", "password"))
    parser.add_argument("--app-db-url", default=os.getenv("APP_DB_URL", "postgresql://user:password@postgres:5432/adventurelog"))
    parser.add_argument("--iceberg-db-url", default=os.getenv("ICEBERG_DB_URL", "postgresql+psycopg2://user:password@postgres:5432/iceberg_catalog"))
    parser.add_argument("--config", default="configs/candidates.yaml")
    parser.add_argument("--experiment-name", default="GemSpot-WillVisit")
    parser.add_argument("--model-name", default="GemSpotWillVisit")
    parser.add_argument("--data-dir", default="/data/training_sets")
    parser.add_argument("--skip-batch", action="store_true", help="Skip batch pipeline (use existing data)")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip canary deployment")
    args = parser.parse_args()

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info("Starting GemSpot Retraining Pipeline | run_date=%s", run_date)

    # Determine script base paths
    training_root = Path(__file__).resolve().parent.parent
    gemspot_root = training_root.parent / "gemspot"

    env = {
        "MLFLOW_TRACKING_URI": args.tracking_uri,
        "ENDPOINT_URL": args.minio_endpoint,
        "AWS_ACCESS_KEY_ID": args.aws_key,
        "AWS_SECRET_ACCESS_KEY": args.aws_secret,
        "APP_DB_URL": args.app_db_url,
        "ICEBERG_DB_URL": args.iceberg_db_url,
    }

    # ── Step 1: Batch Pipeline ───────────────────────────────────────────────
    if not args.skip_batch:
        batch_script = gemspot_root / "batch_pipeline" / "pipeline.py"
        if batch_script.exists():
            if not run_step(
                "Batch Pipeline (PostgreSQL → Iceberg)",
                [sys.executable, str(batch_script)],
                env=env,
            ):
                logger.error("Batch pipeline failed — aborting")
                sys.exit(1)
        else:
            logger.warning("Batch pipeline script not found at %s — skipping", batch_script)

    # ── Step 2: Export Training Data ─────────────────────────────────────────
    export_script = training_root / "scripts" / "export_training_data.py"
    if not run_step(
        "Export Training Data (Iceberg → CSV)",
        [
            sys.executable, str(export_script),
            "--minio-endpoint", args.minio_endpoint,
            "--aws-key", args.aws_key,
            "--aws-secret", args.aws_secret,
            "--iceberg-db-url", args.iceberg_db_url,
            "--output-dir", args.data_dir,
            "--upload-to-s3",
        ],
        env=env,
    ):
        logger.error("Data export failed — aborting")
        sys.exit(1)

    # ── Step 3: Train Model Candidates ───────────────────────────────────────
    train_csv = os.path.join(args.data_dir, "train.csv")
    val_csv = os.path.join(args.data_dir, "val.csv")
    config_path = str(training_root / args.config)

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        logger.error("Training data not found at %s — aborting", args.data_dir)
        sys.exit(1)

    train_script = training_root / "src" / "train.py"
    if not run_step(
        "Train Model Candidates",
        [
            sys.executable, str(train_script),
            "--config", config_path,
            "--train-csv", train_csv,
            "--val-csv", val_csv,
            "--tracking-uri", args.tracking_uri,
            "--experiment-name", args.experiment_name,
        ],
        env=env,
    ):
        logger.error("Training failed — aborting")
        sys.exit(1)

    # ── Step 4: Quality Gate ─────────────────────────────────────────────────
    gate_script = training_root / "scripts" / "quality_gate.py"
    gate_passed = run_step(
        "Quality Gate (evaluate + register)",
        [
            sys.executable, str(gate_script),
            "--tracking-uri", args.tracking_uri,
            "--experiment-name", args.experiment_name,
            "--model-name", args.model_name,
        ],
        env=env,
    )

    if not gate_passed:
        logger.warning("Quality gate failed — no model promoted to Staging")
        logger.info("Pipeline completed with warnings (no new model registered)")
        sys.exit(0)  # Not a hard failure — the system keeps running with the existing model

    # ── Step 5: Deploy to Canary (optional) ──────────────────────────────────
    if not args.skip_deploy:
        deploy_script = training_root.parent / "scripts" / "deploy_model.py"
        if deploy_script.exists():
            run_step(
                "Deploy to Canary",
                [
                    sys.executable, str(deploy_script),
                    "--tracking-uri", args.tracking_uri,
                    "--model-name", args.model_name,
                    "--stage", "Staging",
                ],
                env=env,
            )
        else:
            logger.info("Deploy script not found at %s — skipping deployment", deploy_script)

    logger.info("Retraining pipeline completed successfully")


if __name__ == "__main__":
    main()
