# GemSpot Training Pipeline — DevOps Handoff

> **From**: Rohit (Training role)  
> **To**: DevOps team member  
> **Purpose**: Everything you need to integrate the training pipeline into the automated K8s deployment.

---

## TL;DR — What This Does

The training pipeline takes production user interaction data, trains XGBoost models, evaluates them against quality gates, and registers passing models in MLflow. The DevOps team needs to:

1. **Build one Docker image** (`gemspot-training`)
2. **Run it as a K8s CronJob** (weekly, after the batch pipeline)
3. **Ensure it can reach** MLflow, MinIO, and PostgreSQL

That's it. Everything else is automated inside the container.

---

## Table of Contents

1. [Architecture — How Training Fits in the System](#1-architecture)
2. [Scripts — What Each One Does](#2-scripts)
3. [Docker — Build & Run](#3-docker)
4. [Environment Variables — What to Set](#4-environment-variables)
5. [Kubernetes — CronJob Config](#5-kubernetes-cronjob)
6. [Full Automated Flow — Step by Step](#6-full-automated-flow)
7. [Integration Points — What Talks to What](#7-integration-points)
8. [Testing — How to Verify It Works](#8-testing)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Architecture

```
                      ┌──────────────────────────────────────────────┐
                      │         TRAINING PIPELINE (this repo)        │
                      │                                              │
  Batch Pipeline      │  ┌─────────────┐    ┌─────────────────────┐  │
  (weekly CronJob)    │  │ export_     │    │   train.py          │  │
  produces Iceberg ──►│  │ training_   │───►│   (trains 4         │  │
  dataset in MinIO    │  │ data.py     │    │    candidates,      │  │
                      │  │             │    │    logs to MLflow)   │  │
                      │  │ Reads from  │    │                     │  │
                      │  │ MinIO/      │    └────────┬────────────┘  │
                      │  │ Iceberg     │             │               │
                      │  └─────────────┘             ▼               │
                      │                    ┌─────────────────────┐   │
                      │                    │  quality_gate.py    │   │
                      │                    │  (evaluates metrics,│   │
                      │                    │   registers model   │   │
                      │                    │   if passes)        │   │
                      │                    └────────┬────────────┘   │
                      │                             │                │
                      └─────────────────────────────┼────────────────┘
                                                    │
                                                    ▼
                                           MLflow Model Registry
                                           (stage: "Staging")
                                                    │
                                                    ▼
                                           deploy_model.py
                                           (exports to MinIO,
                                            triggers canary rollout)
```

---

## 2. Scripts

All scripts are in `GemSpot_training/scripts/`. Here's what each does:

### `retrain_pipeline.py` — THE MAIN ENTRY POINT

**This is the only script DevOps needs to call.** It orchestrates everything.

```bash
python scripts/retrain_pipeline.py \
    --tracking-uri http://mlflow:5000 \
    --minio-endpoint http://minio:9000 \
    --app-db-url "postgresql://user:password@postgres:5432/adventurelog" \
    --iceberg-db-url "postgresql+psycopg2://user:password@postgres:5432/iceberg_catalog"
```

**What it does internally (in order):**
1. Calls batch pipeline (`gemspot/batch_pipeline/pipeline.py`) — snapshot PostgreSQL → Iceberg
2. Calls `export_training_data.py` — reads Iceberg, writes `train.csv` + `val.csv`
3. Calls `train.py` — trains all 4 model candidates, logs to MLflow
4. Calls `quality_gate.py` — evaluates, registers best model to MLflow Staging
5. Optionally calls `deploy_model.py` — exports model, triggers canary

**Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--tracking-uri` | `$MLFLOW_TRACKING_URI` or `http://mlflow:5000` | MLflow server |
| `--minio-endpoint` | `$ENDPOINT_URL` or `http://minio:9000` | MinIO S3 |
| `--aws-key` | `$AWS_ACCESS_KEY_ID` or `admin` | MinIO access key |
| `--aws-secret` | `$AWS_SECRET_ACCESS_KEY` or `password` | MinIO secret key |
| `--app-db-url` | `$APP_DB_URL` | PostgreSQL connection |
| `--iceberg-db-url` | `$ICEBERG_DB_URL` | Iceberg catalog DB |
| `--config` | `configs/candidates.yaml` | Model config |
| `--experiment-name` | `GemSpot-WillVisit` | MLflow experiment |
| `--model-name` | `GemSpotWillVisit` | MLflow registered model |
| `--data-dir` | `/data/training_sets` | Temp dir for CSVs |
| `--skip-batch` | false | Skip batch pipeline step |
| `--skip-deploy` | false | Skip canary deployment step |

### `export_training_data.py` — Iceberg → CSV

Reads from MinIO (Iceberg or raw Parquet) and writes `train.csv` + `val.csv`.

```bash
python scripts/export_training_data.py \
    --minio-endpoint http://minio:9000 \
    --output-dir /data/training_sets \
    --upload-to-s3
```

### `quality_gate.py` — Evaluate & Register

Queries MLflow for recent training runs, checks metrics against thresholds:

| Metric | Threshold | Why |
|--------|-----------|-----|
| F1 | >= 0.60 | Must beat majority-class baseline meaningfully |
| ROC-AUC | >= 0.70 | Ranking quality well above random (0.50) |
| Precision | >= 0.55 | Don't recommend irrelevant places |
| Recall | >= 0.50 | Must surface half of relevant places |
| vs baseline | F1 > baseline + 0.05 | No-regression gate |
| vs production | F1 >= production - 0.02 | Don't deploy worse models |

If the best candidate passes ALL gates, it's registered in MLflow as `GemSpotWillVisit` stage=`Staging`.

```bash
python scripts/quality_gate.py \
    --tracking-uri http://mlflow:5000 \
    --experiment-name GemSpot-WillVisit \
    --model-name GemSpotWillVisit
```

**Exit codes:** 0 = model registered, 1 = no model passed gates.

### `train.py` (in `src/`) — Core Training

Trains all candidates defined in `configs/candidates.yaml`. Logs everything to MLflow.

```bash
PYTHONPATH=src python src/train.py \
    --config configs/candidates.yaml \
    --train-csv /data/training_sets/train.csv \
    --val-csv /data/training_sets/val.csv \
    --tracking-uri http://mlflow:5000 \
    --experiment-name GemSpot-WillVisit
```

**What it logs to MLflow (per candidate run):**
- Parameters: all hyperparameters, dataset info, environment info
- Metrics: accuracy, F1, precision, recall, ROC-AUC, avg_precision, train_seconds
- Artifacts: `.joblib` model, sklearn MLflow model, `run_summary.json`, `reference_stats.json` (for drift detection)
- Tags: project, task, candidate_name, candidate_kind, code_version (git SHA)

---

## 3. Docker

### Build the Image

```bash
cd GemSpot_training
docker build -t gemspot-training:latest .
```

The existing `Dockerfile` works as-is:
```dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
CMD ["python", "src/train.py", "--help"]
```

**IMPORTANT**: The Dockerfile only installs `requirements.txt` from the training folder. The `retrain_pipeline.py` script calls the batch pipeline (`gemspot/batch_pipeline/pipeline.py`) as a subprocess, so if you want the full pipeline to run in one container, you need **one of these approaches**:

### Option A: Two CronJobs (RECOMMENDED)

Run batch pipeline and training as separate K8s CronJobs. Training starts after batch is done.

```
CronJob: batch-pipeline  →  runs at Sunday 02:00
CronJob: training         →  runs at Sunday 06:00 (4h later)
                              uses --skip-batch flag
```

Training command:
```bash
python scripts/retrain_pipeline.py --skip-batch \
    --tracking-uri http://mlflow:5000 \
    --minio-endpoint http://minio:9000
```

### Option B: Combined Dockerfile

Build a single image that includes both `gemspot/batch_pipeline/` and `GemSpot_training/`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# Install training deps
COPY GemSpot_training/requirements.txt /app/training-requirements.txt
RUN pip install --no-cache-dir -r /app/training-requirements.txt

# Install batch pipeline deps
RUN pip install --no-cache-dir boto3 pyiceberg psycopg2-binary pyarrow

# Copy training code
COPY GemSpot_training/ /app/
ENV PYTHONPATH=/app/src

# Copy batch pipeline (needed by retrain_pipeline.py step 1)
COPY gemspot/batch_pipeline/ /app/../gemspot/batch_pipeline/

CMD ["python", "scripts/retrain_pipeline.py"]
```

### Run Locally (for testing)

```bash
# Option 1: Full pipeline (needs PostgreSQL, MinIO, MLflow running)
docker run --rm --network host \
    -e MLFLOW_TRACKING_URI=http://localhost:5000 \
    -e ENDPOINT_URL=http://localhost:9000 \
    -e AWS_ACCESS_KEY_ID=admin \
    -e AWS_SECRET_ACCESS_KEY=password \
    -e APP_DB_URL="postgresql://user:password@localhost:5432/adventurelog" \
    -e ICEBERG_DB_URL="postgresql+psycopg2://user:password@localhost:5432/iceberg_catalog" \
    gemspot-training:latest \
    python scripts/retrain_pipeline.py --skip-batch

# Option 2: Train only (needs train/val CSVs mounted)
docker run --rm \
    -v /path/to/data:/data \
    -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
    gemspot-training:latest \
    python src/train.py \
        --config configs/candidates.yaml \
        --train-csv /data/train.csv \
        --val-csv /data/val.csv

# Option 3: Quality gate only
docker run --rm \
    -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
    gemspot-training:latest \
    python scripts/quality_gate.py
```

---

## 4. Environment Variables

Set these in the K8s ConfigMap/Secret. The training container reads them:

### Required

| Variable | Example | Used By |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `http://mlflow-svc:5000` | train.py, quality_gate.py |
| `ENDPOINT_URL` | `http://minio-svc:9000` | export_training_data.py |
| `AWS_ACCESS_KEY_ID` | `admin` | MinIO access |
| `AWS_SECRET_ACCESS_KEY` | `password` | MinIO access |

### Required if NOT using `--skip-batch`

| Variable | Example | Used By |
|----------|---------|---------|
| `APP_DB_URL` | `postgresql://user:pass@postgres-svc:5432/adventurelog` | batch pipeline |
| `ICEBERG_DB_URL` | `postgresql+psycopg2://user:pass@postgres-svc:5432/iceberg_catalog` | batch pipeline, export |

### Optional

| Variable | Default | Used By |
|----------|---------|---------|
| `MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` | `1` | train.py (system metrics) |
| `MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING` | `1` | train.py |

---

## 5. Kubernetes CronJob

### Recommended: Two Separate CronJobs

**Batch pipeline** (already in `k8s/base/batch-pipeline/cronjob.yaml`):
```yaml
schedule: "0 2 * * 0"  # Sunday 02:00 UTC
```

**Training** (update `k8s/base/training/cronjob.yaml`):
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: training
  labels:
    app: training
spec:
  schedule: "0 6 * * 0"  # Sunday 06:00 UTC (4h after batch pipeline)
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      backoffLimit: 1
      activeDeadlineSeconds: 10800  # 3 hour max
      template:
        metadata:
          labels:
            app: training
        spec:
          restartPolicy: OnFailure
          containers:
            - name: training
              image: ghcr.io/gemspot/training:latest
              command:
                - python
                - scripts/retrain_pipeline.py
                - --skip-batch
                - --tracking-uri
                - "http://mlflow-svc:5000"
                - --minio-endpoint
                - "http://minio-svc:9000"
                - --experiment-name
                - "GemSpot-WillVisit"
                - --model-name
                - "GemSpotWillVisit"
                - --data-dir
                - "/data/training_sets"
              envFrom:
                - configMapRef:
                    name: gemspot-config
                - secretRef:
                    name: gemspot-secrets
              resources:
                requests:
                  cpu: "2"
                  memory: 4Gi
                limits:
                  cpu: "4"
                  memory: 8Gi
              volumeMounts:
                - name: training-data
                  mountPath: /data
          volumes:
            - name: training-data
              emptyDir:
                sizeLimit: 2Gi
```

### Manual Trigger (one-off Job)

```bash
kubectl create job --from=cronjob/training manual-training-$(date +%s) -n gemspot-prod
```

---

## 6. Full Automated Flow

Here's exactly what happens when the training CronJob fires:

```
Sunday 06:00 UTC
    │
    ▼
┌─ retrain_pipeline.py (--skip-batch) ─────────────────────────────┐
│                                                                   │
│  Step 1: SKIPPED (batch pipeline ran at 02:00 separately)         │
│                                                                   │
│  Step 2: export_training_data.py                                  │
│    → Reads s3://agent-datalake/transformed/{date}/features.parquet│
│      (produced by batch pipeline at 02:00)                        │
│    → Splits by "split" column into train.csv / val.csv            │
│    → Writes to /data/training_sets/                               │
│    → Uploads to s3://agent-datalake/training_sets/{date}/         │
│                                                                   │
│  Step 3: train.py                                                 │
│    → Loads train.csv and val.csv                                  │
│    → Explodes list-encoded columns → 85 numeric features          │
│    → Trains 4 candidates:                                         │
│        baseline (dummy), hist_gradient_boosting,                  │
│        xgboost_v1, xgboost_v2                                    │
│    → Logs everything to MLflow experiment "GemSpot-WillVisit"     │
│    → Saves reference_stats.json artifact (for drift detection)    │
│    → Duration: ~2-5 minutes for 300K rows                         │
│                                                                   │
│  Step 4: quality_gate.py                                          │
│    → Queries MLflow for runs from step 3                          │
│    → Evaluates each against thresholds:                           │
│        F1>=0.60, AUC>=0.70, Precision>=0.55, Recall>=0.50        │
│    → Checks best candidate beats baseline by 0.05 F1              │
│    → Checks no regression vs current production model             │
│    → If PASS: registers best model as                             │
│        MLflow "GemSpotWillVisit" → stage "Staging"                │
│    → If FAIL: exits 1, pipeline logs warning, NO model registered │
│        (system keeps running with existing production model)      │
│                                                                   │
│  Step 5: deploy_model.py (if gate passed)                         │
│    → Downloads Staging model from MLflow                          │
│    → Extracts XGBoost Booster from sklearn pipeline               │
│    → Exports as .json to MinIO: s3://models/gemspot_xgb.json     │
│    → Triggers: kubectl rollout restart deployment/ml-serving      │
│      in canary namespace                                          │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
ML Serving (canary) picks up new model
    │
    ▼
canary_monitor.py (separate CronJob, every 5 min)
    → Checks Prometheus: latency, error rate, prediction distribution
    → After 2 hours of clean metrics → promotes Staging → Production
    → On degradation → rolls back (scales canary to 0)
```

---

## 7. Integration Points

### What Training READS FROM:

| Service | What | How |
|---------|------|-----|
| **MinIO** (`minio-svc:9000`) | Transformed Parquet from batch pipeline | `s3://agent-datalake/transformed/{date}/features.parquet` |
| **MinIO** | Iceberg catalog data | `s3://agent-datalake/iceberg/` |
| **PostgreSQL** (`postgres-svc:5432`) | Only if running batch pipeline (not skipped) | Direct SQL queries via psycopg2 |

### What Training WRITES TO:

| Service | What | How |
|---------|------|-----|
| **MLflow** (`mlflow-svc:5000`) | Experiment runs, metrics, artifacts, registered models | MLflow Python client |
| **MinIO** (`minio-svc:9000`) | Training CSVs at `s3://agent-datalake/training_sets/{date}/` | boto3 |
| **MinIO** | Exported model at `s3://models/gemspot_xgb.json` | boto3 (via deploy_model.py) |

### What DEPENDS on Training Output:

| Service | What It Reads | Where |
|---------|---------------|-------|
| **ML Serving** | Model from MLflow registry (stage=Production) | `mlflow_loader.py` |
| **ML Serving** | Fallback model from MinIO | `s3://models/gemspot_xgb.json` |
| **ML Serving** | Reference stats for drift detection | MLflow artifact `reference_stats.json` |
| **Canary Monitor** | Prometheus metrics from ml-serving | `/metrics` endpoint |

### Network Access Needed:

```
training-pod ──► mlflow-svc:5000      (HTTP, MLflow tracking + registry)
training-pod ──► minio-svc:9000       (HTTP, S3 API for data + models)
training-pod ──► postgres-svc:5432    (TCP, only if batch pipeline not skipped)
```

Make sure K8s NetworkPolicies (if any) allow these connections.

---

## 8. Testing

### Smoke Test (no external dependencies)

```bash
# Build image
docker build -t gemspot-training:latest -f GemSpot_training/Dockerfile GemSpot_training/

# Run help to verify image works
docker run --rm gemspot-training:latest python src/train.py --help

# Generate demo data and train locally (no MLflow needed)
docker run --rm \
    -v /tmp/gemspot-test:/app/data/demo \
    gemspot-training:latest \
    python scripts/make_demo_dataset.py --output-dir data/demo

docker run --rm \
    -v /tmp/gemspot-test:/app/data/demo \
    -v /tmp/gemspot-artifacts:/app/artifacts \
    gemspot-training:latest \
    python src/train.py \
        --config configs/candidates.yaml \
        --train-csv data/demo/gemspot_train.csv \
        --val-csv data/demo/gemspot_val.csv
```

### Integration Test (with MLflow)

```bash
# Start MLflow server
docker run -d --name mlflow -p 5000:5000 ghcr.io/mlflow/mlflow:latest \
    mlflow server --host 0.0.0.0 --port 5000

# Run training with MLflow
docker run --rm --network host \
    -v /tmp/gemspot-test:/app/data/demo \
    -e MLFLOW_TRACKING_URI=http://localhost:5000 \
    gemspot-training:latest \
    python src/train.py \
        --config configs/candidates.yaml \
        --train-csv data/demo/gemspot_train.csv \
        --val-csv data/demo/gemspot_val.csv \
        --tracking-uri http://localhost:5000

# Run quality gate
docker run --rm --network host \
    -e MLFLOW_TRACKING_URI=http://localhost:5000 \
    gemspot-training:latest \
    python scripts/quality_gate.py \
        --tracking-uri http://localhost:5000

# Check MLflow UI
open http://localhost:5000
```

### In-Cluster Test

```bash
# Create a one-off Job from the CronJob
kubectl create job --from=cronjob/training test-training -n gemspot-staging

# Watch logs
kubectl logs -f job/test-training -n gemspot-staging

# Check if model was registered
kubectl exec -it deployment/mlflow -n gemspot-staging -- \
    mlflow models list --tracking-uri http://localhost:5000
```

---

## 9. Troubleshooting

### "No transformed data found in MinIO"
The batch pipeline hasn't run yet, or its output was cleaned up.
- Check: `aws s3 ls s3://agent-datalake/transformed/ --endpoint-url http://minio:9000`
- Fix: Run batch pipeline first, or use `--skip-batch` with pre-existing data

### "Experiment not found"
MLflow experiment doesn't exist yet.
- Fix: The training script creates it automatically on first run. Make sure `MLFLOW_TRACKING_URI` is correct.

### "No runs passed quality gates"
All model candidates performed below thresholds. This is not a crash — the system keeps the existing production model.
- Check: MLflow UI → experiment "GemSpot-WillVisit" → see metrics
- Common cause: too little training data (batch pipeline produced few rows)

### "MLflow connection refused"
- Check: `kubectl get svc mlflow-svc -n gemspot-prod`
- Check: `kubectl port-forward svc/mlflow-svc 5000:5000` and test locally

### Training takes too long
- Normal for 300K rows: ~2-5 minutes
- If >30 minutes: increase CPU/memory limits in CronJob
- Use `--skip-batch` if batch pipeline already ran

### Model registered but serving doesn't pick it up
- ML Serving checks MLflow for stage=`Production`, not `Staging`
- The canary monitor (`scripts/canary_monitor.py`) handles Staging→Production promotion
- Or manually promote: MLflow UI → Models → GemSpotWillVisit → Transition to Production

---

## File Reference

```
GemSpot_training/
├── Dockerfile                              ← BUILD THIS
├── requirements.txt                        ← Python deps (joblib, mlflow, xgboost, etc.)
├── Makefile                                ← Local dev shortcuts
├── configs/
│   └── candidates.yaml                     ← Model candidates & hyperparameters
├── src/
│   ├── train.py                            ← Core training (called by retrain_pipeline)
│   └── gemspot_training/
│       ├── data.py                         ← Data loading, 85-feature engineering
│       ├── training.py                     ← Pipeline builder, metrics computation
│       └── utils.py                        ← Env info, git SHA collection
├── scripts/
│   ├── retrain_pipeline.py                 ← ★ MAIN ENTRY POINT for CronJob
│   ├── quality_gate.py                     ← Evaluates & registers passing models
│   ├── export_training_data.py             ← Iceberg/MinIO → train.csv + val.csv
│   ├── split_dataset.py                    ← Time-based split (for initial data only)
│   ├── make_demo_dataset.py                ← Synthetic data (for testing only)
│   └── export_run_table.py                 ← Generate run comparison table
└── data/demo/
    └── initial_training_set.csv            ← Initial 337K-row dataset (optional)
```
