# Running GemSpot Locally — Full Guide

This guide walks through running the **full GemSpot training pipeline on your local machine**. It covers the very first model creation, inspecting results, and running the new **incremental retrain** workflow on `initial_training_set_new.csv`.

---

## 📦 Prerequisites

You need one of the following:
- **Docker** (recommended — matches the Chameleon environment exactly)
- **Python 3.11+** with `pip` (if you prefer a native run)

Project root: `/Users/rohitshidid/Documents/New project/`

---

## 🗂️ Data Files at a Glance

| File | Purpose | Rows |
|---|---|---|
| `data/demo/gemspot_train.csv` | Training set | 303,117 |
| `data/demo/gemspot_val.csv` | Validation set | 34,581 |
| `data/demo/initial_training_set.csv` | Original full raw dump | 1,048,574 |
| `data/demo/initial_training_set_old.csv` | Earlier version of the dump | 337,697 |
| `data/demo/initial_training_set_new.csv` | **NEW data for retraining** | 51,000 |

### 📐 Canonical Schema (locked)

Every input CSV must have **exactly these 10 columns** — nothing more, nothing less:

```
user_id          gmap_id                   time
category_encoded avg_rating                location_popularity
destination_vibe_tag  user_total_visits    user_personal_preferences
will_visit (target)
```

- **Missing columns** → hard error (`data.py:enforce_canonical_schema`)
- **Extra columns** → silently dropped with a warning
- The schema is declared in `configs/candidates.yaml` under `dataset.canonical_columns`

---

## ✂️ Splitting the Dataset

Before training, you need `gemspot_train.csv` + `gemspot_val.csv` (and optionally `gemspot_test.csv`).
`scripts/split_dataset.py` supports **three split modes** and an optional **3-way split**.

### 🧭 Which mode should I use?

| Mode | When to use | Keeps class ratio? | Uses row order? |
|---|---|:---:|:---:|
| `time` | Production-like: train on past, test on future | No | Yes (by date) |
| `percent` | General ML benchmarking | ✅ Stratified | No (shuffles) |
| `sequential` | Data is already in the right order — take top 80%, bottom 20% | No | ✅ Linear |

### Mode 1 — Time-based split (default, production-like)

Rows **before** the cutoff go to train; **on or after** go to val.

```bash
cd "/Users/rohitshidid/Documents/New project"

python3 scripts/split_dataset.py \
  --input     data/demo/initial_training_set.csv \
  --output-dir data/demo \
  --mode      time \
  --cutoff    2021-05-01
```

### Mode 2 — Percentage split (random, stratified)

Stratified on `will_visit` — class ratio is preserved in both outputs.

```bash
# 80% train, 20% val (stratified on will_visit)
python3 scripts/split_dataset.py \
  --input      data/demo/initial_training_set.csv \
  --output-dir data/demo \
  --mode       percent \
  --train-pct  0.80 \
  --seed       42
```

### Mode 3 — Sequential (linear positional) split 🆕

**No shuffling, no date math.** The first `train_pct` of rows (in CSV order) → train, rest → val.

This is what you asked for: *"bottom top 80% is training, remaining is val/test."*

```bash
# First 80% of rows → train, last 20% → val
python3 scripts/split_dataset.py \
  --input      data/demo/initial_training_set.csv \
  --output-dir data/demo \
  --mode       sequential \
  --train-pct  0.80
```

⚠️ **Caveat of sequential mode:** Because rows are not shuffled, if the CSV
has any ordering pattern (e.g. first half = one user, second half = another),
the class ratios in train/val may drift. On a 1,000-row test it produced
84.4% positive in train and 84.5% in val — which is fine *if* the CSV is
already well-mixed. If in doubt, use `percent` mode.

### 🆕 3-way split: Train / Val / Test

Add `--test-pct` to get a **third output file** (`gemspot_test.csv`).

You're right that ML best practice is 3-way split: train on train, tune hyperparameters on val, and evaluate final generalization on a never-seen test set. Our earlier 2-way was a simplification.

```bash
# SEQUENTIAL 70 / 15 / 15 — top 70% train, middle 15% val, bottom 15% test
python3 scripts/split_dataset.py \
  --input      data/demo/initial_training_set.csv \
  --output-dir data/demo \
  --mode       sequential \
  --train-pct  0.70 \
  --test-pct   0.15
```

```bash
# PERCENT (stratified) 70 / 15 / 15 — class ratios preserved in all three
python3 scripts/split_dataset.py \
  --input      data/demo/initial_training_set.csv \
  --output-dir data/demo \
  --mode       percent \
  --train-pct  0.70 \
  --test-pct   0.15
```

```bash
# TIME-based 3-way — past = train, recent = val, most-recent = test
python3 scripts/split_dataset.py \
  --input      data/demo/initial_training_set.csv \
  --output-dir data/demo \
  --mode       time \
  --cutoff     2021-05-01 \
  --test-pct   0.10
```

Outputs when `--test-pct > 0`:

```
data/demo/gemspot_train.csv
data/demo/gemspot_val.csv
data/demo/gemspot_test.csv   ← NEW
```

### 📊 Using the test set downstream

After a 3-way split, the workflow becomes:

```bash
# 1. Tune / compare candidates on VAL
python3 src/train.py \
  --config  configs/candidates.yaml \
  --train-csv data/demo/gemspot_train.csv \
  --val-csv   data/demo/gemspot_val.csv

# 2. FINAL evaluation on TEST (only once the best candidate is chosen)
python3 src/retrain.py \
  --config    configs/candidates.yaml \
  --candidate xgboost_v2 \
  --val-csv   data/demo/gemspot_test.csv \
  --new-data-csv data/demo/gemspot_train.csv \
  --old-model artifacts/models/xgboost_v2.joblib \
  --improvement-threshold 0.001
```

### Optional — Downsample for small laptops

Adds an extra stratified downsample AFTER the split:

```bash
python3 scripts/split_dataset.py \
  --input       data/demo/initial_training_set.csv \
  --mode        percent --train-pct 0.85 \
  --max-train-rows 100000 \
  --max-val-rows    10000
```

### Custom output names (e.g. to avoid overwriting)

```bash
python3 scripts/split_dataset.py \
  --input       data/demo/initial_training_set_new.csv \
  --mode        percent --train-pct 0.85 \
  --train-out-name gemspot_train_v2.csv \
  --val-out-name   gemspot_val_v2.csv
```

### Run the split inside Docker (no local Python needed)

```bash
docker run --rm -v "$(pwd):/app" -w /app \
  gemspot-train \
  python3 scripts/split_dataset.py \
    --input data/demo/initial_training_set.csv \
    --mode  percent --train-pct 0.85
```

### 📋 Split CLI Reference

| Flag | Meaning | Default |
|---|---|---|
| `--input` | Raw CSV to split | `data/demo/initial_training_set.csv` |
| `--output-dir` | Where to write outputs | `data/demo` |
| `--mode` | `time`, `percent`, or `sequential` | `time` |
| `--cutoff` | (time mode) date boundary | `2021-05-01` |
| `--train-pct` | (percent/sequential) fraction for training | `0.85` |
| `--test-pct` | Fraction for TEST set (0 = no test, 2-way split) | `0.0` |
| `--seed` | Random seed (percent + downsample) | `42` |
| `--target-col` | Column used for stratification | `will_visit` |
| `--max-train-rows` | Cap train size (stratified downsample) | none |
| `--max-val-rows` | Cap val size (stratified downsample) | none |
| `--max-test-rows` | Cap test size (stratified downsample) | none |
| `--train-out-name` | Output filename for train | `gemspot_train.csv` |
| `--val-out-name` | Output filename for val | `gemspot_val.csv` |
| `--test-out-name` | Output filename for test | `gemspot_test.csv` |

---

## 🧊 Parquet / MinIO Datalake Workflow 🆕

In production the training data doesn't live as CSVs — it lives as **parquet files** inside an Iceberg table on MinIO:

```
s3://agent-datalake/iceberg/adventurelog.db/training_data/data/
```

The upstream `batch_pipeline` job does the following every run:
1. Reads last week's accumulated training set
2. Appends the week's newly-collected rows
3. Re-computes a chronological `split` column (`train` / `eval`) so that
   **eval rows are always the most recent** — this prevents data leakage
4. Writes the whole thing as a new parquet file with a unique name like
   `00000-0-<uuid>.parquet`

### 📥 Schema of the parquet (what the pipeline produces)

| Column | Canonical equivalent used by our models |
|---|---|
| `user_id` | `user_id` |
| `location_id` | → renamed to `gmap_id` |
| `user_total_visits` | `user_total_visits` |
| `location_avg_rating` | → renamed to `avg_rating` |
| `location_popularity` | `location_popularity` |
| `user_personal_preferences` | `user_personal_preferences` |
| `location_vibe_tags` | → renamed to `destination_vibe_tag` |
| `location_category_encoded` | → renamed to `category_encoded` |
| `will_visit` | `will_visit` (target) |
| `split` | train/eval partition flag (dropped from features) |
| `pipeline_run_date` | metadata, logged to MLflow (dropped from features) |
| `snapshot_id` | metadata (dropped) |

The renames happen automatically via `configs/candidates.yaml → column_rename`.

### 🧠 Where does the parquet file come from?

| Environment | Source |
|---|---|
| **Local (this laptop)** | `data/demo/00000-0-*.parquet` (latest by mtime) |
| **Production (Chameleon)** | `s3://agent-datalake/iceberg/adventurelog.db/training_data/data/` (latest by S3 `LastModified`) |

Both are handled by the same `--parquet-path` flag — the script figures out which one you mean.

### 🚀 Train from parquet (local)

```bash
cd "/Users/rohitshidid/Documents/New project"

# The directory form — picks the most recent .parquet automatically
docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  gemspot-train \
  python3 src/train.py \
    --config       configs/candidates.yaml \
    --parquet-path data/demo/ \
    --artifact-dir artifacts/models
```

Or point at a specific file:

```bash
docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  gemspot-train \
  python3 src/train.py \
    --config       configs/candidates.yaml \
    --parquet-path data/demo/00000-0-d1e71472-fd9a-452e-9b9c-ea9388c6e209.parquet
```

### 🚀 Train from parquet (production, MinIO)

```bash
export AWS_ACCESS_KEY_ID=<minio_key>
export AWS_SECRET_ACCESS_KEY=<minio_secret>
export AWS_ENDPOINT_URL=http://<minio-host>:9000
export AWS_REGION=us-east-1

# Run with AWS credentials injected, and pip install s3 dependencies automatically
docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_ENDPOINT_URL=$AWS_ENDPOINT_URL \
  -e AWS_REGION=$AWS_REGION \
  gemspot-train \
  bash -c "pip install --no-cache-dir boto3 s3fs && python3 src/train.py \
    --config       configs/candidates.yaml \
    --parquet-path s3://agent-datalake/iceberg/adventurelog.db/training_data/data/"
```

### 🔁 Retrain from parquet (local)

All decision knobs shown explicitly (even the ones equal to argparse defaults)
so you can tune them without reading the Python source:

```bash
# Default: use ALL rows where split == 'train' (cumulative set)
docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  gemspot-train \
  python3 src/retrain.py \
    --config       configs/candidates.yaml \
    --candidate    xgboost_v2 \
    --parquet-path data/demo/ \
    --old-model    artifacts/models/xgboost_v2.joblib \
    --parquet-retrain-scope all_train \
    --additional-rounds     300 \
    --primary-metric        roc_auc \
    --improvement-threshold 0.001
```

Use **only the new week's rows** instead of the cumulative set:

```bash
docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  gemspot-train \
  python3 src/retrain.py \
    --config       configs/candidates.yaml \
    --candidate    xgboost_v2 \
    --parquet-path data/demo/ \
    --parquet-retrain-scope latest_run_only \
    --old-model    artifacts/models/xgboost_v2.joblib \
    --additional-rounds     300 \
    --primary-metric        roc_auc \
    --improvement-threshold 0.001
```

#### What each retrain flag controls

| Flag | Default | Purpose |
|---|---|---|
| `--parquet-retrain-scope` | `all_train` | `all_train` = use every `split=='train'` row (cumulative). `latest_run_only` = only the latest `pipeline_run_date` delta. |
| `--additional-rounds` | `50` | How many new boosting trees XGBoost adds on top of the old booster. Low values (50) barely move the needle; use **300** for a real retrain. |
| `--primary-metric` | `roc_auc` | Which metric the keep/reject gate compares. Options: `roc_auc`, `f1`, `accuracy`, `average_precision`. |
| `--improvement-threshold` | `0.001` | Minimum Δ in the primary metric to accept the new model. Raise to `0.005`–`0.01` if you see the train/eval distributions are very different (e.g. parquet has train=83% pos / eval=48% pos — small ROC-AUC swings are noise). |

### 🔁 Retrain from parquet (production, MinIO)

```bash
export AWS_ACCESS_KEY_ID=<minio_key>
export AWS_SECRET_ACCESS_KEY=<minio_secret>
export AWS_ENDPOINT_URL=http://<minio-host>:9000
export AWS_REGION=us-east-1

# Run with AWS credentials injected, and pip install s3 dependencies automatically
docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_ENDPOINT_URL=$AWS_ENDPOINT_URL \
  -e AWS_REGION=$AWS_REGION \
  gemspot-train \
  bash -c "pip install --no-cache-dir boto3 s3fs && python3 src/retrain.py \
    --config       configs/candidates.yaml \
    --candidate    xgboost_v2 \
    --parquet-path s3://agent-datalake/iceberg/adventurelog.db/training_data/data/ \
    --old-model    artifacts/models/xgboost_v2.joblib \
    --improvement-threshold 0.001"
```

### 🎯 Which data should the retrain actually use?

You asked this directly. Here's the decision tree:

| Scope flag | What it uses | When to pick it |
|---|---|---|
| `all_train` (default) | **All rows where `split=='train'`** (whole cumulative history) | Weekly full retrain — safest, highest quality, matches the "train on everything you know" philosophy |
| `latest_run_only` | **Only rows where `pipeline_run_date == latest`** (just this week's delta) | Fast incremental learning; use when the weekly batch is large and you want the fastest possible update |

**Our default is `all_train`.** Here's why:

1. **XGBoost's incremental mode (`xgb_model=old_booster`) already reuses the old trees** — feeding it the full `split=='train'` set doesn't re-learn them from scratch, it just grows new trees that further reduce residual error over the whole history. This is both fast AND thorough.

2. **Fresh-only training (`latest_run_only`) risks catastrophic forgetting** — the new trees would only see this week's data, so the booster might optimize for the delta at the expense of accuracy on older examples.

3. **The parquet is reasonably small (~50k rows right now)** — the speed difference between "all train" and "latest only" is seconds.

4. **The `split=='eval'` rows are the pipeline's chronologically newest slice**, specifically held out by the upstream pipeline to prevent data leakage. We always use them (and only them) to score old vs new.

If you ever see retrain's ROC-AUC drop because the eval class balance is very different from the train class balance (e.g. train=83% positive, eval=48% positive like in this parquet), that's a sign to:
- Re-visit the pipeline's split strategy upstream, OR
- Use a calibration-aware metric like `average_precision` as the gate (`--primary-metric average_precision`).

---

## 🐳 Option A — Local Run with Docker (Recommended)

### 1. Build the image (one-time)

```bash
cd "/Users/rohitshidid/Documents/New project"
docker build -t gemspot-train -f Dockerfile .
```

### 2. First-time model creation

This trains all 4 candidates from scratch on `gemspot_train.csv` and logs them to a **local MLflow** (mounted `mlruns/` folder).

```bash
docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  gemspot-train \
  python3 src/train.py \
    --config configs/candidates.yaml \
    --train-csv data/demo/gemspot_train.csv \
    --val-csv  data/demo/gemspot_val.csv \
    --artifact-dir artifacts/models
```

**What you get afterwards:**
- `artifacts/models/baseline.joblib`
- `artifacts/models/hist_gradient_boosting.joblib`
- `artifacts/models/xgboost_v1.joblib`
- `artifacts/models/xgboost_v2.joblib`  ← our best model
- `mlruns/<experiment-id>/...` ← MLflow run metadata

### 3. Inspect results in MLflow UI

```bash
docker run --rm -p 5001:5000 \
  -v "$(pwd)/mlruns:/mlruns" \
  ghcr.io/mlflow/mlflow:v2.15.1 \
  mlflow ui --backend-store-uri /mlruns --host 0.0.0.0
```

Open [http://localhost:5001](http://localhost:5001) in your browser.

### 4. Incremental retraining on new data

This loads `xgboost_v2.joblib`, continues training on `initial_training_set_new.csv`, and only saves the new model **if its ROC-AUC improves by at least 0.001**.

```bash
docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  gemspot-train \
  python3 src/retrain.py \
    --config configs/candidates.yaml \
    --candidate xgboost_v2 \
    --new-data-csv data/demo/initial_training_set_new.csv \
    --val-csv      data/demo/gemspot_val.csv \
    --old-model    artifacts/models/xgboost_v2.joblib \
    --artifact-dir artifacts/models \
    --backup-dir   artifacts/models/_backup \
    --improvement-threshold 0.001 \
    --primary-metric roc_auc \
    --additional-rounds 300
```

**Expected output** (last lines):

```
============================================================
COMPARISON on roc_auc:
  OLD model:       0.796543
  RETRAINED model: 0.798921
  Delta:           +0.002378   (threshold: +0.001)
  Decision:        KEEP new model ✅
============================================================
```


## 🐛 Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'xgboost'` | Run via Docker, OR `pip install -r requirements.txt` |
| `ModuleNotFoundError: gemspot_training` | Set `PYTHONPATH=$(pwd)/src` |
| `Old model not found at artifacts/...` | Run the first-time training (Option A/B Step 2) first |
| MLflow UI shows no runs | Confirm `MLFLOW_TRACKING_URI` matches between training + UI command |
| Out-of-memory on 8 GB laptop | Edit `configs/candidates.yaml` → comment out `hist_gradient_boosting` and keep only XGBoost candidates |
| `Input CSV is missing N required canonical columns` | Your CSV doesn't match the 10-column canonical schema. Use `scripts/split_dataset.py` or regenerate the CSV with exactly the required columns. |
| `[schema] Dropping N non-canonical column(s): ['price', …]` | Info message — extra columns (like `price` or `category`) are silently removed to keep the schema locked. Not an error. |

---

## 📁 Files Touched by This Workflow

```
configs/candidates.yaml            ← models + drop_columns (now drops 'price')
src/train.py                       ← first-time trainer
src/retrain.py                     ← NEW: incremental retrainer + gate
src/gemspot_training/data.py       ← hardened schema alignment
src/gemspot_training/training.py   ← model factory
artifacts/models/*.joblib          ← trained models
artifacts/models/_backup/*.joblib  ← backups of replaced models
mlruns/                            ← MLflow tracking store
```

---

## 🎯 Summary of the Retrain Gate Logic

```
┌──────────────────────────────────────────────────────────┐
│  RETRAIN DECISION FLOW                                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   1. Load old model   (artifacts/models/xgboost_v2)      │
│          │                                               │
│          ▼                                               │
│   2. Evaluate on val CSV → old_roc_auc                   │
│          │                                               │
│          ▼                                               │
│   3. Incremental fit on NEW CSV (xgb_model=old_booster)  │
│          │                                               │
│          ▼                                               │
│   4. Evaluate on same val CSV → new_roc_auc              │
│          │                                               │
│          ▼                                               │
│   5. delta = new_roc_auc - old_roc_auc                   │
│          │                                               │
│          ▼                                               │
│   6. if delta >= threshold (default 0.001):              │
│          • Back up old model to _backup/                 │
│          • Save new model, log to MLflow                 │
│      else:                                               │
│          • Keep old model on disk                        │
│          • Log rejected candidate for traceability       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```
