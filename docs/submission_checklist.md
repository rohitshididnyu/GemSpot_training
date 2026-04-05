# Submission Checklist

Use this checklist before you submit anything to Gradescope.

## A. Joint Team Artifacts

These are shared with your teammates.

- `interfaces/will_visit_input.sample.json`
- `interfaces/will_visit_output.sample.json`

What these files do:

- the data teammate produces something shaped like the input JSON
- the training teammate trains on data derived from that schema
- the inference teammate serves predictions shaped like the output JSON

## B. Your Training Deliverables

### 1. Written document (PDF)

Create a PDF called `gemspot_training_report.pdf`.

Include:

- your name and role: training
- short project description (GemSpot predicts will_visit for user-place pairs)
- MLflow URL: `http://YOUR_FLOATING_IP:8000`
- training runs table with 3 rows (baseline, xgboost_v1, xgboost_v2)
- which candidate is best: xgboost_v2 (highest avg precision, best recall, handles imbalance)
- next experiment you would try (e.g., hyperparameter tuning with Optuna, more features)

### 2. Repository artifacts

Make sure these exist in your repo:

- `Dockerfile`
- `requirements.txt`
- `src/train.py`
- `src/gemspot_training/data.py`
- `src/gemspot_training/training.py`
- `src/gemspot_training/utils.py`
- `configs/candidates.yaml`

Also helpful:

- `scripts/split_dataset.py`
- `scripts/run_training_container.sh`
- `scripts/export_run_table.py`

### 3. Demo video (2-4 minutes)

Record a screen capture showing:

- MLflow UI live on Chameleon (`http://YOUR_FLOATING_IP:8000`)
- `docker ps` showing MLflow container running
- training container starting (`bash scripts/run_training_container.sh`)
- training output for 3 candidates (baseline, xgboost_v1, xgboost_v2)
- the 3 runs appearing in MLflow with metrics and artifacts

### 4. Live service

Keep the MLflow server live on Chameleon until April 7 midnight.

What course staff should be able to do:

- open MLflow in a browser
- inspect runs (3 total: baseline, xgboost_v1, xgboost_v2)
- compare metrics
- download artifacts

## C. What Must Be True To Get Credit

Make sure all of these are true:

- all training runs were executed on Chameleon (not locally)
- training ran inside Docker containers
- every run is tracked in MLflow
- you used one configurable training script (`src/train.py`), not separate scripts
- you logged params, metrics, cost metrics, and environment info for each run
- you included a simple baseline plus stronger XGBoost candidates
- the data was split by time (train on older data, validate on newer data)
- system metrics (cpu, memory, disk) are logged

## D. Gradescope Upload Guide

### Q1: Joint Responsibilities (1 point)
- Upload `interfaces/will_visit_input.sample.json`
- Upload `interfaces/will_visit_output.sample.json`

### Q2.1: Training Runs Table (4 points)
- Upload `gemspot_training_report.pdf` with the runs table
- Verify all MLflow links are clickable

### Q2.2: Repository Artifacts (3 points)
- Upload `Dockerfile`
- Upload `train.py`, `data.py`, `training.py`, `utils.py`, `candidates.yaml`, `requirements.txt`

### Q2.3: Demo Video (1 point)
- Upload video file or paste Google Drive / YouTube link

### Q2.4: Live MLflow Service (1 point)
- Paste: `http://YOUR_FLOATING_IP:8000`

### Free-text box (if available)

```
MLflow URL: http://YOUR_FLOATING_IP:8000
Repository: https://github.com/rohitshididnyu/GemSpot_training.git
Commit SHA: [run git rev-parse --short HEAD]
Best candidate: xgboost_v2 — highest avg precision (0.965) and recall (99.5%),
handles 82/18 class imbalance with scale_pos_weight=4.7.
```

## E. Bonus: Ray Distributed Training (ML6.2)

For bonus credit, include evidence of Ray-based distributed training:

### What to submit

- Screenshots of:
  - Chameleon bare metal instance running
  - `docker ps` showing Ray cluster (4 containers)
  - Ray Dashboard (`http://RAY_IP:8265`) showing cluster and completed job
  - MinIO Console (`http://RAY_IP:9001`) showing checkpoint files
  - MLflow showing Ray Tune experiment runs
  - Terminal output with best trial summary
  - (Optional) Fault tolerance demo: worker killed and training recovered

- In your PDF report, add a "Bonus" section explaining:
  - What Ray does (distributed hyperparameter tuning with ASHA)
  - How checkpointing and fault tolerance work
  - Best trial results compared to your main xgboost_v2

### Live services for bonus

Keep these running until April 7 midnight alongside MLflow:
- Ray Dashboard: `http://RAY_IP:8265`
- MinIO Console: `http://RAY_IP:9001`

## F. Final 5-Minute Sanity Check

Right before submission, verify:

```bash
docker ps
```

You should see `gemspot-mlflow-proj10` and `mlflow-postgres-proj10`.

Open the MLflow URL in a browser and confirm:

- the page loads
- experiment `GemSpot-WillVisit` is visible
- 3 runs are present (baseline, xgboost_v1, xgboost_v2)
- artifact links work
- system metrics tab shows cpu/memory data

Then check your repo tree:

```bash
find . -maxdepth 3 -type f | sort
```
