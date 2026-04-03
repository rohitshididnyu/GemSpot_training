# GemSpot Training Starter

This repository is a starter implementation for the GemSpot training role in the MLOps project.

It is designed to satisfy the initial implementation rubric for the training subsystem:

- one configurable training script
- multiple candidate models
- MLflow tracking for every run
- Dockerized execution
- sample interface JSON files
- beginner-friendly Chameleon run instructions

The primary model implemented here is the `will_visit` recommender/classifier described in the GemSpot proposal.

## Repository Layout

- `src/train.py`: main training entrypoint
- `src/gemspot_training/`: helper modules
- `configs/candidates.yaml`: candidate model configuration
- `scripts/make_demo_dataset.py`: creates a synthetic but representative dataset
- `scripts/start_mlflow_server.sh`: starts a persistent MLflow service in Docker
- `scripts/run_training_container.sh`: runs the training container against your MLflow server
- `interfaces/`: sample JSON input/output payloads for the shared team contract
- `docs/`: submission, Chameleon, and demo guidance

## Quick Start

1. Use Python 3.12 locally if possible.

Python 3.13 caused dependency issues during verification for this stack, while Python 3.12 worked.

2. Create a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Generate demo data:

```bash
python3 scripts/make_demo_dataset.py --output-dir data/demo
```

5. Start a local MLflow server if you want to test on your laptop:

```bash
mkdir -p artifacts/mlflow
mlflow server \
  --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db \
  --default-artifact-root artifacts/mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5000
```

6. In a new terminal, run training:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
PYTHONPATH=src python3 src/train.py \
  --config configs/candidates.yaml \
  --train-csv data/demo/gemspot_train.csv \
  --val-csv data/demo/gemspot_val.csv \
  --experiment-name GemSpot-WillVisit
```

7. Open the MLflow UI:

```text
http://127.0.0.1:5000
```

## MLflow For Grading

For the actual graded Chameleon submission, use the MLflow setup described in your course handout and [mlflow_setup.pdf](/Users/rohitshidid/Downloads/mlflow_setup.pdf):

- VM at `KVM@TACC`
- persistent block storage for PostgreSQL
- `CHI@TACC` object storage for artifacts
- MLflow UI on port `8000`

The helper script `scripts/start_mlflow_server.sh` is only a simplified local fallback for personal testing. It is not the preferred graded setup.

Also, your course instructions require the project ID to appear as a suffix in resource names. So adapt notebook resource names to look like:

- `gemspot-mlflow-server-proj99`
- `gemspot-mlflow-persist-proj99`
- `gemspot-mlflow-artifacts-proj99`

Do not use the example notebook names verbatim if they put the project ID at the front.

## Docker Workflow

Build the image:

```bash
docker build -t gemspot-train-proj99 .
```

Run training inside Docker:

```bash
docker run --rm \
  -e MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  gemspot-train-proj99 \
  python src/train.py \
    --config configs/candidates.yaml \
    --train-csv data/demo/gemspot_train.csv \
    --val-csv data/demo/gemspot_val.csv \
    --experiment-name GemSpot-WillVisit
```

## Important Course Notes

- Run graded training jobs on Chameleon, not only on your laptop.
- Track every candidate in MLflow.
- Use the same training script for all candidates; switch models and hyperparameters through config.
- Include your Chameleon project suffix in resource names, for example `gemspot-mlflow-server-proj99`.

## What To Read Next

- `docs/chameleon_steps.md`
- `docs/submission_checklist.md`
- `docs/demo_video_script.md`
- `docs/run_table_template.md`
