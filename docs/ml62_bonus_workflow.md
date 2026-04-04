# ML6.2 Bonus Workflow For GemSpot

This file adapts `ML6.2.pdf` to a GemSpot-friendly bonus path.

The main idea is:

- keep your normal MLflow submission path from ML6
- add a Ray-based bonus path for scheduling, checkpoints, fault tolerance, and hyperparameter tuning

## 1. What Bonus Story You Can Tell

For bonus credit, your strongest story is:

- main training runs are tracked in MLflow
- bonus path uses Ray to make tuning more robust and scalable
- Ray stores checkpoints in object storage
- if a worker fails, training can resume from checkpoint
- ASHA stops weak hyperparameter trials early to save resources

## 2. What This Repo Now Includes

- `src/train_ray_xgboost.py`: Ray Train + XGBoost distributed training with checkpoints and failure retries
- `src/train_ray_tune.py`: Ray Tune + ASHA hyperparameter optimization
- `configs/ray_bonus.yaml`: Ray bonus configuration
- `ray-runtime.json`: runtime environment file for `ray job submit`
- `requirements-ray.txt`: extra dependencies for Ray
- `scripts/submit_ray_tune_job.sh`: helper to submit the bonus tuning job

## 3. Cluster Setup Like ML6.2

Follow the `ML6.2.pdf` resource setup pattern:

- reserve a node suitable for Ray
- bring up the training host
- install Docker
- clone the repo
- bring up the Ray cluster
- start a Jupyter container for job submission

Important:

- ML6.2’s example names like `node-mltrain-<username>` should be changed to names ending with your project suffix
- keep your project suffix on leases and servers

## 4. Start The Ray Cluster

Inside the training host, use the ML6.2-style commands.

If you are literally following the ML6.2 lab repository on the same host, those commands are:

```bash
export HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)
docker compose -f mltrain-chi/docker/docker-compose-ray-rocm.yaml up -d
docker ps
```

Verify workers:

```bash
docker exec ray-worker-0 "rocm-smi"
docker exec ray-worker-1 "rocm-smi"
```

If you are on NVIDIA instead of AMD, use the NVIDIA Ray compose file from the lab.

## 5. Start The Ray Jupyter Container

From the training host:

```bash
HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)
docker build -t jupyter-ray -f mltrain-chi/docker/Dockerfile.jupyter-ray .
docker run -d --rm -p 8888:8888 \
  -v ~/gemspot:/home/jovyan/work \
  -e RAY_ADDRESS=http://${HOST_IP}:8265/ \
  --name jupyter \
  jupyter-ray
docker exec jupyter jupyter server list
```

Open:

```text
http://YOUR_TRAINING_FLOATING_IP:8888/lab?token=...
```

Inside the Jupyter terminal:

```bash
env | grep RAY_ADDRESS
```

## 6. Prepare Demo Data For The Ray Job

Inside the Jupyter terminal:

```bash
cd ~/work
python scripts/make_demo_dataset.py --output-dir data/demo
```

Because `ray job submit --working-dir .` uploads the working directory, the demo CSVs can travel with the job.

## 7. Run The Ray Tune Bonus Job

Inside the Jupyter terminal:

```bash
cd ~/work
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_FLOATING_IP:8000
bash scripts/submit_ray_tune_job.sh
```

The helper script submits:

```bash
ray job submit \
  --runtime-env ray-runtime.json \
  --working-dir . \
  -- \
  python src/train_ray_tune.py \
    --config configs/ray_bonus.yaml \
    --train-csv data/demo/gemspot_train.csv \
    --val-csv data/demo/gemspot_val.csv \
    --tracking-uri http://YOUR_MLFLOW_FLOATING_IP:8000
```

## 8. What The Bonus Job Does

- samples XGBoost hyperparameters from `configs/ray_bonus.yaml`
- checkpoints the model regularly
- uses ASHA to terminate weak trials early
- keeps the best checkpoints
- writes the best model summary to `artifacts/ray_tune`
- optionally logs best-trial information back to MLflow

## 9. Optional Ray Train Robustness Demo

To show a stronger “robustness” story, run:

```bash
cd ~/work
python src/train_ray_xgboost.py \
  --config configs/ray_bonus.yaml \
  --train-csv data/demo/gemspot_train.csv \
  --val-csv data/demo/gemspot_val.csv \
  --storage-path s3://ray \
  --tracking-uri http://YOUR_MLFLOW_FLOATING_IP:8000
```

This path uses:

- `XGBoostTrainer`
- `FailureConfig(max_failures=2)`
- checkpointing to persistent storage
- distributed Ray scheduling

If you want to demonstrate recovery after failure, let the job get past at least one checkpoint interval and then restart the worker/container that is running the workload, similar to the ML6.2 lab.

## 10. Dashboards To Open

- Ray dashboard: `http://YOUR_TRAINING_FLOATING_IP:8265`
- MinIO dashboard used by the Ray cluster: `http://YOUR_TRAINING_FLOATING_IP:9001`
- MLflow UI: `http://YOUR_MLFLOW_FLOATING_IP:8000`

## 11. What To Screenshot For Bonus Evidence

- Ray dashboard showing active jobs
- MinIO bucket showing Ray checkpoints
- terminal output showing Ray trial statuses
- if you do the fault tolerance demo, before/after evidence of worker failure and recovery
- final best-trial summary

## 12. What To Say In Your Report

Suggested wording:

"For bonus work, we adapted the ML6.2 Ray workflow to GemSpot by adding a Ray-based XGBoost tuning pipeline with checkpointing, ASHA early stopping, and retry configuration. This allows the training platform to recover from failures and to evaluate more candidate configurations efficiently than a sequential search."
