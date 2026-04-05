# ML6.2 Bonus: Ray Distributed Training on Chameleon Bare Metal

Complete step-by-step guide for the bonus part of the GemSpot training assignment.

## What This Demonstrates (Bonus Points)

- **Distributed training** with Ray on a Chameleon bare metal node
- **Hyperparameter tuning** with Ray Tune + ASHA early stopping (12 trials)
- **Fault tolerance** via FailureConfig (auto-retry on crash)
- **Checkpointing** to S3-compatible object storage (MinIO)
- **MLflow integration** logging every trial's metrics
- All on the same real dataset (337k rows, 85 features) used for the main ML6 submission

## Your Reservation Details

- Lease: `training_proj10`
- Node: `c11-09` (bare metal, KVM@TACC)
- Resource type: `physical:host`

## Prerequisites

Before starting, make sure:
- Your ML6 main submission is already running (MLflow server at `http://129.114.24.253:8000`)
- You have the split data files (`gemspot_train.csv`, `gemspot_val.csv`) from the main workflow
- The repo is pushed to GitHub with all Ray bonus files

---

## Step 1: Launch Bare Metal Instance

1. Go to **Chameleon Cloud > KVM@TACC > Compute > Instances**
2. Click **Launch Instance**
   - **Instance Name**: `gemspot-ray-proj10`
   - **Source**: Ubuntu 22.04 (boot from image)
   - **Flavor**: `baremetal` (your reservation is for a physical host)
   - **Networks**: `sharednet1`
   - **Scheduler Hints > Reservation**: Select `training_proj10`
3. Click **Launch**
4. Wait for status = **Active** (bare metal can take 5-10 minutes)
5. **Associate a Floating IP** to the instance
   - Go to **Network > Floating IPs > Allocate IP** (if needed)
   - Associate it to `gemspot-ray-proj10`
   - Write this down as `RAY_IP` (example: `129.114.XX.XX`)

6. **Add Security Groups** (if not already created):
   - `gemspot-allow-ssh-proj10` (port 22)
   - Create `gemspot-allow-ray-proj10` with rules:
     - TCP 8265 (Ray Dashboard)
     - TCP 9001 (MinIO Console)

**TAKE SCREENSHOT**: Instance running in Chameleon dashboard

---

## Step 2: SSH Into Bare Metal Node

```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@RAY_IP
```

Replace `RAY_IP` with your floating IP.

---

## Step 3: Install Docker

```bash
curl -sSL https://get.docker.com/ | sudo sh
sudo groupadd -f docker
sudo usermod -aG docker "$USER"
newgrp docker
docker run hello-world
```

Install Docker Compose:

```bash
sudo apt-get update && sudo apt-get install -y docker-compose-plugin
docker compose version
```

---

## Step 4: Clone Repo and Get Data

```bash
cd ~
git clone https://github.com/rohitshididnyu/GemSpot_training.git gemspot
cd gemspot
mkdir -p data/demo
```

Now copy the split dataset from your MLflow server (which already has the data).
Open a **second SSH terminal to your MLflow server** and run:

```bash
# ON THE MLFLOW SERVER (129.114.24.253):
scp ~/gemspot/data/demo/gemspot_train.csv cc@RAY_IP:~/gemspot/data/demo/
scp ~/gemspot/data/demo/gemspot_val.csv cc@RAY_IP:~/gemspot/data/demo/
```

OR copy from your Mac:

```bash
# ON YOUR MAC:
scp -i ~/.ssh/id_rsa_chameleon \
  "/Users/rohitshidid/Documents/New project/data/demo/gemspot_train.csv" \
  cc@RAY_IP:~/gemspot/data/demo/

scp -i ~/.ssh/id_rsa_chameleon \
  "/Users/rohitshidid/Documents/New project/data/demo/gemspot_val.csv" \
  cc@RAY_IP:~/gemspot/data/demo/
```

If you only have `initial_training_set.csv` locally, upload that and split on the Ray node:

```bash
# ON YOUR MAC:
scp -i ~/.ssh/id_rsa_chameleon \
  "/Users/rohitshidid/Documents/New project/data/demo/initial_training_set.csv" \
  cc@129.114.109.166:~/gemspot/data/demo/

# THEN ON THE RAY NODE:
cd ~/gemspot
docker build -t gemspot-train-proj10 .
docker run --rm -v "$(pwd):/app" gemspot-train-proj10 python3 scripts/split_dataset.py
```

Verify data exists:

```bash
ls -lh ~/gemspot/data/demo/
# Should see gemspot_train.csv and gemspot_val.csv
```

---

## Step 5: Start the Ray Cluster

```bash
cd ~/gemspot
docker compose -f docker/docker-compose-ray.yaml up -d --build
```

This starts 4 containers:
- `minio-proj10` — S3-compatible checkpoint storage
- `ray-head-proj10` — Ray head node (dashboard on port 8265)
- `ray-worker-0-proj10` — Ray worker
- `ray-worker-1-proj10` — Ray worker

Wait ~2 minutes for the build, then verify:

```bash
docker ps
```

You should see all 4 containers running. Check Ray cluster:

```bash
docker exec ray-head-proj10 ray status
```

Should show 3 nodes (head + 2 workers) with CPUs available.

**TAKE SCREENSHOT**: `docker ps` showing all 4 containers running

Open in browser:
- **Ray Dashboard**: `http://RAY_IP:8265`
- **MinIO Console**: `http://RAY_IP:9001` (login: minioadmin / minioadmin)

**TAKE SCREENSHOT**: Ray Dashboard showing cluster nodes

---

## Step 6: Run Ray Tune (Hyperparameter Search)

### START SCREEN RECORDING ON YOUR MAC NOW

Set your MLflow server IP (the one from the main ML6 submission):

```bash
export MLFLOW_IP=129.114.24.253
```

Run the Ray Tune job:

```bash
docker exec -e AWS_ACCESS_KEY_ID=minioadmin \
  -e AWS_SECRET_ACCESS_KEY=minioadmin \
  -e AWS_ENDPOINT_URL=http://minio:9000 \
  ray-head-proj10 \
  python /app/src/train_ray_tune.py \
    --config /app/configs/ray_bonus.yaml \
    --train-csv /app/data/demo/gemspot_train.csv \
    --val-csv /app/data/demo/gemspot_val.csv \
    --tracking-uri http://${MLFLOW_IP}:8000 \
    --storage-path s3://ray/checkpoints \
    --num-samples 12 \
    --cpu-per-trial 4 \
    --max-concurrent-trials 3
```

This will:
- Launch 12 hyperparameter trials
- Run max 3 at a time (across your 12 CPUs on head + workers)
- ASHA scheduler kills weak trials early
- Checkpoints saved to MinIO
- Each trial logged to MLflow

**Expected runtime**: 10-20 minutes depending on hardware.

While it runs, watch:
- **Terminal**: trial progress (shows which trials are running, stopped, completed)
- **Ray Dashboard** (`http://RAY_IP:8265`): click Jobs to see active job
- **MinIO Console** (`http://RAY_IP:9001`): navigate to `ray` bucket to see checkpoints

### STOP SCREEN RECORDING after the best trial summary prints

**TAKE SCREENSHOT**: Terminal showing final "Best Trial Summary" output
**TAKE SCREENSHOT**: Ray Dashboard showing completed job
**TAKE SCREENSHOT**: MinIO showing checkpoint files in the `ray` bucket

---

## Step 7 (Optional): Run Ray Train with Fault Tolerance Demo

This shows distributed training with automatic failure recovery:

```bash
docker exec -e AWS_ACCESS_KEY_ID=minioadmin \
  -e AWS_SECRET_ACCESS_KEY=minioadmin \
  -e AWS_ENDPOINT_URL=http://minio:9000 \
  ray-head-proj10 \
  python /app/src/train_ray_xgboost.py \
    --config /app/configs/ray_bonus.yaml \
    --train-csv /app/data/demo/gemspot_train.csv \
    --val-csv /app/data/demo/gemspot_val.csv \
    --tracking-uri http://${MLFLOW_IP}:8000 \
    --storage-path s3://ray/checkpoints \
    --num-workers 2 \
    --cpu-per-worker 4 \
    --max-failures 2
```

To **demonstrate fault tolerance** (impressive for bonus):

1. Start the training (command above)
2. Wait until you see checkpoint progress (~25 rounds)
3. In a second terminal, kill a worker: `docker restart ray-worker-0-proj10`
4. Watch the training automatically recover and continue from checkpoint
5. **TAKE SCREENSHOT**: Terminal showing recovery after worker restart

---

## Step 8: Verify in MLflow

Open `http://129.114.24.253:8000` (your MLflow server) and check:

- **Experiment**: `GemSpot-WillVisit-RayTune` — should show individual trial runs + best summary
- **Experiment**: `GemSpot-WillVisit-RayTrain` — should show the distributed training run (if you did Step 7)
- Each run has: params, metrics, artifacts (exported model + summary JSON)

**TAKE SCREENSHOT**: MLflow showing Ray Tune experiments

---

## Step 9: What To Include in Your Report

Add a bonus section to your PDF report:

```
BONUS: Ray Distributed Training (ML6.2)

We extended the GemSpot training pipeline with Ray for distributed
hyperparameter tuning on a Chameleon bare metal node (c11-09).

What we did:
- Set up a Ray cluster (1 head + 2 workers) using Docker Compose
- Used MinIO for S3-compatible checkpoint storage
- Ran Ray Tune with ASHA scheduler to search 12 XGBoost configurations
- ASHA early-stopped weak trials, saving compute resources
- Best trial achieved roc_auc=X.XXX (vs xgboost_v2's 0.836)
- Demonstrated fault tolerance: killed a worker mid-training,
  Ray automatically recovered from checkpoint and continued

Key Ray features demonstrated:
- Distributed scheduling across multiple workers
- ASHA early stopping (Adaptive Successive Halving)
- Checkpoint persistence to object storage (MinIO/S3)
- FailureConfig with max_failures=2 for automatic retry
- Full MLflow integration for experiment tracking

Dashboards:
- Ray Dashboard: http://RAY_IP:8265
- MinIO Console: http://RAY_IP:9001
- MLflow: http://129.114.24.253:8000
```

---

## Step 10: Cleanup (After Grading)

After your grade is posted:

```bash
# Stop Ray cluster
cd ~/gemspot
docker compose -f docker/docker-compose-ray.yaml down -v

# Release bare metal instance (in Chameleon dashboard)
# Delete the instance and release the floating IP
```

---

## Quick Reference: All Dashboards

| Service | URL | Purpose |
|---------|-----|---------|
| MLflow | `http://129.114.24.253:8000` | Experiment tracking (main + bonus) |
| Ray Dashboard | `http://RAY_IP:8265` | Ray cluster status, job progress |
| MinIO Console | `http://RAY_IP:9001` | Checkpoint storage browser |

## Quick Reference: Screenshots Needed

1. Chameleon dashboard showing bare metal instance running
2. `docker ps` with 4 Ray containers
3. Ray Dashboard showing cluster nodes
4. Terminal: Ray Tune final summary with best trial
5. Ray Dashboard: completed job
6. MinIO: checkpoint files in `ray` bucket
7. MLflow: Ray Tune experiments
8. (Optional) Terminal showing fault tolerance recovery
