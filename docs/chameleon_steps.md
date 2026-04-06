# Chameleon Steps For GemSpot Training

This guide covers every step to run GemSpot training on Chameleon Cloud.

## 1. What You Are Building

Your goal is to show that:

- you can run training inside a Docker container on Chameleon
- each training run is tracked in MLflow
- you have a baseline and two tuned XGBoost candidates
- you can compare quality and cost tradeoffs

The model predicts: given a user and a candidate place, will the user visit? (`will_visit` = 0 or 1)

Three candidates are trained from one config file:

- `baseline` — naive dummy that always predicts the majority class
- `xgboost_v1` — XGBoost with default params (no imbalance handling)
- `xgboost_v2` — tuned XGBoost with `scale_pos_weight=4.7` for class imbalance

## 2. What Resources To Create

Use one VM for both MLflow and training (simplest path for this project).

- small or medium general-purpose VM at KVM@TACC
- persistent storage volume attached (for MLflow data)
- one floating IP
- name example: `gemspot-mlflow-server-proj10`

Important:

- put your project suffix (`proj10`) at the end of ALL resource names
- only keep instances running while you are actively working
- use only one floating IP

## 3. Launch MLflow First

Create these Chameleon resources:

- security groups: `gemspot-allow-ssh-proj10`, `gemspot-allow-8000-proj10`
- volume: `gemspot-mlflow-persist-proj10` (10 GiB)
- instance: `gemspot-mlflow-server-proj10` (Ubuntu-22.04, m1.medium)
- floating IP: associate to the instance, write it down as `MLFLOW_IP`

### SSH into the server

```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@129.114.24.253
```

### Format and mount persistent volume

```bash
lsblk
sudo mkfs.ext4 /dev/vdb
sudo mkdir -p /mnt/mlflow-proj10
sudo mount /dev/vdb /mnt/mlflow-proj10
sudo chown -R $USER:$USER /mnt/mlflow-proj10
```

### Install Docker

```bash
curl -sSL https://get.docker.com/ | sudo sh
sudo groupadd -f docker
sudo usermod -aG docker "$USER"
newgrp docker
docker run hello-world
```

### Start PostgreSQL and MLflow

```bash
mkdir -p /mnt/mlflow-proj10/postgres-data /mnt/mlflow-proj10/mlflow-artifacts

docker run -d --restart unless-stopped --name mlflow-postgres-proj10 \
  -e POSTGRES_USER=mlflow -e POSTGRES_PASSWORD=mlflow -e POSTGRES_DB=mlflow \
  -v /mnt/mlflow-proj10/postgres-data:/var/lib/postgresql/data \
  -p 5432:5432 postgres:15

# Wait 10 seconds, then start MLflow
docker run -d --restart unless-stopped --name gemspot-mlflow-proj10 \
  --link mlflow-postgres-proj10:postgres -p 8000:5000 \
  -v /mnt/mlflow-proj10/mlflow-artifacts:/mlflow/artifacts \
  ghcr.io/mlflow/mlflow:v2.12.2 \
  mlflow server --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow \
    --default-artifact-root /mlflow/artifacts --host 0.0.0.0 --port 5000
```

Verify:

```bash
docker ps          # Both containers should show "Up"
curl http://127.0.0.1:8000   # Should return HTML
```

Open in browser: `http://MLFLOW_IP:8000`

## 4. Prepare Training on the Same Machine

### Install git and clone repo

```bash
sudo apt update && sudo apt install -y git
cd ~
git clone https://github.com/rohitshididnyu/GemSpot_training.git gemspot
cd gemspot
```

### Upload the real dataset

From your Mac, upload `initial_training_set.csv` to the server:

```bash
scp -i ~/.ssh/gemspot-key-proj10.pem \
  "/Users/rohitshidid/Documents/New project/data/demo/initial_training_set.csv" \
  ubuntu@MLFLOW_IP:~/gemspot/data/demo/
```

### Build Docker image

```bash
export PROJECT_SUFFIX=proj10
docker build -t gemspot-train-${PROJECT_SUFFIX} .
```

### Split dataset by time (train on old data, validate on new)
### fixed 
```bash 
docker run --rm -v "$(pwd):/app" gemspot-train-${PROJECT_SUFFIX} python3 scripts/split_dataset.py 
```

This creates:

- `data/demo/gemspot_train.csv` — 303,117 rows (before May 2021)
- `data/demo/gemspot_val.csv` — 34,581 rows (May 2021 onward)


Verify data exists:

```bash
ls -lh ~/gemspot/data/demo/
# Should see gemspot_train.csv and gemspot_val.csv
```



## 5. Run Training

### START SCREEN RECORDING ON YOUR MAC NOW

```bash
cd ~/gemspot
export PROJECT_SUFFIX=proj10
PRIVATE_IP=$(hostname -I | awk '{print $1}')
export MLFLOW_TRACKING_URI=http://${PRIVATE_IP}:8000
unset DOCKER_EXTRA_ARGS
bash scripts/run_training_container.sh
```

### retraining
```bash
cd ~/gemspot && git pull  
export PROJECT_SUFFIX=proj10
PRIVATE_IP=$(hostname -I | awk '{print $1}')
export MLFLOW_TRACKING_URI=http://${PRIVATE_IP}:8000
bash scripts/run_training_container.sh
```

This trains all 3 candidates (baseline, xgboost_v1, xgboost_v2) and logs to MLflow.

### STOP SCREEN RECORDING

## 6. Verify in MLflow

Open `http://MLFLOW_IP:8000` and check:

- experiment `GemSpot-WillVisit`
- 3 runs (baseline, xgboost_v1, xgboost_v2)
- parameters, metrics, artifacts for each run
- system metrics (cpu, memory, disk, network)

## 7. Export Run Table

```bash
cd ~/gemspot
docker run --rm \
  -v "$(pwd):/app" \
  -e MLFLOW_TRACKING_URI=http://${PRIVATE_IP}:8000 \
  gemspot-train-proj10 \
  python scripts/export_run_table.py \
    --experiment-name GemSpot-WillVisit \
    --tracking-uri http://${PRIVATE_IP}:8000 \
    --ui-base-url http://MLFLOW_IP:8000
```

Copy the Markdown table output for your PDF report.

## 8. What To Say If Asked Why These Models

- GemSpot uses structured tabular data (user preferences, place metadata, category encodings)
- XGBoost is the state-of-the-art for tabular classification
- the dummy baseline establishes a lower bound (ROC-AUC = 0.50)
- xgboost_v1 proves the model architecture works with defaults
- xgboost_v2 improves on v1 by handling the 82/18 class imbalance with `scale_pos_weight=4.7`
- the time-based split ensures we validate on future data, not random samples
- MLflow lets us compare quality against runtime cost
