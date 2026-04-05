# ML6 Start To Finish For GemSpot

This file adapts the `ML6.pdf` lab to the GemSpot training subsystem.

Use this when you want the main graded workflow:

- live MLflow service on Chameleon
- training code running from a container
- runs visible in MLflow
- demo video and report material

## 1. Before Lease Time

Prepare these values:

- your project suffix: `proj10`
- your MLflow floating IP (from the MLflow setup)
- your repo URL: `https://github.com/rohitshididnyu/GemSpot_training.git`

## 2. Bring Up Training Server Using The ML6 Pattern

Use the Chameleon workflow from `ML6.pdf`, but rename resources with your project suffix.

Examples:

- lease: `gemspot-train-lease-proj10`
- server: `gemspot-train-server-proj10`
- security groups: `gemspot-allow-ssh-proj10`, `gemspot-allow-8000-proj10`

If you are on a GPU VM, use the same Docker and NVIDIA container toolkit steps as ML6.

To configure security groups in Chameleon:

1. Go to your Chameleon Dashboard (Horizon)
2. Navigate to Network > Security Groups
3. Create security groups and add Ingress Rules for TCP on port 22 (SSH) and port 8000 (MLflow)
4. Go to Compute > Instances, find your server, and attach the security groups

## 3. On The Training Server

SSH in:

```bash
ssh -i ~/.ssh/gemspot-key-proj10.pem ubuntu@YOUR_FLOATING_IP
```

Clone the repo:

```bash
git clone https://github.com/rohitshididnyu/GemSpot_training.git gemspot
cd gemspot
```

Install Docker:

```bash
curl -sSL https://get.docker.com/ | sudo sh
sudo groupadd -f docker
sudo usermod -aG docker "$USER"
newgrp docker
docker run hello-world
```

If using a GPU VM, install the NVIDIA container toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt-get install -y nvidia-container-toolkit jq
sudo nvidia-ctk runtime configure --runtime=docker
sudo jq 'if has("exec-opts") then . else . + {"exec-opts": ["native.cgroupdriver=cgroupfs"]} end' \
  /etc/docker/daemon.json | \
  sudo tee /etc/docker/daemon.json.tmp > /dev/null
sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi
```

## 4. Upload Real Data and Build Image

Upload `initial_training_set.csv` from your Mac:

```bash
scp -i ~/.ssh/gemspot-key-proj10.pem \
  "/Users/rohitshidid/Documents/New project/data/demo/initial_training_set.csv" \
  ubuntu@YOUR_FLOATING_IP:~/gemspot/data/demo/
```

Build the Docker image:

```bash
cd ~/gemspot
export PROJECT_SUFFIX=proj10
docker build -t gemspot-train-${PROJECT_SUFFIX} .
```

## 5. Split Dataset By Time

The data spans Nov 2020 to Sep 2021. We split by time: train on older data, validate on newer data.

```bash
docker run --rm -v "$(pwd):/app" gemspot-train-${PROJECT_SUFFIX} \
  python scripts/split_dataset.py
```

Verify:

```bash
ls -la data/demo/gemspot_train.csv data/demo/gemspot_val.csv
```

Train: 303,117 rows (before May 2021). Val: 34,581 rows (May 2021 onward).

## 6. Run Training (MLflow-Tracked)

Set your MLflow server address:

```bash
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_FLOATING_IP:8000
```

If on the same machine as MLflow, use the private IP:

```bash
PRIVATE_IP=$(hostname -I | awk '{print $1}')
export MLFLOW_TRACKING_URI=http://${PRIVATE_IP}:8000
```

Run training:

```bash
unset DOCKER_EXTRA_ARGS
bash scripts/run_training_container.sh
```

This trains 3 candidates: baseline, xgboost_v1, xgboost_v2.

## 7. Verify In MLflow

Open: `http://YOUR_MLFLOW_FLOATING_IP:8000`

Check:

- experiment `GemSpot-WillVisit`
- 3 runs (baseline, xgboost_v1, xgboost_v2)
- parameters (hyperparams, environment info)
- metrics (accuracy, f1, roc_auc, average_precision, train_seconds)
- artifacts (exported model files, run_summary.json)
- system metrics (cpu, memory, disk, network)

## 8. Export Run Table

```bash
cd ~/gemspot
docker run --rm \
  -v "$(pwd):/app" \
  -e MLFLOW_TRACKING_URI=http://${PRIVATE_IP}:8000 \
  gemspot-train-proj10 \
  python scripts/export_run_table.py \
    --experiment-name GemSpot-WillVisit \
    --tracking-uri http://${PRIVATE_IP}:8000 \
    --ui-base-url http://YOUR_MLFLOW_FLOATING_IP:8000
```

## 9. What To Capture For Submission

- MLflow URL: `http://YOUR_MLFLOW_FLOATING_IP:8000`
- run table (from export_run_table.py output)
- commit SHA: `git rev-parse --short HEAD`
- demo video of the container run and MLflow UI
- explanation: xgboost_v2 is the best candidate (highest avg precision, best recall, handles imbalance)
