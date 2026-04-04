# ML6 Start To Finish For GemSpot

This file adapts the `ML6.pdf` lab to the GemSpot training subsystem.

Use this when you want the main graded workflow:

- live MLflow service on Chameleon
- training code running from a container
- runs visible in MLflow
- demo video and report material

## 1. Before Lease Time

Prepare these values:

- your project suffix, for example `proj99`
- your MLflow floating IP from `mlflow_setup.pdf`
- your repo URL

Your MLflow service should already exist from [mlflow_setup.pdf](/Users/rohitshidid/Downloads/mlflow_setup.pdf).

## 2. Bring Up Training Server Using The ML6 Pattern

Use the Chameleon Jupyter workflow from `ML6.pdf`, but rename resources so the project ID is a suffix.

Examples:

- lease: `gemspot-train-lease-proj99`
- server: `gemspot-train-server-proj99`
- security groups: `gemspot-allow-ssh-proj99`, `gemspot-allow-8888-proj99`

If you are on a GPU VM, use the same Docker and NVIDIA container toolkit steps as ML6.
1. Chameleon Security Group (Most Likely)
By default, Chameleon blocks all incoming web traffic to protect your virtual machine. If you look at Section 2 of your docs/ml6_start_to_finish.md, it mentions creating a security group called gemspot-allow-8888-proj99.

You likely need to configure this in the Chameleon Cloud Dashboard:

Go to your Chameleon Dashboard (Horizon).
Navigate to Network > Security Groups.
Create a new security group (or edit an existing one) and add an Ingress Rule for TCP on Port 8888.
Go to Compute > Instances, find your training server, and Attach this security group to the instance.

## 3. On The Training Server

SSH in:

```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@YOUR_TRAINING_FLOATING_IP
```

Clone the repo:

```bash
git clone <your-repo-url> gemspot
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

## 4. Start The ML6-Style Jupyter Container

Set your existing MLflow server address:

```bash
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_FLOATING_IP:8000
```

If you are on a GPU machine:

```bash
export DOCKER_EXTRA_ARGS="--gpus all"
bash scripts/start_ml6_jupyter_container.sh
```

If you are on a CPU machine:

```bash
unset DOCKER_EXTRA_ARGS
bash scripts/start_ml6_jupyter_container.sh
```

Get the Jupyter token:

```bash
docker exec gemspot-jupyter jupyter server list
```

Open:

```text
http://YOUR_TRAINING_FLOATING_IP:8888/lab?token=...
```

Inside the Jupyter terminal, confirm the tracking URI:

```bash
env | grep MLFLOW_TRACKING_URI
```

## 5. Generate Demo Data

Inside the Jupyter terminal:

```bash
cd /home/jovyan/work
python scripts/make_demo_dataset.py --output-dir data/demo
```

## 6. Run A Non-Tracked Smoke Test First

This mirrors ML6’s “run the model once before adding tracking” idea.

```bash
cd /home/jovyan/work
PYTHONPATH=src python src/train.py \
  --config configs/candidates.yaml \
  --train-csv data/demo/gemspot_train.csv \
  --val-csv data/demo/gemspot_val.csv \
  --experiment-name GemSpot-WillVisit-Smoke
```

If it starts correctly, stop it only if you are just doing a crash test. Otherwise let it complete.

## 7. Run The Real MLflow-Tracked Training

The training script already includes MLflow logging and system metrics.

```bash
cd /home/jovyan/work
PYTHONPATH=src python src/train.py \
  --config configs/candidates.yaml \
  --train-csv data/demo/gemspot_train.csv \
  --val-csv data/demo/gemspot_val.csv \
  --experiment-name GemSpot-WillVisit \
  --tracking-uri "${MLFLOW_TRACKING_URI}"
```

Or use the Docker runner:

```bash
cd /home/jovyan/work
export PROJECT_SUFFIX=proj10
export DOCKER_EXTRA_ARGS="--gpus all"
bash scripts/run_training_container.sh
```

## 8. Verify In MLflow

Open:

```text
http://YOUR_MLFLOW_FLOATING_IP:8000
```

Check:

- experiment `GemSpot-WillVisit`
- one run per candidate
- parameters
- metrics
- exported model artifacts
- `environment/nvidia_smi.txt` if on NVIDIA hardware
- system metrics in MLflow

## 9. Replace Demo Data With Real GemSpot Data

Once the pipeline works, switch to your actual processed CSVs:

```bash
cd /home/jovyan/work
export PROJECT_SUFFIX=proj10
export TRAIN_CSV=data/processed/gemspot_train.csv
export VAL_CSV=data/processed/gemspot_val.csv
export EXPERIMENT_NAME=GemSpot-WillVisit
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_FLOATING_IP:8000
export DOCKER_EXTRA_ARGS="--gpus all"
bash scripts/run_training_container.sh
```

## 10. What To Capture For Submission

- MLflow URL
- run table
- commit SHA
- demo video of the container run and MLflow UI
- explanation of which candidate is best and why
