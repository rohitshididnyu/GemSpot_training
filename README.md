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

## Use This Workflow For The Course

Because you already ran [mlflow_setup.pdf](/Users/rohitshidid/Downloads/mlflow_setup.pdf), treat the MLflow server as already provisioned.

For training and testing the model, follow the infrastructure pattern from [ML5.pdf](/Users/rohitshidid/Downloads/ML5.pdf):

- use the existing MLflow server from `mlflow_setup.pdf`
- use the `ML5` single-GPU Chameleon flow for the training machine
- use Docker containers to run and test the model
- if you are on a GPU machine, verify container GPU access with `docker run --rm --gpus all ubuntu nvidia-smi`

Important:

- `mlflow_setup.pdf` is for the tracking server
- `ML5.pdf` is for the training machine workflow
- unlike ML5, you do not need the BLIP notebooks or Jupyter container for GemSpot
- we reuse the ML5 server setup pattern, Docker pattern, and GPU verification pattern, but we run `src/train.py` from this repo instead
- this starter model is scikit-learn based, so it does not require a GPU, but it is still fine to run it on a GPU VM if that is the workflow your course expects

## What To Do Now

1. Keep the MLflow server from [mlflow_setup.pdf](/Users/rohitshidid/Downloads/mlflow_setup.pdf) running.
2. Note the MLflow URL from that notebook:

```bash
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_FLOATING_IP:8000
```

3. Bring up a training machine using the `ML5` single-GPU pattern.
4. On that training machine, follow the ML5-style container setup steps below.
5. Build this repo’s Docker image, generate demo data, and run training.
6. Refresh the MLflow UI and confirm the runs appear.

## Course-Aligned Chameleon Training Flow

This is the recommended path for your actual project work.

### A. MLflow Server

You already handled this with [mlflow_setup.pdf](/Users/rohitshidid/Downloads/mlflow_setup.pdf).

That means:

- MLflow is already running on a Chameleon VM
- it is backed by PostgreSQL on a persistent volume
- artifacts are stored in CHI@TACC object storage
- the UI is on port `8000`

Do not start a second local MLflow for grading. Reuse the one you already created.

### B. Training Server Using The ML5 Pattern

Use [ML5.pdf](/Users/rohitshidid/Downloads/ML5.pdf) as the model for how to bring up and prepare the training machine:

- bring up a Chameleon training server
- clone code onto the server
- install Docker
- install the NVIDIA container toolkit if you are using a GPU server
- verify the GPU is visible inside Docker
- build the training image
- run the model inside a container with a mounted workspace

When adapting ML5, do not keep the notebook’s default resource names like `node-llm-single-<username>`.

Your training resources should also use the project suffix rule, for example:

- lease: `gemspot-train-lease-proj99`
- server: `gemspot-train-server-proj99`
- volume, if any: `gemspot-train-data-proj99`

For GemSpot, adapt the ML5 flow like this after you SSH into the training machine:

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

If this is a GPU machine, install the NVIDIA container toolkit using the ML5 pattern:

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
```

Test GPU access exactly like ML5:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

If you are using a CPU VM instead, skip the NVIDIA toolkit and `--gpus all` commands.

### C. Build The GemSpot Image

This follows the same “build the container on the Chameleon machine” pattern as ML5:

```bash
export PROJECT_SUFFIX=proj99
docker build -t gemspot-train-${PROJECT_SUFFIX} .
```

### D. Generate Demo Data Inside The Container

This keeps the workflow container-first, which is closer to ML5:

```bash
docker run --rm \
  -v "$(pwd):/app" \
  gemspot-train-${PROJECT_SUFFIX} \
  python scripts/make_demo_dataset.py --output-dir data/demo
```

### E. Run Training Inside Docker And Log To Your Existing MLflow Server

If you are on a GPU server:

```bash
export DOCKER_EXTRA_ARGS="--gpus all"
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_FLOATING_IP:8000
bash scripts/run_training_container.sh
```

If you are on a CPU server:

```bash
unset DOCKER_EXTRA_ARGS
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_FLOATING_IP:8000
bash scripts/run_training_container.sh
```

This script now supports:

- `DOCKER_EXTRA_ARGS` for ML5-style GPU flags
- `TRAIN_CSV` and `VAL_CSV` overrides for real data
- `EXPERIMENT_NAME` override

Example with real data:

```bash
export DOCKER_EXTRA_ARGS="--gpus all"
export TRAIN_CSV=data/processed/gemspot_train.csv
export VAL_CSV=data/processed/gemspot_val.csv
export EXPERIMENT_NAME=GemSpot-WillVisit
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_FLOATING_IP:8000
bash scripts/run_training_container.sh
```

### F. Test That Everything Worked

1. Watch the training logs in the terminal.
2. Open the MLflow UI from `mlflow_setup.pdf`.
3. Verify that one run appears for each candidate in `configs/candidates.yaml`.
4. Open a run and confirm it contains:

- parameters
- metrics
- exported model artifact
- run summary JSON

### G. What From ML5 We Reused

We are reusing these parts of [ML5.pdf](/Users/rohitshidid/Downloads/ML5.pdf):

- Chameleon training server workflow
- Docker-first execution pattern
- NVIDIA container toolkit installation
- `docker run --rm --gpus all ubuntu nvidia-smi` GPU verification
- building the image directly on the Chameleon machine
- running the model from inside a container with a mounted workspace
- the general order: provision server, validate container runtime, then run the training workload

We are not reusing these parts:

- BLIP-2 model code
- Jupyter notebook training workflow
- the `single/` and `multi/` lab code itself

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
  -v "$(pwd):/app" \
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
