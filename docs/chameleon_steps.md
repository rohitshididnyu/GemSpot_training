# Chameleon Steps For GemSpot Training

This guide assumes you are using an Ubuntu image on Chameleon and you are responsible for the training role.

Important:

- for the graded MLflow service, follow the setup pattern in [mlflow_setup.pdf](/Users/rohitshidid/Downloads/mlflow_setup.pdf)
- that handout uses PostgreSQL plus CHI@TACC object storage and serves MLflow on port `8000`
- treat `mlflow_setup.pdf` as already completed if you have already run it
- follow [ML5.pdf](/Users/rohitshidid/Downloads/ML5.pdf) for the training machine workflow
- the simplified script in this repo is only for local testing
- adapt all resource names so your project ID is a suffix, even if the notebook examples use it as a prefix

## 1. What You Are Building

For the initial implementation, your goal is not a full end-to-end GemSpot product.

Your goal is to show that:

- you can run training inside a Docker container on Chameleon
- each training run is tracked in MLflow
- you have multiple model candidates, not just one model file
- you can compare quality and cost tradeoffs

For this starter, the model predicts:

- input: one user plus one candidate place
- output: whether the user is likely to visit or save that place

## 2. What Resources To Create

You may use either one or two Chameleon machines.

Fastest path:

- one VM for MLflow plus training, if your training job is lightweight

More scalable path:

- one VM for MLflow
- one separate training VM

For this starter repo, the fastest path is usually enough.

1. MLflow machine
- small or medium general-purpose VM
- persistent storage volume attached
- one floating IP
- name example: `gemspot-mlflow-server-proj99`

2. Training machine
- the smallest machine that comfortably runs your training job
- for this starter, a CPU VM is enough because the provided models are scikit-learn models
- name example: `gemspot-train-server-proj99`

If your course staff expects you to follow the GPU workflow from `ML5`, use a GPU training machine and the ML5 container setup steps anyway. The model will still run correctly there.

Important:

- put your project suffix at the end of resource names
- only keep instances running while you are actively working
- use only one floating IP, usually on the MLflow machine

## 3. Launch MLflow First

Your instructors specifically want MLflow running with persistent storage before training.

Use the notebook flow from [mlflow_setup.pdf](/Users/rohitshidid/Downloads/mlflow_setup.pdf), but rename resources to satisfy the suffix rule. For example:

- bucket: `gemspot-mlflow-artifacts-proj99`
- lease: `gemspot-mlflow-lease-proj99`
- server: `gemspot-mlflow-server-proj99`
- volume: `gemspot-mlflow-persist-proj99`
- security groups: `gemspot-allow-ssh-proj99`, `gemspot-allow-8000-proj99`

### On Chameleon

Run the notebook-equivalent setup from the PDF so that:

- PostgreSQL data lives on the mounted volume
- artifacts go to CHI@TACC object storage
- the MLflow UI is available on port `8000`

Check that it is running:

```bash
docker ps
curl http://127.0.0.1:8000
```

In your browser, open:

```text
http://YOUR_FLOATING_IP:8000
```

Keep this URL. You need it for:

- training runs
- your run table links
- your final submission

## 4. Prepare the Training Machine

If you use a second machine for training, follow the training-machine pattern from `ML5.pdf`.

That means:

- bring up the training machine
- install Docker
- if it is a GPU node, install NVIDIA container toolkit
- test with `docker run --rm --gpus all ubuntu nvidia-smi`
- build and run your training image on that machine

Important:

- if you adapt the ML5 notebook, change its default resource names
- do not leave names like `node-llm-single-<username>`
- use names with your project suffix, for example `gemspot-train-server-proj99` and `gemspot-train-lease-proj99`

SSH into the training machine:

```bash
ssh cc@TRAINING_MACHINE_IP
```

Install Docker and Git:

```bash
curl -sSL https://get.docker.com/ | sudo sh
sudo groupadd -f docker
sudo usermod -aG docker "$USER"
newgrp docker
docker run hello-world
```

Clone your repo:

```bash
git clone <your-repo-url> gemspot
cd gemspot
```

If this is a GPU server, install the NVIDIA container toolkit using the ML5 flow:

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

## 5. Create the Dataset

For your graded project, you should replace the synthetic demo data with your real processed GemSpot training data.

But to get unstuck immediately and validate the pipeline, first create the demo dataset.

If you want to stay close to the ML5 container-first workflow, generate the demo data inside Docker:

```bash
export PROJECT_SUFFIX=proj99
docker build -t gemspot-train-${PROJECT_SUFFIX} .
docker run --rm \
  -v "$(pwd):/app" \
  gemspot-train-${PROJECT_SUFFIX} \
  python scripts/make_demo_dataset.py --output-dir data/demo
```

If you prefer to do it on the host, this also works:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 scripts/make_demo_dataset.py --output-dir data/demo
```

Later, when your data pipeline is ready, your real CSV should keep the same columns as the config expects.

## 6. Run Training In Docker

Set your MLflow server address:

```bash
export PROJECT_SUFFIX=proj99
export MLFLOW_TRACKING_URI=http://YOUR_FLOATING_IP:8000
```

Build and run:

```bash
export DOCKER_EXTRA_ARGS="--gpus all"
bash scripts/run_training_container.sh
```

If you are on a CPU-only training VM, unset the GPU flag:

```bash
unset DOCKER_EXTRA_ARGS
bash scripts/run_training_container.sh
```

That command will:

- build the Docker image
- run the training script in a container
- send every run to MLflow

## 7. What To Check In MLflow

Open MLflow in the browser and verify that you can see:

- one run per model candidate
- parameters
- metrics
- artifacts
- model artifact
- environment information

For this repo, each run logs:

- candidate model name and hyperparameters
- accuracy, precision, recall, F1, ROC AUC, average precision
- training time
- rows per second
- environment details such as host and GPU count

## 8. How To Move From Demo Data To Real Data

Replace `data/demo/gemspot_train.csv` and `data/demo/gemspot_val.csv` with your real data export.

Your real data should contain:

- user behavior features like visit counts or save counts
- user preference features
- destination metadata
- optional review text or aggregated review text
- the binary target column `will_visit`

Do not create different scripts for different models.

Instead:

- keep one script: `src/train.py`
- change model candidates in `configs/candidates.yaml`
- rerun training

## 9. Recommended Run Sequence

Use this exact order:

1. `dummy_most_frequent`
2. `logistic_regression_baseline`
3. `random_forest_v1`
4. `hist_gradient_boosting_v1`

Why this order:

- dummy gives a hard lower bound
- logistic is your simple real baseline
- random forest checks non-linear interactions
- gradient boosting is a strong candidate for tabular ranking-style tasks

## 10. What To Say If Asked Why These Models

You can say:

- GemSpot is mostly structured tabular data with a small amount of text
- we needed a simple baseline plus stronger non-linear candidates
- logistic regression is fast, explainable, and cheap
- random forest and boosting test whether feature interactions improve recommendation quality
- MLflow lets us compare quality against runtime cost
