#!/usr/bin/env bash
set -euo pipefail

PROJECT_SUFFIX="${PROJECT_SUFFIX:?Set PROJECT_SUFFIX, for example proj99}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:?Set MLFLOW_TRACKING_URI, for example http://1.2.3.4:8000}"
IMAGE_NAME="gemspot-train-${PROJECT_SUFFIX}"
CONFIG_PATH="${CONFIG_PATH:-configs/candidates.yaml}"
TRAIN_CSV="${TRAIN_CSV:-data/demo/gemspot_train.csv}"
VAL_CSV="${VAL_CSV:-data/demo/gemspot_val.csv}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-GemSpot-WillVisit}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/models}"
DOCKER_EXTRA_ARGS="${DOCKER_EXTRA_ARGS:-}"

read -r -a docker_extra_args <<< "${DOCKER_EXTRA_ARGS}"

docker build -t "${IMAGE_NAME}" .

docker run --rm \
  "${docker_extra_args[@]}" \
  --name "${IMAGE_NAME}-run" \
  -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
  -v "$(pwd):/app" \
  "${IMAGE_NAME}" \
  python src/train.py \
    --config "${CONFIG_PATH}" \
    --train-csv "${TRAIN_CSV}" \
    --val-csv "${VAL_CSV}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --artifact-dir "${ARTIFACT_DIR}"
