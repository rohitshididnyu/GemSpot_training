#!/usr/bin/env bash
set -euo pipefail

PROJECT_SUFFIX="${PROJECT_SUFFIX:?Set PROJECT_SUFFIX, for example proj99}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:?Set MLFLOW_TRACKING_URI, for example http://1.2.3.4:5000}"
IMAGE_NAME="gemspot-train-${PROJECT_SUFFIX}"

docker build -t "${IMAGE_NAME}" .

docker run --rm \
  --name "${IMAGE_NAME}-run" \
  -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  "${IMAGE_NAME}" \
  python src/train.py \
    --config configs/candidates.yaml \
    --train-csv data/demo/gemspot_train.csv \
    --val-csv data/demo/gemspot_val.csv \
    --experiment-name GemSpot-WillVisit
