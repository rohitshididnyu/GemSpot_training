#!/usr/bin/env bash
set -euo pipefail

PROJECT_SUFFIX="${PROJECT_SUFFIX:?Set PROJECT_SUFFIX, for example proj99}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_DATA_DIR="${MLFLOW_DATA_DIR:-/mnt/mlflow-${PROJECT_SUFFIX}}"
MLFLOW_IMAGE="${MLFLOW_IMAGE:-ghcr.io/mlflow/mlflow:v2.12.2}"
CONTAINER_NAME="gemspot-mlflow-${PROJECT_SUFFIX}"

mkdir -p "${MLFLOW_DATA_DIR}/artifacts"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --restart unless-stopped \
  --name "${CONTAINER_NAME}" \
  -p "${MLFLOW_PORT}:5000" \
  -v "${MLFLOW_DATA_DIR}:/mlflow" \
  "${MLFLOW_IMAGE}" \
  mlflow server \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000

echo "MLflow is starting at http://<this-host-ip>:${MLFLOW_PORT}"
