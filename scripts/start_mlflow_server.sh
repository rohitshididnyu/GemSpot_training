#!/usr/bin/env bash
set -euo pipefail

PROJECT_SUFFIX="${PROJECT_SUFFIX:?Set PROJECT_SUFFIX, for example proj99}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_DATA_DIR="${MLFLOW_DATA_DIR:-/mnt/mlflow-${PROJECT_SUFFIX}}"
MLFLOW_IMAGE="${MLFLOW_IMAGE:-ghcr.io/mlflow/mlflow:v2.12.2}"
CONTAINER_NAME="gemspot-mlflow-${PROJECT_SUFFIX}"

# MinIO S3 configuration for scalable artifact storage
# Set MINIO_ENDPOINT to the bare-metal node IP running MinIO
MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://129.114.109.166:9000}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-s3://mlflow-artifacts}"

mkdir -p "${MLFLOW_DATA_DIR}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# Install boto3 (required for S3 artifact store) and start MLflow
docker run -d \
  --restart unless-stopped \
  --name "${CONTAINER_NAME}" \
  -p "${MLFLOW_PORT}:5000" \
  -v "${MLFLOW_DATA_DIR}:/mlflow" \
  -e AWS_ACCESS_KEY_ID=minioadmin \
  -e AWS_SECRET_ACCESS_KEY=minioadmin \
  -e MLFLOW_S3_ENDPOINT_URL="${MINIO_ENDPOINT}" \
  "${MLFLOW_IMAGE}" \
  bash -c "pip install -q boto3 && mlflow server \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --default-artifact-root ${ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5000"

echo "MLflow is starting at http://<this-host-ip>:${MLFLOW_PORT}"
echo "Artifacts will be stored in MinIO at ${MINIO_ENDPOINT} → ${ARTIFACT_ROOT}"
