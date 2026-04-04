#!/usr/bin/env bash
set -euo pipefail

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:?Set MLFLOW_TRACKING_URI, for example http://A.B.C.D:8000}"
CONTAINER_NAME="${CONTAINER_NAME:-gemspot-jupyter}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)}"
DOCKER_EXTRA_ARGS="${DOCKER_EXTRA_ARGS:-}"

read -r -a docker_extra_args <<< "${DOCKER_EXTRA_ARGS}"

docker build -t gemspot-jupyter-mlflow -f Dockerfile.jupyter-mlflow .

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d --rm \
  "${docker_extra_args[@]}" \
  -p "${JUPYTER_PORT}:8888" \
  --shm-size 16G \
  -v "${WORKSPACE_DIR}:/home/jovyan/work" \
  -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
  --name "${CONTAINER_NAME}" \
  gemspot-jupyter-mlflow

echo "Jupyter container started."
echo "Next command:"
echo "docker exec ${CONTAINER_NAME} jupyter server list"
