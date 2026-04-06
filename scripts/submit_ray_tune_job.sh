#!/usr/bin/env bash
set -euo pipefail

RAY_RUNTIME_FILE="${RAY_RUNTIME_FILE:-ray-runtime.json}"
TRAIN_CSV="${TRAIN_CSV:-data/demo/gemspot_train.csv}"
VAL_CSV="${VAL_CSV:-data/demo/gemspot_val.csv}"
CONFIG_PATH="${CONFIG_PATH:-configs/ray_bonus.yaml}"
RAY_EXTRA_ARGS="${RAY_EXTRA_ARGS:-}"
TRACKING_URI_ARG=()
EXPERIMENT_ARG=()
RUN_NAME_ARG=()

if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
  TRACKING_URI_ARG=(--tracking-uri "${MLFLOW_TRACKING_URI}")
fi

if [[ -n "${RAY_EXPERIMENT_NAME:-}" ]]; then
  EXPERIMENT_ARG=(--experiment-name "${RAY_EXPERIMENT_NAME}")
fi

if [[ -n "${RAY_RUN_NAME:-}" ]]; then
  RUN_NAME_ARG=(--run-name "${RAY_RUN_NAME}")
fi

read -r -a ray_extra_args <<< "${RAY_EXTRA_ARGS}"

ray job submit \
  "${ray_extra_args[@]}" \
  --runtime-env "${RAY_RUNTIME_FILE}" \
  --working-dir . \
  -- \
  python src/train_ray_tune.py \
    --config "${CONFIG_PATH}" \
    --train-csv "${TRAIN_CSV}" \
    --val-csv "${VAL_CSV}" \
    "${RUN_NAME_ARG[@]}" \
    "${TRACKING_URI_ARG[@]}" \
    "${EXPERIMENT_ARG[@]}"
