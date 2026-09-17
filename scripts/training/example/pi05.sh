#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/paths.env"
export OPENPI_TASK_NAME=${OPENPI_TASK_NAME:-example_task}
export OPENPI_LEROBOT_REPO_ID=${OPENPI_LEROBOT_REPO_ID:-example_yam_dataset}
export OPENPI_TRAIN_CONFIG_NAME=${OPENPI_TRAIN_CONFIG_NAME:-pi05_yam}
export OPENPI_GPU_IDS=${OPENPI_GPU_IDS:-0,1,2,3}
export OPENPI_BATCH_SIZE=${OPENPI_BATCH_SIZE:-64}
export OPENPI_NUM_TRAIN_STEPS=${OPENPI_NUM_TRAIN_STEPS:-3000}
export OPENPI_SAVE_INTERVAL=${OPENPI_SAVE_INTERVAL:-500}
exec bash "${SCRIPT_DIR}/../train_pi05_yam_cluster.sh" \
  "${1:-plan}" "${2:-example-run}"
