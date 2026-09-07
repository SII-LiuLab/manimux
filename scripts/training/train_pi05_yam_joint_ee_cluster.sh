#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-train}
run_name=${2:-assemble-screwdriver-joint-ee-v1-s0-4xh100-3k}

export OPENPI_TRAIN_CONFIG_NAME=pi05_yam_joint_ee
export OPENPI_LEROBOT_REPO_ID=${OPENPI_LEROBOT_REPO_ID:-yam_assemble_screwdriver_20260825_v1_joint_ee}
export PI05_EE_AUX=true

exec bash "${SCRIPT_DIR}/train_pi05_yam_cluster.sh" "${mode}" "${run_name}"
