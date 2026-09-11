#!/usr/bin/env bash
set -euo pipefail

# Data preparation profile, not a QZ job submission or resource selection.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export YAM_TRAIN_ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd3/project/agent-driven-world-model/ky26300/manimux_training}
export OPENPI_LEROBOT_REPO_ID=yam_put_bottles_into_the_bin_20260905_joint_ee_v1
export OPENPI_TASK_NAME=put_bottles_into_the_bin
export OPENPI_EXPECTED_EPISODES=50
export OPENPI_EXPECTED_FRAMES=35118
export OPENPI_ASSETS_BASE_DIR=${OPENPI_ASSETS_BASE_DIR:-${YAM_TRAIN_ROOT}/assets}
export OPENPI_NUM_TRAIN_STEPS=${OPENPI_NUM_TRAIN_STEPS:-15000}

exec bash "${SCRIPT_DIR}/train_pi05_yam_joint_ee_cluster.sh" \
  "${1:-prepare}" "${2:-put-bottles-joint-ee-v1-s0-15k}"
