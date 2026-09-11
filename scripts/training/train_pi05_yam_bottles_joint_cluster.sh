#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export YAM_TRAIN_ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd3/project/agent-driven-world-model/ky26300/manimux_training}
# Same episodes and images as joint+EEF; pi05_yam does not consume the EE columns.
export OPENPI_LEROBOT_REPO_ID=yam_put_bottles_into_the_bin_20260905_joint_ee_v1
export OPENPI_TASK_NAME=put_bottles_into_the_bin
export OPENPI_EXPECTED_EPISODES=50
export OPENPI_EXPECTED_FRAMES=35118
export OPENPI_ASSETS_BASE_DIR=${OPENPI_ASSETS_BASE_DIR:-${YAM_TRAIN_ROOT}/assets}
export OPENPI_NUM_TRAIN_STEPS=${OPENPI_NUM_TRAIN_STEPS:-15000}
export OPENPI_TRAIN_CONFIG_NAME=pi05_yam
export PI05_EE_AUX=false

exec bash "${SCRIPT_DIR}/train_pi05_yam_cluster.sh" \
  "${1:-prepare}" "${2:-put-bottles-joint-v1-s0-15k}"
