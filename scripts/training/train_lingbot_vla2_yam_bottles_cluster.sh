#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}

mode=${1:-train}
run_name=${2:-put-bottles-lingbot-vla2-joint-ee-native-depth-8xh100-b64-15k}

export LINGBOT_VLA2_DATASET_PATH=${LINGBOT_VLA2_DATASET_PATH:-${ROOT}/datasets/lerobot/yam_put_bottles_into_the_bin_20260905_joint_ee_v1}
export LINGBOT_VLA2_DATASET_NAME=${LINGBOT_VLA2_DATASET_NAME:-yam_put_bottles_into_the_bin_20260905_joint_ee_v1}
export LINGBOT_VLA2_EXPECTED_EPISODES=${LINGBOT_VLA2_EXPECTED_EPISODES:-50}
export LINGBOT_VLA2_EXPECTED_FRAMES=${LINGBOT_VLA2_EXPECTED_FRAMES:-35118}
export LINGBOT_VLA2_TRAINING_CONFIG=${REPO_ROOT}/configs/lingbot-vla2/yam/native-depth-training.yaml
export LINGBOT_VLA2_ROBOT_CONFIG_ROOT=${REPO_ROOT}/configs/lingbot-vla2/yam/robot_configs
export LINGBOT_VLA2_ROBOT_NAME=yam_dual_joint_ee_relative
export LINGBOT_VLA2_DEPLOY_ROBOT_CONFIG=${REPO_ROOT}/XPolicyLab/policy/LingBot_VLA2/robot_configs/yam_dual_packed_relative.yaml
export LINGBOT_VLA2_EE_AUX=true

export LINGBOT_VLA2_GPU_IDS=${LINGBOT_VLA2_GPU_IDS:-0,1,2,3,4,5,6,7}
export LINGBOT_VLA2_MICRO_BATCH_SIZE=${LINGBOT_VLA2_MICRO_BATCH_SIZE:-1}
export LINGBOT_VLA2_GRAD_ACCUM_STEPS=${LINGBOT_VLA2_GRAD_ACCUM_STEPS:-8}
export LINGBOT_VLA2_GLOBAL_BATCH_SIZE=${LINGBOT_VLA2_GLOBAL_BATCH_SIZE:-64}
export LINGBOT_VLA2_MAX_STEPS=${LINGBOT_VLA2_MAX_STEPS:-15000}
export LINGBOT_VLA2_SAVE_STEPS=${LINGBOT_VLA2_SAVE_STEPS:-500}

export LINGBOT_VLA2_MOGE_PATH=${LINGBOT_VLA2_MOGE_PATH:-${ROOT}/weights/base/moge-2-vitb-normal/model.pt}
export LINGBOT_VLA2_MORGBD_PATH=${LINGBOT_VLA2_MORGBD_PATH:-${ROOT}/weights/base/lingbot-vla-v2-6b/depth/model.pt}
export LINGBOT_VLA2_DINO_VIDEO_CKPT=${LINGBOT_VLA2_DINO_VIDEO_CKPT:-${ROOT}/weights/base/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth}
export LINGBOT_VLA2_DINO_VIDEO_CONFIG=${LINGBOT_VLA2_DINO_VIDEO_CONFIG:-${ROOT}/weights/base/lingbot-vla-v2-6b/dino_video/config.yaml}

exec bash "${SCRIPT_DIR}/train_lingbot_vla2_yam_cluster.sh" "${mode}" "${run_name}"
