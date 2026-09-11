#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}

mode=${1:-train}
run_name=${2:-put-bottles-xr1-ee-8xh100-b64-30k}

export XR1_DATASET_PATH=${XR1_DATASET_PATH:-${ROOT}/datasets/xr1/RoboDojo_real-put_bottles_into_the_bin-yam_dual-ee}
export XR1_YAM_EPISODES=${XR1_YAM_EPISODES:-${ROOT}/datasets/raw/put_bottles_into_the_bin}
export XR1_DATA_CONFIG_NAME=${XR1_DATA_CONFIG_NAME:-yam_put_bottles_into_the_bin}
export XR1_TASK_NAME=${XR1_TASK_NAME:-put_bottles_into_the_bin}
export XR1_INSTRUCTION=${XR1_INSTRUCTION:-Put the bottles into the bin.}
export XR1_EXPECTED_EPISODES=${XR1_EXPECTED_EPISODES:-50}
export XR1_EXPECTED_FRAMES=${XR1_EXPECTED_FRAMES:-35118}

export XR1_GPU_IDS=${XR1_GPU_IDS:-0,1,2,3,4,5,6,7}
export XR1_MICRO_BATCH_SIZE=${XR1_MICRO_BATCH_SIZE:-1}
export XR1_GRAD_ACCUM_STEPS=${XR1_GRAD_ACCUM_STEPS:-8}
export XR1_MAX_STEPS=${XR1_MAX_STEPS:-30000}
export XR1_SAVE_INTERVAL=${XR1_SAVE_INTERVAL:-5000}

exec bash "${SCRIPT_DIR}/train_xr1_yam_cluster.sh" "${mode}" "${run_name}"
