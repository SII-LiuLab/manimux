#!/usr/bin/env bash
set -euo pipefail

# Fixed QZ hdd3 profile for the 50-episode put-bottles OpenWAM finetune.
# This script does not allocate resources or create a QZ job.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OPENWAM_TRAIN_ROOT=${OPENWAM_TRAIN_ROOT:-/inspire/hdd3/project/agent-driven-world-model/ky26300/manimux_training}
export OPENWAM_WORKSPACE=${OPENWAM_WORKSPACE:-${OPENWAM_TRAIN_ROOT}/operate/manimux-openwam}
export OPENWAM_PYTHON=${OPENWAM_PYTHON:-${OPENWAM_TRAIN_ROOT}/envs/openwam/.venv/bin/python}
export OPENWAM_DATASET_DIR=${OPENWAM_DATASET_DIR:-${OPENWAM_TRAIN_ROOT}/datasets/openwam/yam_put_bottles_into_the_bin_20260905_v1}
export OPENWAM_FINETUNE_CKPT_PATH=${OPENWAM_FINETUNE_CKPT_PATH:-${OPENWAM_TRAIN_ROOT}/weights/base/openwam/OpenWAM-Alpha-Pretrain-Foundation-Model}
export OPENWAM_OUTPUT_ROOT=${OPENWAM_OUTPUT_ROOT:-${OPENWAM_TRAIN_ROOT}/weights/finetuned/openwam}
export OPENWAM_LOG_DIR=${OPENWAM_LOG_DIR:-${OPENWAM_TRAIN_ROOT}/runs/openwam/logs}
export OPENWAM_GPU_IDS=${OPENWAM_GPU_IDS:-0,1,2,3}
export OPENWAM_MAX_STEPS=${OPENWAM_MAX_STEPS:-30000}
export OPENWAM_SAVE_INTERVAL=${OPENWAM_SAVE_INTERVAL:-5000}
export OPENWAM_BATCH_SIZE=${OPENWAM_BATCH_SIZE:-1}
export OPENWAM_GRADIENT_ACCUMULATION_STEPS=${OPENWAM_GRADIENT_ACCUMULATION_STEPS:-8}
export OPENWAM_KEEP_LAST_K_CKPTS=${OPENWAM_KEEP_LAST_K_CKPTS:-6}
export OPENWAM_EXPECTED_EPISODES=50
export OPENWAM_EXPECTED_FRAMES=35118
export OPENWAM_EXPECTED_FOUNDATION_SHA256=180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d

exec bash "${SCRIPT_DIR}/train_openwam_yam_cluster.sh" \
  "${1:-ready}" "${2:-put-bottles-openwam-v1-s0-30k}"
