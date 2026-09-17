#!/usr/bin/env bash
set -euo pipefail

mode=${1:-train}
run_name=${2:-gr00t-yam}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROOT=${YAM_TRAIN_ROOT:-${REPO_ROOT}/data/training}
source "${REPO_ROOT}/scripts/training/_batch.sh"
WORKSPACE=${GR00T_WORKSPACE:-${REPO_ROOT}}
POLICY=${WORKSPACE}/XPolicyLab/policy/GR00T_N17
SOURCE=${POLICY}/gr00t_n17
ENV_DIR=${ROOT}/envs/gr00t-n17/.venv
DATA_ROOT=${ROOT}/datasets/lerobot
TASK_NAME=${GR00T_TASK_NAME:?Set GR00T_TASK_NAME in the task recipe}
SOURCE_DATASET=${GR00T_SRC_DATASET:?Set GR00T_SRC_DATASET in the task recipe}
PROCESSED_DATASET=yam-${TASK_NAME}-yam_dual-joint
BASE_MODEL=${ROOT}/weights/base/GR00T-N1.7-3B
COSMOS_MODEL=${ROOT}/weights/base/Cosmos-Reason2-2B
OUTPUT=${ROOT}/weights/finetuned/gr00t-n17/${run_name}
LOG_DIR=${ROOT}/runs/gr00t-n17
GPU_IDS=${GR00T_GPU_IDS:-0,1,2,3}
GPU_COUNT=$(training_gpu_count "${GPU_IDS}")
GLOBAL_BATCH_SIZE=${GR00T_GLOBAL_BATCH_SIZE:-64}
training_check_batch GR00T "${mode}" "${GLOBAL_BATCH_SIZE}" "${GPU_COUNT}" "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} (global)"
[[ "${GR00T_NUM_GPUS:-${GPU_COUNT}}" == "${GPU_COUNT}" ]] || { echo "GR00T_NUM_GPUS disagrees with GPU IDs" >&2; exit 2; }
training_plan "${mode}" "entry=${POLICY}/train.sh" "dataset=${DATA_ROOT}/${SOURCE_DATASET}" \
  "prepared=${DATA_ROOT}/${PROCESSED_DATASET}" "output=${OUTPUT}" \
  "prepare=converted LeRobot required; policy/process_data.sh prepares v2.1, modality and stats" "steps=${GR00T_MAX_STEPS:-10000} (optimizer updates)"

export PATH="${ROOT}/envs/bin:${PATH}"
export HF_HOME=${ROOT}/cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export UV_CACHE_DIR=${ROOT}/cache/uv
export UV_NO_SOURCES=${UV_NO_SOURCES:-1}
export MAX_JOBS=${MAX_JOBS:-16}
export TORCH_HOME=${ROOT}/cache/torch
export GR00T_ENV_DIR=${ENV_DIR}
export GR00T_LEROBOT_HOME=${DATA_ROOT}
export GR00T_SRC_DATASET=${SOURCE_DATASET}
export GR00T_BASE_MODEL=${BASE_MODEL}
export GR00T_COSMOS_MODEL=${COSMOS_MODEL}
export GR00T_CHECKPOINT_DIR=${OUTPUT}
export GR00T_VIDEO_BACKEND=${GR00T_VIDEO_BACKEND:-pyav}
export HF_HUB_OFFLINE=1
export NUM_GPUS=${GPU_COUNT}
export MAX_STEPS=${GR00T_MAX_STEPS:-10000}
export SAVE_STEPS=${GR00T_SAVE_STEPS:-100}
export SAVE_TOTAL_LIMIT=${GR00T_SAVE_TOTAL_LIMIT:-10}
export GLOBAL_BATCH_SIZE
export DATALOADER_NUM_WORKERS=${GR00T_DATALOADER_NUM_WORKERS:-8}
export GR00T_LEROBOT_SPEC=${GR00T_LEROBOT_SPEC:-lerobot==0.4.2}
export USE_WANDB=0 WANDB_MODE=disabled WANDB_DISABLED=true

mkdir -p "${ENV_DIR%/.venv}" "${LOG_DIR}"

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "Required artifact is missing: $1" >&2
    exit 1
  fi
}

install_environment() {
  if [[ -x "${ENV_DIR}/bin/python" ]] && "${ENV_DIR}/bin/python" -c \
      'from importlib.metadata import version; import flash_attn, gr00t, torch; assert torch.cuda.is_available(); assert version("numpy") == "1.26.4"; assert version("datasets") == "3.6.0"; assert version("wandb") == "0.23.0"; assert version("av") == "16.1.0"' >/dev/null 2>&1; then
    echo "[GR00T_N17] reusing ${ENV_DIR}"
    return
  fi
  echo "[GR00T_N17] installing environment at ${ENV_DIR}"
  bash "${POLICY}/install.sh"
}

preflight() {
  require_file "${SOURCE}/gr00t/experiment/launch_finetune.py"
  require_file "${BASE_MODEL}/model.safetensors.index.json"
  require_file "${BASE_MODEL}/model-00001-of-00002.safetensors"
  require_file "${BASE_MODEL}/model-00002-of-00002.safetensors"
  require_file "${COSMOS_MODEL}/model.safetensors"
  require_file "${DATA_ROOT}/${SOURCE_DATASET}/meta/info.json"
  require_file "${POLICY}/configs/yam_dual_config.py"
  "${ENV_DIR}/bin/python" - "${DATA_ROOT}/${SOURCE_DATASET}" <<'PY'
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
info = json.loads((dataset / "meta/info.json").read_text())
assert info["codebase_version"] == "v3.0"
import os
assert info["total_episodes"] > 0
if os.environ.get("GR00T_EXPECTED_EPISODES"):
    assert info["total_episodes"] == int(os.environ["GR00T_EXPECTED_EPISODES"])
assert info["features"]["observation.state"]["shape"] == [14]
assert info["features"]["action"]["shape"] == [14]
print(json.dumps({
    "dataset": dataset.name,
    "episodes": info["total_episodes"],
    "frames": info["total_frames"],
    "fps": info["fps"],
    "gpus": __import__("torch").cuda.device_count(),
}, indent=2))
PY
}

prepare_data() {
  if [[ -s "${DATA_ROOT}/${PROCESSED_DATASET}/meta/stats.json" ]] && \
      [[ -s "${DATA_ROOT}/${PROCESSED_DATASET}/meta/modality.json" ]]; then
    echo "[GR00T_N17] reusing processed dataset ${PROCESSED_DATASET}"
    return
  fi
  bash "${POLICY}/process_data.sh" yam "${TASK_NAME}" yam_dual joint
}

case "${mode}" in
  prepare)
    install_environment
    preflight
    prepare_data
    ;;
  smoke|train)
    install_environment
    preflight
    prepare_data
    if [[ "${mode}" == "smoke" ]]; then
      export MAX_STEPS=1 SAVE_STEPS=1
    fi
    if [[ -d "${OUTPUT}" && -n "$(ls -A "${OUTPUT}")" ]]; then
      echo "Refusing to overwrite existing run: ${OUTPUT}" >&2; exit 2
    fi
    CUDA_VISIBLE_DEVICES=${GPU_IDS} bash "${POLICY}/train.sh" \
      yam "${TASK_NAME}" yam_dual joint 0 "${GPU_IDS}" \
      2>&1 | tee "${LOG_DIR}/${run_name}.log"
    ;;
  *)
    echo "Usage: $0 [plan|prepare|smoke|train] [run_name]" >&2
    exit 2
    ;;
esac
