#!/usr/bin/env bash
set -euo pipefail

mode=${1:-train}
run_name=${2:-assemble-screwdriver-v1-s0-4xh100-3k}

ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=${PI05_WORKSPACE:-${REPO_ROOT}}
POLICY=${WORKSPACE}/XPolicyLab/policy/Pi_05
OPENPI=${POLICY}/openpi
VENV=${OPENPI_ENV_DIR:-${ROOT}/envs/pi05/.venv}
DATASET_NAME=${OPENPI_LEROBOT_REPO_ID:-yam_assemble_screwdriver_20260825_v1}
DATASET=${ROOT}/datasets/lerobot/${DATASET_NAME}
TRAIN_CONFIG_NAME=${OPENPI_TRAIN_CONFIG_NAME:-pi05_yam}
EE_AUX=${PI05_EE_AUX:-false}
BASE_PARAMS=${OPENPI_BASE_PARAMS:-${ROOT}/weights/base/pi05_base/params}
ASSETS_BASE=${OPENPI_ASSETS_BASE_DIR:-${ROOT}/cache/pi05/assets}
NORM_STATS=${ASSETS_BASE}/${TRAIN_CONFIG_NAME}/${DATASET_NAME}/norm_stats.json
OUTPUT=${ROOT}/weights/finetuned/pi05/${run_name}
LOG_DIR=${ROOT}/runs/pi05
GPU_IDS=${OPENPI_GPU_IDS:-0,1,2,3}

export PATH="${ROOT}/envs/bin:${PATH}"
export UV_CACHE_DIR=${ROOT}/cache/uv
export UV_PROJECT_ENVIRONMENT=${VENV}
export HF_HOME=${ROOT}/cache/huggingface
export HF_LEROBOT_HOME=${ROOT}/datasets/lerobot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME=${ROOT}/cache/openpi
export OPENPI_LOCAL_CACHE_ROOT=${ROOT}/cache/pi05/${HOSTNAME}
export OPENPI_TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME}
export OPENPI_LEROBOT_REPO_ID=${DATASET_NAME}
export OPENPI_BASE_PARAMS=${BASE_PARAMS}
export OPENPI_ASSETS_BASE_DIR=${ASSETS_BASE}
export OPENPI_FSDP_DEVICES=${OPENPI_FSDP_DEVICES:-4}
export OPENPI_NUM_WORKERS=${OPENPI_NUM_WORKERS:-0}
export OPENPI_WANDB_ENABLED=false

mkdir -p "${LOG_DIR}" "${ROOT}/cache/pi05/${HOSTNAME}"

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "Required artifact is missing: $1" >&2
    exit 1
  fi
}

preflight() {
  require_file "${OPENPI}/scripts/train.py"
  require_file "${DATASET}/meta/info.json"
  require_file "${BASE_PARAMS}/manifest.ocdbt"
  require_file "${NORM_STATS}"
  python3 - "${DATASET}" "${EE_AUX}" <<'PY'
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
ee_aux = sys.argv[2].lower() == "true"
info = json.loads((dataset / "meta/info.json").read_text())
assert info["codebase_version"] == "v3.0"
assert info["total_episodes"] == 19
assert info["total_frames"] == 17789
assert info["features"]["observation.state"]["shape"] == [14]
assert info["features"]["action"]["shape"] == [14]
if ee_aux:
    assert info["features"]["observation.ee_pose"]["shape"] == [24]
    assert info["features"]["action.ee_pose"]["shape"] == [24]
print(json.dumps({"dataset": dataset.name, "episodes": 19, "frames": 17789}, indent=2))
PY
  python3 - "${NORM_STATS}" "${EE_AUX}" <<'PY'
import json
import sys
from pathlib import Path

stats = json.loads(Path(sys.argv[1]).read_text())["norm_stats"]
ee_aux = sys.argv[2].lower() == "true"
expected_action_dim = 26 if ee_aux else 14
assert len(stats["state"]["mean"]) == 14
assert len(stats["actions"]["mean"]) == expected_action_dim
print(f"Pi05 norm stats verified: state=14D actions={expected_action_dim}D")
PY
}

compute_stats() {
  if [[ -s "${NORM_STATS}" ]]; then
    echo "[Pi_05] reusing norm stats: ${NORM_STATS}"
    return
  fi
  echo "[Pi_05] computing norm stats from ${DATASET}"
  (
    cd "${OPENPI}"
    OPENPI_LEROBOT_REPO_ID=${DATASET_NAME} \
    OPENPI_ASSETS_BASE_DIR=${ASSETS_BASE} \
    OPENPI_BASE_PARAMS=${BASE_PARAMS} \
      uv run scripts/compute_norm_stats.py --config-name "${TRAIN_CONFIG_NAME}"
  )
  require_file "${NORM_STATS}"
}

install_environment() {
  if [[ -x "${VENV}/bin/python" ]] && \
      "${VENV}/bin/python" - "${OPENPI}" <<'PY' >/dev/null 2>&1
import pathlib
import sys
import jax
import openpi
assert len(jax.devices()) > 0
pathlib.Path(openpi.__file__).resolve().relative_to(pathlib.Path(sys.argv[1]).resolve())
PY
  then
    echo "[Pi_05] reusing ${VENV}"
    return
  fi
  mkdir -p "${VENV%/.venv}"
  bash "${POLICY}/install.sh"
}

case "${mode}" in
  gate-train)
    OPENPI_BATCH_SIZE=4 OPENPI_NUM_TRAIN_STEPS=1 OPENPI_SAVE_INTERVAL=1 \
      OPENPI_MAX_TO_KEEP=1 bash "$0" smoke "${run_name}"
    require_file "${ROOT}/weights/finetuned/pi05/${run_name}-smoke/1/params/manifest.ocdbt"
    OPENPI_BATCH_SIZE=${OPENPI_BATCH_SIZE:-32} OPENPI_NUM_TRAIN_STEPS=3000 \
      OPENPI_SAVE_INTERVAL=500 OPENPI_MAX_TO_KEEP=10 \
      bash "$0" train "${run_name}"
    ;;
  prepare)
    install_environment
    compute_stats
    preflight
    ;;
  smoke|train)
    install_environment
    compute_stats
    preflight
    if [[ "${mode}" == "smoke" ]]; then
      export OPENPI_BATCH_SIZE=${OPENPI_BATCH_SIZE:-4}
      export OPENPI_NUM_TRAIN_STEPS=${OPENPI_NUM_TRAIN_STEPS:-1}
      export OPENPI_SAVE_INTERVAL=${OPENPI_SAVE_INTERVAL:-1}
      export OPENPI_MAX_TO_KEEP=${OPENPI_MAX_TO_KEEP:-1}
      export OPENPI_CHECKPOINT_DIR=${OUTPUT}-smoke
    else
      export OPENPI_BATCH_SIZE=${OPENPI_BATCH_SIZE:-32}
      export OPENPI_NUM_TRAIN_STEPS=${OPENPI_NUM_TRAIN_STEPS:-3000}
      export OPENPI_SAVE_INTERVAL=${OPENPI_SAVE_INTERVAL:-500}
      export OPENPI_MAX_TO_KEEP=${OPENPI_MAX_TO_KEEP:-10}
      export OPENPI_CHECKPOINT_DIR=${OUTPUT}
      if [[ -e "${OUTPUT}" ]]; then
        echo "Refusing to overwrite existing run: ${OUTPUT}" >&2
        exit 2
      fi
    fi
    bash "${POLICY}/train.sh" \
      yam assemble_the_screwdriver yam_dual joint 0 "${GPU_IDS}" \
      2>&1 | tee "${LOG_DIR}/${run_name}-${mode}.log"
    ;;
  *)
    echo "Usage: $0 [prepare|smoke|train|gate-train] [run_name]" >&2
    exit 2
    ;;
esac
