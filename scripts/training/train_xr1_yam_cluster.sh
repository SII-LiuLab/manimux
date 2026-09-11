#!/usr/bin/env bash
set -euo pipefail

mode=${1:-train}
run_name=${2:-yam-v1-s0-4xh100-3k}

ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=${XR1_WORKSPACE:-${REPO_ROOT}}
POLICY=${WORKSPACE}/XPolicyLab/policy/Xiaomi_Robotics_1
XR1=${POLICY}/xiaomi_robotics_1/xr1
VENV=${ROOT}/envs/xr1/.venv
DATASET=${XR1_DATASET_PATH:-}
BASE_MODEL=${XR1_BASE_MODEL_PATH:-${ROOT}/weights/base/xiaomi/model_states.pt}
PROCESSOR=${XR1_PROCESSOR_PATH:-${ROOT}/weights/base/xiaomi/qwen3_vl_4b_processor}
OUTPUT=${ROOT}/weights/finetuned/xiaomi-xr1/${run_name}
LOG_DIR=${ROOT}/runs/xiaomi-xr1
GPU_IDS=${XR1_GPU_IDS:-0,1,2,3}
DATA_CONFIG_NAME=${XR1_DATA_CONFIG_NAME:-}
TASK_NAME=${XR1_TASK_NAME:-}
INSTRUCTION=${XR1_INSTRUCTION:-}

export PATH="${ROOT}/envs/bin:${PATH}"
export HF_HOME=${ROOT}/cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR=${ROOT}/cache/uv
export UV_INDEX_URL=${UV_INDEX_URL:-http://nexus.sii.shaipower.online/repository/pypi/simple}
export UV_INSECURE_HOST=${UV_INSECURE_HOST:-nexus.sii.shaipower.online}
export PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
export TORCH_HOME=${ROOT}/cache/torch
export XR1_QWEN_VL_CONFIG_SOURCE=${PROCESSOR}
export XR1_LOGGER=${XR1_LOGGER:-tensorboard}
export MAX_LENGTH=${XR1_MAX_LENGTH:-20000}

mkdir -p "${VENV%/.venv}" "${LOG_DIR}"

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "Required artifact is missing: $1" >&2
    exit 1
  fi
}

require_value() {
  if [[ -z "$2" ]]; then
    echo "$1 must be set" >&2
    exit 1
  fi
}

validate_configuration() {
  require_value XR1_DATASET_PATH "${DATASET}"
  require_value XR1_DATA_CONFIG_NAME "${DATA_CONFIG_NAME}"
  require_value XR1_TASK_NAME "${TASK_NAME}"
  require_value XR1_INSTRUCTION "${INSTRUCTION}"
}

find_episodes() {
  if [[ -n "${XR1_YAM_EPISODES:-}" ]]; then
    printf '%s\n' "${XR1_YAM_EPISODES}"
    return
  fi
  echo "XR1_YAM_EPISODES must be set when the converted dataset is absent" >&2
  exit 1
}

install_environment() {
  if [[ -x "${VENV}/bin/python" ]] && "${VENV}/bin/python" -c \
      'import decord, deepspeed, flash_attn, lightning, mmengine, torch, transformers; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    echo "[Xiaomi_Robotics_1] reusing ${VENV}"
    return
  fi
  command -v uv >/dev/null
  if [[ ! -x "${VENV}/bin/python" ]]; then
    uv venv --python 3.12 "${VENV}"
  fi
  export UV_LINK_MODE=copy
  export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
  uv pip install --python "${VENV}/bin/python" pip setuptools wheel packaging ninja
  uv pip install --python "${VENV}/bin/python" \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url "${PYTORCH_INDEX_URL}"
  uv pip install --python "${VENV}/bin/python" \
    -r "${XR1}/assets/requirements.txt"
  uv pip install --python "${VENV}/bin/python" flash-attn==2.8.3 --no-build-isolation
  uv pip install --python "${VENV}/bin/python" -e "${WORKSPACE}/XPolicyLab"
}

prepare_data() {
  mkdir -p "${DATASET}"
  if [[ "${XR1_REBUILD_DATA:-0}" != "1" && -s "${DATASET}/manifest.json" ]]; then
    echo "[Xiaomi_Robotics_1] reusing converted dataset: ${DATASET}"
  else
    local episodes
    episodes=$(find_episodes)
    echo "[Xiaomi_Robotics_1] raw episodes=${episodes}"
    PYTHONPATH="${WORKSPACE}/src" "${VENV}/bin/python" \
      "${WORKSPACE}/scripts/datasets/prepare_xr1_yam_dataset.py" \
      --episodes "${episodes}" \
      --output "${DATASET}" \
      --instruction "${INSTRUCTION}" \
      --config-name "${DATA_CONFIG_NAME}" \
      --batch-size "${XR1_MICRO_BATCH_SIZE:-1}"
  fi

  local manifest_config
  manifest_config=$("${VENV}/bin/python" - "${DATASET}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "manifest.json").read_text())
config = root / manifest["config"]["path"]
assert config.is_file(), config
print(config)
PY
  )
  cp "${manifest_config}" "${XR1}/configs/data/${DATA_CONFIG_NAME}.yaml"
}

preflight() {
  require_file "${WORKSPACE}/scripts/datasets/prepare_xr1_yam_dataset.py"
  require_file "${BASE_MODEL}"
  require_file "${PROCESSOR}/config.json"
  require_file "${PROCESSOR}/tokenizer.json"
  require_file "${DATASET}/manifest.json"
  require_file "${DATASET}/norm_stats.json"
  require_file "${XR1}/configs/data/${DATA_CONFIG_NAME}.yaml"
  "${VENV}/bin/python" - "${DATASET}" "${BASE_MODEL}" <<'PY'
import json
import sys
from pathlib import Path

dataset, model = map(Path, sys.argv[1:])
manifest = json.loads((dataset / "manifest.json").read_text())
assert manifest["schema"] == "manimux.xr1_yam_dataset.v1"
assert manifest["episodes"] > 0
assert manifest["frames"] > 0
import os
for field, variable in (("episodes", "XR1_EXPECTED_EPISODES"), ("frames", "XR1_EXPECTED_FRAMES")):
    if os.environ.get(variable):
        assert manifest[field] == int(os.environ[variable]), (field, manifest[field], os.environ[variable])
assert model.stat().st_size > 9_000_000_000
print(json.dumps({
    "dataset": str(dataset),
    "episodes": manifest["episodes"],
    "frames": manifest["frames"],
    "model_bytes": model.stat().st_size,
    "gpus": __import__("torch").cuda.device_count(),
}, indent=2))
PY
}

case "${mode}" in
  gate-train)
    XR1_MAX_STEPS=2 XR1_SAVE_INTERVAL=2 \
      bash "$0" smoke "${run_name}-smoke"
    smoke_root="${ROOT}/weights/finetuned/xiaomi-xr1/${run_name}-smoke"
    if ! find "${smoke_root}" -type d -name 'last.ckpt' -print -quit | grep -q .; then
      echo "XR-1 smoke did not produce a checkpoint under ${smoke_root}" >&2
      exit 1
    fi
    XR1_MAX_STEPS=${XR1_MAX_STEPS:-3000} XR1_SAVE_INTERVAL=${XR1_SAVE_INTERVAL:-500} \
      bash "$0" train "${run_name}"
    ;;
  prepare)
    validate_configuration
    install_environment
    prepare_data
    preflight
    ;;
  smoke|train)
    validate_configuration
    install_environment
    prepare_data
    preflight
    if [[ "${mode}" == smoke ]]; then
      max_steps=${XR1_MAX_STEPS:-2}
      save_interval=${XR1_SAVE_INTERVAL:-2}
    else
      max_steps=${XR1_MAX_STEPS:-3000}
      save_interval=${XR1_SAVE_INTERVAL:-500}
      if [[ -d "${OUTPUT}" ]] && find "${OUTPUT}" -mindepth 1 -print -quit | grep -q .; then
        echo "Refusing to overwrite existing run: ${OUTPUT}" >&2
        exit 2
      fi
    fi
    PATH="${VENV}/bin:${PATH}" \
    OUTPUT_DIR="${DATASET}" \
    DATA_CONFIG_NAME="${DATA_CONFIG_NAME}" \
    PRETRAINED_PATH="${BASE_MODEL}" \
    RUN_ROOT="${OUTPUT}" \
    PROJECT="yam-xiaomi-xr1" \
    MAX_STEPS="${max_steps}" \
    SAVE_INTERVAL="${save_interval}" \
    XR1_LOGGER="${XR1_LOGGER}" \
    bash "${POLICY}/train.sh" \
      RoboDojo_real "${TASK_NAME}" yam_dual ee 0 "${GPU_IDS}" \
      trainer.accumulate_grad_batches="${XR1_GRAD_ACCUM_STEPS:-8}" \
      2>&1 | tee "${LOG_DIR}/${run_name}.log"
    ;;
  *)
    echo "Usage: $0 [prepare|smoke|train|gate-train] [run_name]" >&2
    exit 2
    ;;
esac
