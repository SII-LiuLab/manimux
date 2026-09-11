#!/usr/bin/env bash
set -euo pipefail

mode=${1:-train}
run_name=${2:-yam-v1-s0-4xh100-3k}

ROOT=${YAM_TRAIN_ROOT:-/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=${LINGBOT_VLA2_WORKSPACE:-${REPO_ROOT}}
POLICY=${WORKSPACE}/XPolicyLab/policy/LingBot_VLA2
SOURCE=${POLICY}/lingbot_vla_v2
VENV=${ROOT}/envs/lingbot-vla2/.venv
DATASET=${LINGBOT_VLA2_DATASET_PATH:-}
DATASET_NAME=${LINGBOT_VLA2_DATASET_NAME:-${DATASET##*/}}
MODEL=${LINGBOT_VLA2_MODEL_PATH:-${ROOT}/weights/base/lingbot-vla-v2-6b}
TOKENIZER=${LINGBOT_VLA2_TOKENIZER_PATH:-${ROOT}/weights/base/xiaomi/qwen3_vl_4b_processor}
TRAINING_CONFIG=${LINGBOT_VLA2_TRAINING_CONFIG:-${POLICY}/training/yam_dual.yaml}
ROBOT_CONFIG_ROOT=${LINGBOT_VLA2_ROBOT_CONFIG_ROOT:-${POLICY}/robot_configs}
ROBOT_NAME=${LINGBOT_VLA2_ROBOT_NAME:-yam_dual_packed_absolute}
TRAINING_ROBOT_CONFIG=${ROBOT_CONFIG_ROOT}/${ROBOT_NAME}.yaml
DEPLOY_ROBOT_CONFIG=${LINGBOT_VLA2_DEPLOY_ROBOT_CONFIG:-${TRAINING_ROBOT_CONFIG}}
STATS_DIR=${LINGBOT_VLA2_STATS_DIR:-${ROOT}/cache/lingbot-vla2/${DATASET_NAME}}
STATS=${STATS_DIR}/norm_stats.json
RESOLVED_TRAINING_CONFIG=${LINGBOT_VLA2_RESOLVED_TRAINING_CONFIG:-${STATS_DIR}/training.yaml}
OUTPUT=${ROOT}/weights/finetuned/lingbot-vla2/${run_name}
LOG_DIR=${ROOT}/runs/lingbot-vla2
GPU_IDS=${LINGBOT_VLA2_GPU_IDS:-0,1,2,3}

export PATH="${ROOT}/envs/bin:${PATH}"
export HF_HOME=${ROOT}/cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export UV_CACHE_DIR=${ROOT}/cache/uv
export UV_INDEX_URL=${UV_INDEX_URL:-http://nexus.sii.shaipower.online/repository/pypi/simple}
export UV_INSECURE_HOST=${UV_INSECURE_HOST:-nexus.sii.shaipower.online}
export PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-${UV_INDEX_URL}}
export LINGBOT_VLA2_LEROBOT_SPEC=${LINGBOT_VLA2_LEROBOT_SPEC:-lerobot==0.4.2}
export MAX_JOBS=${MAX_JOBS:-16}
export TORCH_HOME=${ROOT}/cache/torch
export LINGBOT_VLA2_ENV_DIR=${VENV}
export LINGBOT_VLA2_MODEL_PATH=${MODEL}
export LINGBOT_VLA2_TOKENIZER_PATH=${TOKENIZER}
export LINGBOT_VLA2_DATASET_PATH=${DATASET}
export LINGBOT_VLA2_NORM_STATS_PATH=${STATS}
export LINGBOT_VLA2_ACTION_HORIZON=${LINGBOT_VLA2_ACTION_HORIZON:-50}
export LINGBOT_VLA2_NATIVE_HZ=${LINGBOT_VLA2_NATIVE_HZ:-30}
export LINGBOT_VLA2_TRAIN_WORKERS=${LINGBOT_VLA2_TRAIN_WORKERS:-8}
export LINGBOT_VLA2_MICRO_BATCH_SIZE=${LINGBOT_VLA2_MICRO_BATCH_SIZE:-1}
export LINGBOT_VLA2_GRAD_ACCUM_STEPS=${LINGBOT_VLA2_GRAD_ACCUM_STEPS:-8}
export LINGBOT_VLA2_USE_WANDB=false
export LINGBOT_VLA2_ENABLE_RESUME=false

WORKSPACE_PYTHONPATH=${SOURCE}:${WORKSPACE}/XPolicyLab:${WORKSPACE}/src
if [[ -n "${PYTHONPATH:-}" ]]; then
  WORKSPACE_PYTHONPATH=${WORKSPACE_PYTHONPATH}:${PYTHONPATH}
fi

mkdir -p "${STATS_DIR}" "${LOG_DIR}"

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
  require_value LINGBOT_VLA2_DATASET_PATH "${DATASET}"
  require_value LINGBOT_VLA2_DATASET_NAME "${DATASET_NAME}"
}

install_environment() {
  if [[ -x "${VENV}/bin/python" ]] && "${VENV}/bin/python" -c \
      'import flash_attn, lingbotvla, mdm, moge, torch, utils3d; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    echo "[LingBot_VLA2] reusing ${VENV}"
    return
  fi
  echo "[LingBot_VLA2] installing environment at ${VENV}"
  bash "${POLICY}/install.sh"
}

resolve_training_config() {
  require_file "${TRAINING_CONFIG}"
  "${VENV}/bin/python" - \
    "${TRAINING_CONFIG}" \
    "${RESOLVED_TRAINING_CONFIG}" \
    "${LINGBOT_VLA2_MOGE_PATH:-}" \
    "${LINGBOT_VLA2_MORGBD_PATH:-}" \
    "${LINGBOT_VLA2_DINO_VIDEO_CKPT:-}" \
    "${LINGBOT_VLA2_DINO_VIDEO_CONFIG:-}" <<'PY'
import sys
from pathlib import Path

import yaml

source, destination = map(Path, sys.argv[1:3])
moge_path, morgbd_path, video_ckpt, video_config = sys.argv[3:]
config = yaml.safe_load(source.read_text())
align = config.get("train", {}).get("align_params")
if align is not None:
    depth = align.setdefault("depth", {})
    video = align.setdefault("video", {})
    if moge_path:
        depth["moge_path"] = moge_path
    if morgbd_path:
        depth["morgbd_path"] = morgbd_path
    if video_ckpt:
        video["ckpt_path"] = video_ckpt
    if video_config:
        video["config_path"] = video_config
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(yaml.safe_dump(config, sort_keys=False))
PY
}

compute_stats() {
  if [[ -s "${STATS}" ]]; then
    echo "[LingBot_VLA2] reusing norm stats: ${STATS}"
    return
  fi
  echo "[LingBot_VLA2] computing norm stats from ${DATASET}"
  local stats_gpu=${GPU_IDS%%,*}
  (
    cd "${SOURCE}"
    CUDA_VISIBLE_DEVICES=${stats_gpu} PATH="${VENV}/bin:${PATH}" \
      PYTHONPATH="${WORKSPACE_PYTHONPATH}" \
      bash -o pipefail train.sh scripts/compute_norm_stats.py configs/vla/norm_compute/post_data.yaml \
        --data.data_name "${ROBOT_NAME}" \
        --data.robot_name "${ROBOT_NAME}" \
        --data.train_path "${DATASET}" \
        --data.robot_config_root "${ROBOT_CONFIG_ROOT}" \
        --data.norm_path "${STATS}" \
        --data.num_workers "${LINGBOT_VLA2_STATS_WORKERS:-8}" \
        --train.chunk_size "${LINGBOT_VLA2_ACTION_HORIZON}" \
        --train.micro_batch_size "${LINGBOT_VLA2_STATS_BATCH_SIZE:-32}" \
        --train.output_dir "${STATS_DIR}/compute"
  )
  require_file "${STATS}"
}

preflight() {
  require_file "${SOURCE}/tasks/vla/train_lingbotvla.py"
  require_file "${MODEL}/model.safetensors.index.json"
  require_file "${TOKENIZER}/tokenizer.json"
  require_file "${DATASET}/meta/info.json"
  require_file "${TRAINING_ROBOT_CONFIG}"
  require_file "${DEPLOY_ROBOT_CONFIG}"
  require_file "${TRAINING_CONFIG}"
  require_file "${RESOLVED_TRAINING_CONFIG}"
  require_file "${STATS}"
  for artifact in \
    "${LINGBOT_VLA2_MOGE_PATH:-}" \
    "${LINGBOT_VLA2_MORGBD_PATH:-}" \
    "${LINGBOT_VLA2_DINO_VIDEO_CKPT:-}" \
    "${LINGBOT_VLA2_DINO_VIDEO_CONFIG:-}"; do
    if [[ -n "${artifact}" ]]; then
      require_file "${artifact}"
    fi
  done
  "${VENV}/bin/python" - \
    "${DATASET}" \
    "${STATS}" \
    "${LINGBOT_VLA2_EE_AUX:-false}" \
    "${LINGBOT_VLA2_EXPECTED_EPISODES:-}" \
    "${LINGBOT_VLA2_EXPECTED_FRAMES:-}" <<'PY'
import json
import sys
from pathlib import Path

dataset, stats_path = map(Path, sys.argv[1:3])
ee_aux = sys.argv[3].lower() == "true"
expected_episodes = int(sys.argv[4]) if sys.argv[4] else None
expected_frames = int(sys.argv[5]) if sys.argv[5] else None
info = json.loads((dataset / "meta/info.json").read_text())
assert info["codebase_version"] == "v3.0"
assert info["total_episodes"] > 0
assert info["total_frames"] > 0
if expected_episodes is not None:
    assert info["total_episodes"] == expected_episodes
if expected_frames is not None:
    assert info["total_frames"] == expected_frames
assert info["features"]["observation.state"]["shape"] == [14]
assert info["features"]["action"]["shape"] == [14]
if ee_aux:
    assert info["features"]["observation.ee_pose"]["shape"] == [24]
    assert info["features"]["action.ee_pose"]["shape"] == [24]
    assert info["features"]["observation.state.end.position"]["shape"] == [14]
    assert info["features"]["action.end.position"]["shape"] == [14]
stats = json.loads(stats_path.read_text())
assert stats["count"] == info["total_frames"]
expected = {
    "observation.state.arm.position": 12,
    "observation.state.effector.position": 2,
    "action.arm.position": 12,
    "action.effector.position": 2,
}
if ee_aux:
    expected.update({
        "observation.state.end.position": 14,
        "action.end.position": 14,
    })
for key, width in expected.items():
    assert len(stats["norm_stats"][key]["mean"]) == width
print(json.dumps({
    "dataset": dataset.name,
    "episodes": info["total_episodes"],
    "frames": info["total_frames"],
    "fps": info["fps"],
    "ee_aux": ee_aux,
    "gpus": __import__("torch").cuda.device_count(),
}, indent=2))
PY
}

run_training() {
  local train_args=(
    tasks/vla/train_lingbotvla.py "${RESOLVED_TRAINING_CONFIG}"
    --model.model_path "${MODEL}"
    --model.tokenizer_path "${TOKENIZER}"
    --data.data_name "${ROBOT_NAME}"
    --data.train_path "${DATASET}"
    --data.robot_config_root "${ROBOT_CONFIG_ROOT}"
    --data.norm_stats_file "${STATS}"
    --data.num_workers "${LINGBOT_VLA2_TRAIN_WORKERS:-8}"
    --train.output_dir "${OUTPUT}"
    --train.seed 0
    --train.chunk_size "${LINGBOT_VLA2_ACTION_HORIZON}"
    --train.micro_batch_size "${LINGBOT_VLA2_MICRO_BATCH_SIZE}"
    --train.gradient_accumulation_steps "${LINGBOT_VLA2_GRAD_ACCUM_STEPS}"
    --train.max_steps "${LINGBOT_VLA2_MAX_STEPS:-60000}"
    --train.save_steps "${LINGBOT_VLA2_SAVE_STEPS:-1000}"
    --train.enable_resume false
    --train.use_wandb false
  )
  if [[ -n "${LINGBOT_VLA2_GLOBAL_BATCH_SIZE:-}" ]]; then
    train_args+=(--train.global_batch_size "${LINGBOT_VLA2_GLOBAL_BATCH_SIZE}")
  fi

  (
    cd "${SOURCE}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} PATH="${VENV}/bin:${PATH}" \
      PYTHONPATH="${WORKSPACE_PYTHONPATH}" \
      bash -o pipefail train.sh "${train_args[@]}"
  ) 2>&1 | tee "${LOG_DIR}/${run_name}.log"

  cp -f "${STATS}" "${OUTPUT}/norm_stats.json"
  cp -f "${DEPLOY_ROBOT_CONFIG}" "${OUTPUT}/robot_config.yaml"
  cp -f "${TRAINING_ROBOT_CONFIG}" "${OUTPUT}/training_robot_config.yaml"
}

case "${mode}" in
  gate-train)
    LINGBOT_VLA2_MAX_STEPS=1 LINGBOT_VLA2_SAVE_STEPS=1 \
      bash "$0" smoke "${run_name}-smoke"
    require_file "${ROOT}/weights/finetuned/lingbot-vla2/${run_name}-smoke/checkpoints/global_step_1/hf_ckpt/model.safetensors.index.json"
    LINGBOT_VLA2_MAX_STEPS=${LINGBOT_VLA2_MAX_STEPS:-3000} LINGBOT_VLA2_SAVE_STEPS=${LINGBOT_VLA2_SAVE_STEPS:-500} \
      bash "$0" train "${run_name}"
    ;;
  prepare)
    validate_configuration
    install_environment
    resolve_training_config
    compute_stats
    preflight
    ;;
  smoke|train)
    validate_configuration
    install_environment
    resolve_training_config
    compute_stats
    preflight
    if [[ "${mode}" == "smoke" ]]; then
      export LINGBOT_VLA2_MAX_STEPS=${LINGBOT_VLA2_MAX_STEPS:-1}
      export LINGBOT_VLA2_SAVE_STEPS=${LINGBOT_VLA2_SAVE_STEPS:-1}
    else
      export LINGBOT_VLA2_MAX_STEPS=${LINGBOT_VLA2_MAX_STEPS:-3000}
      export LINGBOT_VLA2_SAVE_STEPS=${LINGBOT_VLA2_SAVE_STEPS:-500}
    fi
    run_training
    ;;
  *)
    echo "Usage: $0 [prepare|smoke|train|gate-train] [run_name]" >&2
    exit 2
    ;;
esac
