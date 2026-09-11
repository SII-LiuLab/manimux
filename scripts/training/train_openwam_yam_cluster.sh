#!/usr/bin/env bash
set -euo pipefail

# QZ-ready OpenWAM/YAM launcher. This script never allocates QZ resources; it
# runs inside an already allocated shell/job and exposes a CPU-only ready gate.
mode=${1:-ready}
run=${2:-yam-openwam}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=${OPENWAM_WORKSPACE:-${REPO}}
POLICY=${WORKSPACE}/XPolicyLab/policy/OpenWAM
TRAIN_ROOT=${OPENWAM_TRAIN_ROOT:-${WORKSPACE}}

export OPENWAM_PYTHON=${OPENWAM_PYTHON:-${TRAIN_ROOT}/envs/openwam/.venv/bin/python}
export OPENWAM_DATASET_DIR=${OPENWAM_DATASET_DIR:?Set OPENWAM_DATASET_DIR to prepared native HDF5}
if [[ -z "${OPENWAM_RESUME_CKPT_PATH:-}" ]]; then
    export OPENWAM_FINETUNE_CKPT_PATH=${OPENWAM_FINETUNE_CKPT_PATH:-${TRAIN_ROOT}/weights/base/openwam/OpenWAM-Alpha-Pretrain-Foundation-Model}
fi
CHECKPOINT_SOURCE=${OPENWAM_RESUME_CKPT_PATH:-${OPENWAM_FINETUNE_CKPT_PATH:-}}
GPU_IDS=${OPENWAM_GPU_IDS:-0,1,2,3}
steps=${OPENWAM_MAX_STEPS:-30000}
save=${OPENWAM_SAVE_INTERVAL:-5000}
batch=${OPENWAM_BATCH_SIZE:-1}
accum=${OPENWAM_GRADIENT_ACCUMULATION_STEPS:-8}
keep=${OPENWAM_KEEP_LAST_K_CKPTS:-6}
zero=${OPENWAM_ZERO_STAGE:-2}
if [[ -n "${OPENWAM_RESUME_CKPT_PATH:-}" ]]; then
    EXPECTED_SHA=${OPENWAM_EXPECTED_RESUME_SHA256:-}
else
    EXPECTED_SHA=${OPENWAM_EXPECTED_FOUNDATION_SHA256:-180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d}
fi

case "${run}" in ""|.|..|*/*) echo "run_name must be a directory name" >&2; exit 2 ;; esac
[[ -x "${OPENWAM_PYTHON}" ]] || { echo "Existing Python not found: ${OPENWAM_PYTHON}" >&2; exit 2; }
export PYTHONPATH="${WORKSPACE}:${POLICY}/OpenWAM:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

if [[ -n "${OPENWAM_RESUME_CKPT_PATH:-}" ]]; then
    if [[ "${mode}" == smoke || "${mode}" == gate-train ]]; then
        echo "Resume is only allowed with train, never smoke/gate-train" >&2; exit 2
    fi
    if [[ -n "${OPENWAM_FINETUNE_CKPT_PATH:-}" ]]; then
        echo "Unset OPENWAM_FINETUNE_CKPT_PATH when resuming" >&2; exit 2
    fi
fi

LOG_DIR=${OPENWAM_LOG_DIR:-${TRAIN_ROOT}/runs/openwam/logs}
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/${run}-${mode}.log") 2>&1
printf '[OpenWAM] mode=%s run=%s python=%s\n' "${mode}" "${run}" "${OPENWAM_PYTHON}"

preflight() {
  "${OPENWAM_PYTHON}" - "${WORKSPACE}" "${OPENWAM_DATASET_DIR}" \
    "${CHECKPOINT_SOURCE}" "${EXPECTED_SHA}" \
    "${OPENWAM_EXPECTED_EPISODES:-}" "${OPENWAM_EXPECTED_FRAMES:-}" <<'PY'
import hashlib
import importlib
import json
from pathlib import Path
import sys

workspace, dataset, checkpoint, expected_sha, expected_episodes, expected_frames = sys.argv[1:]
for module in ("torch", "accelerate", "deepspeed", "hydra", "h5py", "cv2", "openwam", "XPolicyLab"):
    importlib.import_module(module)
dataset = Path(dataset)
manifest_path = dataset / "meta/openwam_yam_manifest.json"
if not manifest_path.is_file():
    raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
if manifest.get("embodiment") != "yam_dual" or manifest.get("action_mode") != "eef":
    raise ValueError(f"Unexpected dataset contract: {manifest}")
if manifest.get("frequency_hz") != 30 or manifest.get("action_target") != "next_achieved_state":
    raise ValueError(f"Unexpected temporal/action contract: {manifest}")
if expected_episodes and manifest.get("episodes") != int(expected_episodes):
    raise ValueError(("episodes", manifest.get("episodes"), int(expected_episodes)))
if expected_frames and manifest.get("frames") != int(expected_frames):
    raise ValueError(("frames", manifest.get("frames"), int(expected_frames)))
files = sorted(dataset.glob("*/yam_dual/data/episode_*.hdf5"))
if len(files) != manifest.get("episodes"):
    raise ValueError(("hdf5 files", len(files), manifest.get("episodes")))

checkpoint = Path(checkpoint)
config = checkpoint / "config.yaml"
weights = sorted(checkpoint.glob("checkpoint_step_*.safetensors"))
tokenizer = checkpoint / "tokenizer/google/umt5-xxl/tokenizer.json"
for path in (config, tokenizer):
    if not path.is_file():
        raise FileNotFoundError(f"Missing foundation artifact: {path}")
if not weights:
    raise ValueError(f"No checkpoint_step_*.safetensors found under {checkpoint}")
weights.sort(key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
weights = [weights[-1]]
digest = hashlib.sha256()
with weights[0].open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
actual_sha = digest.hexdigest()
if expected_sha and actual_sha != expected_sha:
    raise ValueError(("foundation sha256", actual_sha, expected_sha))
print(json.dumps({
    "workspace": str(Path(workspace).resolve()),
    "dataset": str(dataset.resolve()),
    "episodes": manifest["episodes"],
    "frames": manifest["frames"],
    "foundation": str(checkpoint.resolve()),
    "foundation_sha256": actual_sha,
}, indent=2))
PY
}

prepare() {
  bash "${POLICY}/process_data.sh" RoboDojo_real "${run}" yam_dual ee
}

training_args() {
  printf '%s\n' \
    "training.max_steps=${steps}" \
    "training.num_epochs=null" \
    "training.save_steps=${save}" \
    "training.batch_size=${batch}" \
    "training.gradient_accumulation_steps=${accum}" \
    "training.keep_last_k_ckpts=${keep}" \
    "training.zero_stage=${zero}" \
    "training.mixed_precision=bf16" \
    "training.use_gradient_checkpointing=true" \
    "training.save_full_states_for_resume=true" \
    "project.wandb.project=null"
}

run_training() {
  local dry=${1:-false}
  mapfile -t args < <(training_args)
  if [[ "${dry}" == true ]]; then args+=(--dry-run); fi
  bash "${POLICY}/train.sh" RoboDojo_real "${run}" yam_dual ee 0 "${GPU_IDS}" "${args[@]}"
}

resolve_deploy_checkpoint_dir() {
  "${OPENWAM_PYTHON}" - "${OPENWAM_CHECKPOINT_DIR}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
if (root / "config.yaml").is_file() and any(root.glob("checkpoint_step_*.safetensors")):
    print(root)
    raise SystemExit(0)

candidates = [
    path
    for path in root.iterdir()
    if path.is_dir()
    and (path / "config.yaml").is_file()
    and any(path.glob("checkpoint_step_*.safetensors"))
]
if not candidates:
    raise FileNotFoundError(f"no deployable OpenWAM checkpoint found under {root}")
print(max(candidates, key=lambda path: path.stat().st_mtime_ns))
PY
}

case "${mode}" in
  prepare)
    preflight
    prepare
    ;;
  dry-run)
    preflight
    export OPENWAM_CHECKPOINT_DIR=${OPENWAM_OUTPUT_ROOT:-${TRAIN_ROOT}/weights/finetuned/openwam}/${run}
    run_training true
    ;;
  ready)
    preflight
    prepare
    export OPENWAM_CHECKPOINT_DIR=${OPENWAM_OUTPUT_ROOT:-${TRAIN_ROOT}/weights/finetuned/openwam}/${run}
    run_training true
    ;;
  gate-train)
    OPENWAM_MAX_STEPS=1 OPENWAM_SAVE_INTERVAL=1 bash "$0" smoke "${run}"
    bash "$0" train "${run}"
    ;;
  smoke|train)
    preflight
    prepare
    if [[ "${mode}" == smoke ]]; then steps=1; save=1; run="${run}-smoke"; fi
    export OPENWAM_CHECKPOINT_DIR=${OPENWAM_OUTPUT_ROOT:-${TRAIN_ROOT}/weights/finetuned/openwam}/${run}
    if [[ -n "${OPENWAM_RESUME_CKPT_PATH:-}" ]]; then
        export OPENWAM_CHECKPOINT_DIR=${OPENWAM_RESUME_CKPT_PATH}
    fi
    run_training false
    deploy_checkpoint_dir=$(resolve_deploy_checkpoint_dir)
    "${OPENWAM_PYTHON}" "${WORKSPACE}/scripts/servers/openwam_yam_server.py" \
      --config "${WORKSPACE}/configs/openwam/yam/server/finetune.yaml" \
      --checkpoint "${deploy_checkpoint_dir}" --check
    ;;
  *) echo "Usage: $0 [prepare|dry-run|ready|smoke|train|gate-train] [run_name]" >&2; exit 2 ;;
esac
