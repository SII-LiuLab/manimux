#!/usr/bin/env bash
set -euo pipefail
# Use the active XPolicy environment; no environment creation or pip installs.
mode=${1:-prepare}
run=${2:-yam-openwam}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
POLICY=${REPO}/XPolicyLab/policy/OpenWAM
export OPENWAM_PYTHON=${OPENWAM_PYTHON:-python}
export OPENWAM_DATASET_DIR=${OPENWAM_DATASET_DIR:?Set OPENWAM_DATASET_DIR to prepared native HDF5}
GPU_IDS=${OPENWAM_GPU_IDS:-0}
steps=${OPENWAM_MAX_STEPS:-3000}
save=${OPENWAM_SAVE_INTERVAL:-500}
batch=${OPENWAM_BATCH_SIZE:-1}
case "${run}" in ""|.|..|*/*) echo "run_name must be a directory name" >&2; exit 2 ;; esac
command -v "${OPENWAM_PYTHON}" >/dev/null || { echo "Existing Python not found: ${OPENWAM_PYTHON}" >&2; exit 2; }
if [[ -n "${OPENWAM_RESUME_CKPT_PATH:-}" ]]; then
    if [[ "${mode}" == smoke || "${mode}" == gate-train ]]; then
        echo "Resume is only allowed with train, never smoke/gate-train" >&2; exit 2
    fi
    if [[ -n "${OPENWAM_FINETUNE_CKPT_PATH:-}" ]]; then
        echo "Unset OPENWAM_FINETUNE_CKPT_PATH when resuming" >&2; exit 2
    fi
fi
LOG_DIR=${OPENWAM_LOG_DIR:-${REPO}/data/training/openwam/logs}
mkdir -p "${LOG_DIR}"
# Append so repeated/resumed launches retain their previous diagnostics.
exec > >(tee -a "${LOG_DIR}/${run}-${mode}.log") 2>&1
printf '[OpenWAM] mode=%s run=%s python=%s\n' "${mode}" "${run}" "${OPENWAM_PYTHON}"
prepare() { bash "${POLICY}/process_data.sh" RoboDojo_real "${run}" yam_dual ee; }
case "${mode}" in
  prepare) prepare ;;
  gate-train)
    OPENWAM_MAX_STEPS=1 OPENWAM_SAVE_INTERVAL=1 bash "$0" smoke "${run}"
    bash "$0" train "${run}"
    ;;
  smoke|train)
    prepare
    if [[ "${mode}" == smoke ]]; then steps=1; save=1; run="${run}-smoke"; fi
    export OPENWAM_CHECKPOINT_DIR=${OPENWAM_OUTPUT_ROOT:-${POLICY}/checkpoints}/${run}
    if [[ -n "${OPENWAM_RESUME_CKPT_PATH:-}" ]]; then
        export OPENWAM_CHECKPOINT_DIR=${OPENWAM_RESUME_CKPT_PATH}
    fi
    bash "${POLICY}/train.sh" RoboDojo_real "${run}" yam_dual ee 0 "${GPU_IDS}" \
      "training.max_steps=${steps}" "training.num_epochs=null" \
      "training.save_steps=${save}" "training.batch_size=${batch}" \
      "training.save_full_states_for_resume=true"
    "${OPENWAM_PYTHON}" "${REPO}/scripts/servers/openwam_yam_server.py" \
      --config "${REPO}/configs/openwam/yam/server/finetune.yaml" \
      --checkpoint "${OPENWAM_CHECKPOINT_DIR}" --check
    ;;
  *) echo "Usage: $0 [prepare|smoke|train|gate-train] [run_name]" >&2; exit 2 ;;
esac
