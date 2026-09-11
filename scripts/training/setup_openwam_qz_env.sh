#!/usr/bin/env bash
set -euo pipefail

# Build an isolated OpenWAM environment while reusing the CUDA/Torch stack
# provided by the QZ training image. This script does not create a QZ job.
TRAIN_ROOT=${OPENWAM_TRAIN_ROOT:-/inspire/hdd3/project/agent-driven-world-model/ky26300/manimux_training}
WORKSPACE=${OPENWAM_WORKSPACE:-${TRAIN_ROOT}/operate/manimux-openwam}
SYSTEM_PYTHON=${OPENWAM_QZ_SYSTEM_PYTHON:-/usr/bin/python3}
VENV=${OPENWAM_VENV:-${TRAIN_ROOT}/envs/openwam/.venv}

[[ -x "${SYSTEM_PYTHON}" ]] || { echo "QZ system Python not found: ${SYSTEM_PYTHON}" >&2; exit 2; }
[[ -f "${WORKSPACE}/XPolicyLab/pyproject.toml" ]] || { echo "XPolicyLab checkout not found: ${WORKSPACE}" >&2; exit 2; }

if [[ ! -x "${VENV}/bin/python" ]]; then
    "${SYSTEM_PYTHON}" -m venv --system-site-packages "${VENV}"
fi

"${VENV}/bin/python" -m pip install --upgrade \
    pip==26.2.1 setuptools==84.0.0 wheel==0.48.0 packaging==24.2
"${VENV}/bin/python" -m pip install numpy==1.26.4
"${VENV}/bin/python" -m pip install -e "${WORKSPACE}/XPolicyLab" \
    -e "${WORKSPACE}/XPolicyLab/policy/OpenWAM/OpenWAM"
"${VENV}/bin/python" -m pip check

PYTHONPATH="${WORKSPACE}:${WORKSPACE}/XPolicyLab/policy/OpenWAM/OpenWAM:${PYTHONPATH:-}" \
    "${VENV}/bin/python" - <<'PY'
import cv2
import numpy
import torch
import torchvision
import transformers
import accelerate
import deepspeed
import hydra
import h5py
import openwam
import XPolicyLab

print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "numpy": numpy.__version__,
    "opencv": cv2.__version__,
})
PY
