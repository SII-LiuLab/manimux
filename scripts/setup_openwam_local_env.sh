#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_PYTHON=${OPENWAM_BASE_PYTHON:-${ROOT}/envs/yam/.venv/bin/python}
VENV=${OPENWAM_VENV:-${ROOT}/envs/openwam/.venv}
UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/manimux-uv-cache}

[[ -x "${BASE_PYTHON}" ]] || {
    echo "YAM base Python not found: ${BASE_PYTHON}" >&2
    exit 2
}

if [[ ! -x "${VENV}/bin/python" ]]; then
    UV_CACHE_DIR="${UV_CACHE_DIR}" uv venv --python "${BASE_PYTHON}" "${VENV}"
fi

base_site=$(
    "${BASE_PYTHON}" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
venv_site=$(
    "${VENV}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
printf '%s\n' "${base_site}" > "${venv_site}/yam_runtime_base.pth"

UV_CACHE_DIR="${UV_CACHE_DIR}" uv pip install --python "${VENV}/bin/python" --no-deps \
    'transformers==5.17.0' \
    'diffusers==0.40.0' \
    'modelscope==1.40.0' \
    'ftfy==6.3.1' \
    'deepspeed==0.18.9' \
    'hydra-core==1.3.6' \
    'av==17.1.0' \
    'timm==1.0.29' \
    'opencv-python-headless==4.14.0.94' \
    'huggingface-hub==1.31.0' \
    'tokenizers==0.23.2' \
    'regex==2026.9.10' \
    'typer==0.27.2' \
    'modelscope-hub==0.4.2' \
    'antlr4-python3-runtime==4.9.3' \
    'cryptography==50.0.1' \
    'cffi==2.1.1' \
    'wcwidth==0.8.3' \
    'py-cpuinfo==9.0.0' \
    'psutil==7.2.2' \
    'hjson==3.1.0' \
    'ninja==1.13.2' \
    'setuptools==84.0.0' \
    'packaging==26.3' \
    -e "${ROOT}/XPolicyLab" \
    -e "${ROOT}/XPolicyLab/policy/OpenWAM/OpenWAM"

"${VENV}/bin/python" - <<'PY'
import torch
import transformers
from openwam.deploy.server import build_server_from_config
import XPolicyLab

print(
    {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "transformers": transformers.__version__,
        "openwam_server_import": bool(build_server_from_config),
        "xpolicylab": XPolicyLab.__name__,
    }
)
PY
