#!/usr/bin/env bash
set -euo pipefail

# Offline only: unit/mock tests, no policy weights, camera or robot connection.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${ROOT}"
PYTHON=${MANIMUX_TEST_PYTHON:-${ROOT}/envs/yam/.venv/bin/python}
OPENPI_PYTHON=${OPENPI_TEST_PYTHON:-${ROOT}/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/src:${ROOT}:${ROOT}/XPolicyLab:${PYTHONPATH:-}"

for policy in Pi_05 LingBot_VLA2 Xiaomi_Robotics_1 OpenWAM; do
    for script in "${ROOT}/XPolicyLab/policy/${policy}"/*.sh; do
        bash -n "${script}"
    done
done
for script in scripts/training/train_{pi05,lingbot_vla2,xr1,openwam}_yam_cluster.sh; do
    bash -n "${script}"
done
"${PYTHON}" -m pytest -o addopts='' -q -p no:cacheprovider tests \
    XPolicyLab/tests/unit/test_isaac05_xpolicy_adapter.py \
    XPolicyLab/tests/unit/test_openwam_artifact_identity.py
JAX_PLATFORMS=cpu \
PYTHONPATH="${ROOT}/XPolicyLab/policy/Pi_05/openpi/src:${PYTHONPATH}" \
    "${OPENPI_PYTHON}" -m pytest -o addopts='' -q -p no:cacheprovider \
    XPolicyLab/policy/Pi_05/openpi/src/openpi/policies/yam_policy_test.py
