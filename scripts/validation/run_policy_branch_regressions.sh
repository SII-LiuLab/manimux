#!/usr/bin/env bash
set -euo pipefail

# Offline only: unit/mock tests, no policy weights, camera or robot connection.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${ROOT}"
PYTHON=${MANIMUX_TEST_PYTHON:-${ROOT}/envs/yam/.venv/bin/python}
OPENPI_PYTHON=${OPENPI_TEST_PYTHON:-${ROOT}/third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}:${ROOT}/third_party:${ROOT}/third_party/XPolicyLab:${PYTHONPATH:-}"

for policy in Pi_05 LingBot_VLA2 Xiaomi_Robotics_1 OpenWAM; do
    for script in "${ROOT}/third_party/XPolicyLab/policy/${policy}"/*.sh; do
        bash -n "${script}"
    done
done
"${PYTHON}" -m pytest -o addopts='' -q -p no:cacheprovider tests \
    third_party/XPolicyLab/tests/unit/test_isaac05_xpolicy_adapter.py \
    third_party/XPolicyLab/tests/unit/test_openwam_artifact_identity.py
JAX_PLATFORMS=cpu \
PYTHONPATH="${ROOT}/third_party/XPolicyLab/policy/Pi_05/openpi/src:${PYTHONPATH}" \
    "${OPENPI_PYTHON}" -m pytest -o addopts='' -q -p no:cacheprovider \
    third_party/XPolicyLab/policy/Pi_05/openpi/src/openpi/policies/yam_policy_test.py
