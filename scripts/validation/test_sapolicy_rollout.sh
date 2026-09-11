#!/usr/bin/env bash
set -euo pipefail
# Offline regression only: mock robot, no cameras, CAN, model weights or service reload.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${ROOT}"
PYTHON=${MANIMUX_TEST_PYTHON:-${ROOT}/envs/yam/.venv/bin/python}
export PYTHONPATH="${ROOT}/src:${ROOT}:${ROOT}/XPolicyLab:${PYTHONPATH:-}"
"${PYTHON}" -m pytest -o addopts='' -q \
    tests/unit/test_config.py tests/unit/test_executors.py \
    tests/unit/test_kinematics.py tests/unit/test_timeline.py \
    tests/unit/test_yam_interrupt_safety.py tests/unit/test_rtc_runtime.py \
    tests/unit/test_sapolicy_decode_timing.py \
    tests/unit/test_sapolicy_xpl_model.py tests/unit/test_sapolicy_xpl_transport.py \
    tests/unit/test_pi05_algorithm_configs.py tests/unit/test_xiaomi_xr1_server_config.py \
    tests/integration/test_action_decoder.py tests/integration/test_mock_run.py
