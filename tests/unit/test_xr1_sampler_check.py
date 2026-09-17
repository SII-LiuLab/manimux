from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manimux.integrations.xr1_yam.codec import ACTION_DIM, ACTION_PARTS, STATE_DIM, rotm2aa_batch


def test_xr1_codec_preserves_action_layout() -> None:
    assert ACTION_DIM == STATE_DIM == 60
    expected_parts = (
        ("left_ee_pos", slice(0, 3)),
        ("left_ee_aa", slice(3, 6)),
        ("left_gripper", slice(6, 7)),
        ("right_ee_pos", slice(8, 11)),
        ("right_ee_aa", slice(11, 14)),
        ("right_gripper", slice(14, 15)),
        ("waist", slice(16, 17)),
        ("base", slice(17, 20)),
    )
    assert expected_parts == ACTION_PARTS


@pytest.mark.parametrize(
    "rotvec",
    [
        [0.0, 0.0, 0.0],
        [np.pi, 0.0, 0.0],
        [0.0, np.pi, 0.0],
        [0.0, 0.0, np.pi],
        [0.2, -0.4, 0.6],
    ],
)
def test_xr1_codec_preserves_rotations(rotvec: list[float]) -> None:
    rotations = Rotation.from_rotvec([rotvec]).as_matrix()
    result = rotm2aa_batch(rotations)
    assert result.shape == (1, 3)
    np.testing.assert_allclose(Rotation.from_rotvec(result).as_matrix(), rotations, atol=1e-6)


def test_xr1_codec_does_not_import_model_dependencies() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import manimux.integrations.xr1_yam.codec; "
            "assert not {'torch', 'transformers', 'mibot'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_xr1_sampler_check_runs_without_model_or_gpu() -> None:
    python = Path("envs/xr1/.venv/bin/python")
    if not python.is_file():
        pytest.skip("XR-1 environment is not installed")
    result = subprocess.run(
        [str(python), "scripts/validation/check_xr1_rtc_sampler.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["model_constructed"] is False
    assert payload["gpu_used"] is False
    assert payload["xpolicy"]["conditioned_inside_generate"] is True
