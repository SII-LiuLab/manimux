"""Flange adapter checks against a fake SDK and the real offline libKine."""

from __future__ import annotations

import platform
import subprocess
import sys

import numpy as np
import pytest

from manimux.embodiments.arm.tianji.kinematics import DEFAULT_CONFIG, TianjiSDKKinematics
from manimux.embodiments.arm.tianji.sdk.marvin import fx_kine
from manimux.kinematics.base import FlangeKinematicsBase


class FakeKine:
    def __init__(self):
        self.arm = None
        self.params = None
        self.fk_joints = None
        self.outcome = "success"
        self.load_ok = True
        self.init_ok = True
        self.remove_ok = True
        self.tool_removed = False
        self.fk_result = np.eye(4)
        self.fk_result[:3, 3] = [100, -200, 300]

    def log_switch(self, value):
        pass

    def load_config(self, arm, path):
        self.arm = arm
        if not self.load_ok:
            return None
        return {
            "TYPE": [1017, 1017],
            "DH": [[[0.0] * 4 for _ in range(8)]] * 2,
            "PNVA": [[[170.0, -170.0, 180.0, 450.0]] * 7] * 2,
            "BD": [[[0.0] * 3 for _ in range(4)]] * 2,
        }

    def initial_kine(self, robot_type, dh, pnva, j67):
        return self.init_ok

    def remove_tool_kine(self):
        self.tool_removed = True
        return self.remove_ok

    def fk(self, joints):
        self.fk_joints = joints
        return self.fk_result

    def ik(self, structure_data):
        params = structure_data
        self.params = params
        for i, value in enumerate(params.m_Input_IK_RefJoint.to_list()):
            params.m_Output_RetJoint.data[i] = value
        if self.outcome == "unreachable":
            params.m_Output_IsOutRange = True
        elif self.outcome == "singular":
            params.m_Output_IsDeg[3] = True
        elif self.outcome == "joint_limit":
            params.m_Output_IsJntExd = True
        elif self.outcome == "numeric_limit":
            params.m_Output_RetJoint.data[0] = 180
        elif self.outcome == "invalid":
            params.m_Output_RetJoint.data[0] = float("nan")
        return (
            False
            if self.outcome in {"no_solution", "unreachable", "singular", "joint_limit"}
            else params
        )

    def ik_nsp(self, sturcture_data):
        self.nsp_called = True
        return self.outcome != "nsp_failed"

    @staticmethod
    def mat4x4_to_mat1x16(matrix):
        return np.asarray(matrix).ravel().tolist()

    @staticmethod
    def mat4x4_to_xyzabc(pose_mat):
        from scipy.spatial.transform import Rotation

        matrix = np.asarray(pose_mat)
        return np.r_[
            matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
        ].tolist()


@pytest.fixture
def sdk(monkeypatch):
    fake = FakeKine()
    monkeypatch.setattr(fx_kine, "Marvin_Kine", lambda: fake)
    return fake


@pytest.mark.parametrize(("arm", "index"), [("left", 0), ("right", 1)])
def test_sdk_units_arm_selection_and_result(sdk, arm, index):
    kin = TianjiSDKKinematics(arm)
    assert isinstance(kin, FlangeKinematicsBase)
    assert kin.num_arm_joints == 7 and sdk.arm == index
    q = np.radians([10, -20, 30, -40, 50, -20, 30])
    pose = kin.fk_flange(q)
    np.testing.assert_allclose(sdk.fk_joints, np.degrees(q))
    np.testing.assert_allclose(pose[:3, 3], [0.1, -0.2, 0.3])
    result = kin.ik_flange(pose, q)
    assert result.converged and sdk.tool_removed and sdk.nsp_called
    np.testing.assert_allclose(result.joints, q)
    np.testing.assert_allclose(sdk.params.m_Input_IK_RefJoint.to_list(), np.degrees(q))
    expected = np.eye(4)
    expected[:3, 3] = [100, -200, 300]
    np.testing.assert_allclose(sdk.params.m_Input_IK_TargetTCP.to_list(), expected.ravel())
    assert sdk.params.m_Input_IK_ZSPType == 0
    np.testing.assert_allclose(pose[:3, 3], [0.1, -0.2, 0.3])  # input is not mutated


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        ("unreachable", "ik_failed"),
        ("singular", "ik_failed"),
        ("joint_limit", "ik_failed"),
        ("numeric_limit", "joint_limit"),
        ("no_solution", "ik_failed"),
        ("nsp_failed", "ik_nsp_failed"),
    ],
)
def test_rejected_ik_never_returns_commandable_joints(sdk, outcome, reason):
    kin = TianjiSDKKinematics()
    sdk.outcome = outcome
    result = kin.ik_flange(kin.fk_flange(np.zeros(7)), np.zeros(7))
    assert not result.converged and result.joints is None and result.reason == reason


@pytest.mark.parametrize("kind", ["position", "rotation"])
def test_success_flag_is_not_enough_without_fk_validation(sdk, kind):
    kin = TianjiSDKKinematics()
    target = kin.fk_flange(np.zeros(7))
    if kind == "position":
        target[0, 3] += 0.01
    else:
        target[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    result = kin.ik_flange(target, np.zeros(7))
    assert not result.converged and result.reason == "fk_mismatch"


@pytest.mark.parametrize("setting", ["load_ok", "init_ok", "remove_ok"])
def test_initialization_errors_are_not_solve_rejections(sdk, setting):
    setattr(sdk, setting, False)
    with pytest.raises(RuntimeError):
        TianjiSDKKinematics()


def test_backend_failure_and_invalid_output_raise(sdk):
    kin = TianjiSDKKinematics()
    sdk.outcome = "invalid"
    with pytest.raises(RuntimeError, match="invalid joint"):
        kin.ik_flange(np.eye(4), np.zeros(7))
    sdk.fk_result = False
    with pytest.raises(RuntimeError, match="FK failed"):
        kin.fk_flange(np.zeros(7))
    sdk.fk_result = np.zeros((4, 4))
    with pytest.raises(ValueError, match="rigid"):
        kin.fk_flange(np.zeros(7))


@pytest.mark.parametrize("q", [np.zeros(8), np.zeros((7, 1)), np.full(7, np.nan)])
def test_invalid_joint_inputs(sdk, q):
    kin = TianjiSDKKinematics()
    with pytest.raises(ValueError):
        kin.fk_flange(q)
    with pytest.raises(ValueError):
        kin.ik_flange(np.eye(4), q)


@pytest.mark.parametrize(
    "target", [np.zeros((4, 4)), np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan), np.eye(3)]
)
def test_invalid_pose_inputs(sdk, target):
    with pytest.raises(ValueError):
        TianjiSDKKinematics().ik_flange(target, np.zeros(7))


def test_invalid_configuration_fails_early(sdk):
    with pytest.raises(ValueError, match="arm"):
        TianjiSDKKinematics("invalid")


def test_import_does_not_load_vendor_or_legacy_modules():
    code = """
import sys
from manimux.embodiments.arm.tianji import TianjiSDKKinematics
assert not any(n.startswith('manimux.embodiments.arm.tianji.sdk') for n in sys.modules)
assert 'manimux.kinematics.tianji' not in sys.modules
assert 'manimux.robots.tianji.sdk' not in sys.modules
assert not any('backup' in n for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


native_sdk = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() not in {"x86_64", "AMD64"},
    reason="bundled libKine targets Linux x86-64",
)


@native_sdk
@pytest.mark.parametrize("arm", ["left", "right"])
def test_native_fk_and_ik_round_trip(arm):
    kin = TianjiSDKKinematics(arm)
    np.testing.assert_allclose(kin.fk_flange(np.zeros(7))[:3, 3], [0, 0, 0.8705], atol=1e-12)
    sign = 1 if arm == "left" else -1
    q = np.radians([sign * 21.8, -41, sign * -4.74, -63.67, sign * 10.15, 14.72, sign * 7.68])
    target = kin.fk_flange(q)
    for seed in (q, q + np.radians([0.1, 0, 0, 0, 0, 0, 0])):
        result = kin.ik_flange(target, seed)
        assert result.converged, result.reason
        np.testing.assert_allclose(kin.fk_flange(result.joints), target, atol=1e-7)
    target[0, 3] = 10
    result = kin.ik_flange(target, q)
    assert not result.converged and result.joints is None


@native_sdk
def test_native_instances_restore_model_and_remove_tool_offsets(tmp_path):
    original = TianjiSDKKinematics()
    q = np.zeros(7)
    baseline = original.fk_flange(q)
    config = tmp_path / "different.MvKDCfg"
    config.write_text(DEFAULT_CONFIG.read_text().replace("174.500000", "184.500000"))
    other = TianjiSDKKinematics(config_path=config)
    shifted = other.fk_flange(q)
    assert shifted[2, 3] == pytest.approx(baseline[2, 3] + 0.01)
    np.testing.assert_allclose(original.fk_flange(q), baseline, atol=1e-12)
    np.testing.assert_allclose(other.fk_flange(q), shifted, atol=1e-12)
    tool = np.eye(4)
    tool[2, 3] = 100  # deliberately contaminate the native slot with a 100 mm tool
    assert other._kine.set_tool_kine(tool.tolist())
    np.testing.assert_allclose(original.fk_flange(q), baseline, atol=1e-12)
