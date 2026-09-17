import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manimux.kinematics import (
    ComposedManipulatorKinematics,
    FixedToolGeometry,
    FlangeKinematicsBase,
    IKResult,
    KinematicCoordinate,
    ToolGeometryBase,
)


def pose(xyz, angles):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    result[:3, 3] = xyz
    return result


class CartesianArm(FlangeKinematicsBase):
    num_arm_joints = 6

    def fk_flange(self, joints):
        return pose(joints[:3], joints[3:])

    def ik_flange(self, target_flange, seed_joints):
        self.target = target_flange.copy()
        self.seed = seed_joints.copy()
        return IKResult(
            True,
            np.r_[
                target_flange[:3, 3], Rotation.from_matrix(target_flange[:3, :3]).as_euler("xyz")
            ],
        )


class MovingTool(ToolGeometryBase):
    coordinates = (KinematicCoordinate("opening", "normalized"),)
    base_frame = "tool_base"
    tcp_frame = "grasp"

    def tcp_transform(self, tool_state=None):
        return pose([0.1, 0, 0.2 * tool_state[0]], [0.3, -0.2, 0.1])


def make_model(tool=None, arm=None, **options):
    return ComposedManipulatorKinematics(
        arm or CartesianArm(),
        tool or MovingTool(),
        arm_coordinates=tuple(
            KinematicCoordinate(f"j{i}", "m" if i < 3 else "rad") for i in range(6)
        ),
        mount=pose([0.03, -0.02, 0.1], [0.2, 0.4, -0.3]),
        base_frame="arm_base",
        **options,
    )


def test_fk_and_ik_use_target_tool_state_and_transform_order():
    arm = CartesianArm()
    model = make_model(arm=arm)
    q = np.array([0.4, -0.3, 0.2, 0.1, 0.2, -0.4, 0.8])
    expected = (
        pose(q[:3], q[3:6])
        @ pose([0.03, -0.02, 0.1], [0.2, 0.4, -0.3])
        @ pose([0.1, 0, 0.16], [0.3, -0.2, 0.1])
    )
    np.testing.assert_allclose(model.fk(q), expected)
    seed = np.zeros(7)
    result = model.ik(expected, seed, fixed_coordinates={"opening": 0.8})
    assert result.converged
    np.testing.assert_allclose(result.joints, q, atol=1e-12)
    np.testing.assert_allclose(arm.target, pose(q[:3], q[3:6]), atol=1e-12)
    np.testing.assert_array_equal(seed, np.zeros(7))
    assert model.base_frame == "arm_base" and model.tcp_frame == "grasp"


def test_tool_without_coordinates():
    model = make_model(tool=FixedToolGeometry(np.eye(4)))
    q = np.zeros(6)
    result = model.ik(model.fk(q), q, fixed_coordinates={})
    assert result.converged and result.joints.shape == (6,)


@pytest.mark.parametrize(
    "fixed,error",
    [
        ({}, NotImplementedError),
        ({"opening": 0.4, "j0": 0}, NotImplementedError),
        ({"missing": 0}, KeyError),
        ({"opening": np.nan}, ValueError),
        ({"opening": 2}, ValueError),
        ({"opening": [0.5]}, ValueError),
    ],
)
def test_bad_constraints(fixed, error):
    with pytest.raises(error):
        make_model().ik(np.eye(4), np.zeros(7), fixed_coordinates=fixed)


@pytest.mark.parametrize("state", [np.zeros(6), np.zeros((7, 1)), np.full(7, np.nan)])
def test_bad_state(state):
    with pytest.raises(ValueError):
        make_model().fk(state)


def test_bad_target():
    with pytest.raises(ValueError):
        make_model().ik(np.diag([2.0, 1, 1, 1]), np.zeros(7), fixed_coordinates={"opening": 0})


def test_flange_rejection_and_backend_error_propagate(monkeypatch):
    arm = CartesianArm()
    model = make_model(arm=arm)
    monkeypatch.setattr(arm, "ik_flange", lambda *args: IKResult(False, reason="joint_limit"))
    result = model.ik(np.eye(4), np.zeros(7), fixed_coordinates={"opening": 0})
    assert not result.converged and result.reason == "joint_limit" and result.joints is None

    def fail(*args):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(arm, "ik_flange", fail)
    with pytest.raises(RuntimeError, match="backend unavailable"):
        model.ik(np.eye(4), np.zeros(7), fixed_coordinates={"opening": 0})


@pytest.mark.parametrize("index", [0, 5])
def test_tcp_residual_rejects_bad_solver_solution(monkeypatch, index):
    arm = CartesianArm()
    model = make_model(arm=arm)
    wrong = np.zeros(6)
    wrong[index] = 0.1
    monkeypatch.setattr(arm, "ik_flange", lambda *args: IKResult(True, wrong))
    result = model.ik(model.fk(np.zeros(7)), np.zeros(7), fixed_coordinates={"opening": 0})
    assert not result.converged and result.reason == "tcp_pose_tolerance"
    assert result.joints is None


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_bad_tolerance(value):
    with pytest.raises(ValueError):
        make_model(position_tolerance=value)


def test_duplicate_coordinate_names():
    tool = FixedToolGeometry(np.eye(4), coordinates=(KinematicCoordinate("j0", "normalized"),))
    with pytest.raises(ValueError, match="unique"):
        make_model(tool=tool)
