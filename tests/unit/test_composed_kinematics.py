import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manimux.kinematics import (
    ArmKinematicsBase,
    ComposedManipulatorKinematics,
    FixedToolGeometry,
    IKResult,
    KinematicCoordinate,
    ToolGeometryBase,
    composed,
)


def pose(xyz, angles):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    result[:3, 3] = xyz
    return result


class CartesianArm(ArmKinematicsBase):
    num_joints = 6

    def fk(self, joints):
        return pose(joints[:3], joints[3:])

    def ik(self, target_flange, seed_joints, *, duration_s=None):
        del duration_s
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
    assert result.converged and result.target_reached
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
    monkeypatch.setattr(arm, "ik", lambda *args, **kwargs: IKResult(False, reason="joint_limit"))
    result = model.ik(np.eye(4), np.zeros(7), fixed_coordinates={"opening": 0})
    assert not result.converged and result.reason == "joint_limit" and result.joints is None

    def fail(*args, **kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(arm, "ik", fail)
    with pytest.raises(RuntimeError, match="backend unavailable"):
        model.ik(np.eye(4), np.zeros(7), fixed_coordinates={"opening": 0})


@pytest.mark.parametrize("index", [0, 5])
def test_tcp_residual_rejects_bad_solver_solution(monkeypatch, index):
    arm = CartesianArm()
    model = make_model(arm=arm)
    wrong = np.zeros(6)
    wrong[index] = 0.1
    monkeypatch.setattr(arm, "ik", lambda *args, **kwargs: IKResult(True, wrong))
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


def test_ik_validates_the_target_once_and_inverts_the_tool_offset_once(monkeypatch):
    """Per-call work in ik() is paid once per differential substep.

    A 30 Hz action knot is nine substeps at ik_validation_dt_s=0.004, so
    validating the same target twice or re-inverting an unchanged tool offset
    costs about 290 times per two-arm chunk. There is no IK over the tool: it
    only moves the target from the TCP frame to the flange.
    """
    model = make_model()
    seed = np.r_[np.zeros(6), 0.4]
    target = model.fk(seed)
    validated, inverted = [], []
    real_validate, real_inverse = composed.rigid_transform, np.linalg.inv

    def counting_validate(value, name):
        validated.append(name)
        return real_validate(value, name)

    def counting_inverse(matrix):
        inverted.append(matrix)
        return real_inverse(matrix)

    monkeypatch.setattr(composed, "rigid_transform", counting_validate)
    monkeypatch.setattr(np.linalg, "inv", counting_inverse)
    for _ in range(9):
        assert model.ik(target, seed, fixed_coordinates={"opening": 0.4}).converged
    assert validated == ["target_tcp"] * 9
    assert len(inverted) == 1

    moved = np.r_[np.zeros(6), 0.7]
    assert model.ik(model.fk(moved), moved, fixed_coordinates={"opening": 0.7}).converged
    # A tool whose TCP moves with its state still gets its own offset.
    assert len(inverted) == 2
