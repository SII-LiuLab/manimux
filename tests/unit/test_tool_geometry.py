import numpy as np
import pytest

from manimux.kinematics import FixedToolGeometry, KinematicCoordinate, ToolGeometryBase


def offset():
    return np.array([[0, -1, 0, 0.12], [1, 0, 0, -0.02], [0, 0, 1, 0.03], [0, 0, 0, 1.0]])


def test_fixed_transform_owns_its_data():
    source = offset()
    tool = FixedToolGeometry(source, base_frame="gripper_base", tcp_frame="grasp")
    source[:] = 0
    result = tool.tcp_transform()
    np.testing.assert_array_equal(result, offset())
    result[:] = 0
    np.testing.assert_array_equal(tool.tcp_transform(np.empty(0)), offset())
    assert isinstance(tool, ToolGeometryBase)
    assert tool.coordinates == ()
    assert tool.base_frame == "gripper_base"
    assert tool.tcp_frame == "grasp"


def test_fixed_tcp_retains_opening_coordinate():
    tool = FixedToolGeometry(offset(), coordinates=(KinematicCoordinate("opening", "normalized"),))
    assert tool.num_coordinates == 1
    for opening in (0.0, 0.5, 1.0):
        np.testing.assert_array_equal(tool.tcp_transform(np.array([opening])), offset())


@pytest.mark.parametrize("state", [None, [], [0, 1], [[0.5]], [np.nan], [np.inf], [-0.01], [1.01]])
def test_invalid_opening_state(state):
    tool = FixedToolGeometry(np.eye(4), coordinates=(KinematicCoordinate("opening", "normalized"),))
    with pytest.raises(ValueError):
        tool.tcp_transform(state)


def test_no_implicit_coordinate_for_rigid_tool():
    with pytest.raises(ValueError):
        FixedToolGeometry(np.eye(4)).tcp_transform(np.array([0.5]))


@pytest.mark.parametrize(
    "transform",
    [
        np.eye(3),
        np.full((4, 4), np.nan),
        np.diag([1, 1, 1, 2]),
        np.diag([-1, 1, 1, 1]),
        np.diag([2, 1, 1, 1]),
        np.array([[1, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    ],
)
def test_invalid_transform(transform):
    with pytest.raises(ValueError):
        FixedToolGeometry(transform)


def test_duplicate_coordinate_names_are_rejected():
    # 重名会破坏按名称固定工具坐标的 IK 约束。
    with pytest.raises(ValueError, match="unique"):
        FixedToolGeometry(
            np.eye(4),
            coordinates=(KinematicCoordinate("opening", "m"), KinematicCoordinate("opening", "m")),
        )


def test_multiple_tool_coordinates_keep_declared_units():
    tool = FixedToolGeometry(
        offset(),
        coordinates=(
            KinematicCoordinate("slide", "m"),
            KinematicCoordinate("rotation", "rad"),
        ),
    )
    np.testing.assert_array_equal(tool.tcp_transform(np.array([-0.02, 2.0])), offset())


def test_mount_is_applied_outside_tool_geometry():
    tool = FixedToolGeometry(offset())
    mount = np.array([[1, 0, 0, 0.05], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    composed = mount @ tool.tcp_transform()
    np.testing.assert_allclose(composed[:3, 3], [0.17, -0.03, -0.02])
    np.testing.assert_array_equal(tool.tcp_transform(), offset())
