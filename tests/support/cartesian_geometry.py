"""Analytic test geometry for transport/runtime checks, not a robot model."""

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.kinematics import IKResult, KinematicCoordinate


class CartesianGeometry:
    """xyz, rotation vector, optional redundant coordinates, normalized opening.

    This deliberately idealized plant tests target conversion and scheduling.
    It does not reproduce Franka/YAM dynamics, reachability or task success.
    """

    def __init__(self, dimension=7):
        if dimension < 7:
            raise ValueError("Cartesian test geometry needs at least 6+1 coordinates")
        self.coordinates = tuple(
            KinematicCoordinate(str(i), "m" if i < 3 else "rad") for i in range(dimension - 1)
        ) + (KinematicCoordinate("gripper", "normalized"),)
        self.fail = False

    def fk(self, q):
        target = np.eye(4)
        target[:3, 3] = q[:3]
        target[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return target

    def ik(self, target, seed, *, fixed_coordinates):
        if self.fail:
            return IKResult(False, reason="outside_test_workspace")
        joints = np.array(seed, dtype=float, copy=True)
        joints[:3] = target[:3, 3]
        joints[3:6] = Rotation.from_matrix(target[:3, :3]).as_rotvec()
        joints[-1] = fixed_coordinates["gripper"]
        return IKResult(True, joints)


class CartesianArmKinematics:
    """Legacy arm FK interface used only by the offline AAC metric fixture."""

    num_arm_joints = 6

    def fk(self, joints, gripper):
        return CartesianGeometry().fk(np.r_[joints, gripper])
