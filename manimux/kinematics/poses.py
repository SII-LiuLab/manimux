"""Pose representation conversion; xyz metres and quaternion wxyz."""

import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(value):
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("EE pose must be finite [xyz, quaternion wxyz]")
    if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-4):
        raise ValueError("EE quaternion must be unit length")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def matrix_pose(matrix):
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.r_[matrix[:3, 3], quat[[3, 0, 1, 2]]]
