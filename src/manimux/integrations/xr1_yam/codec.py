"""NumPy-only YAM action layout and rotation conversion for XR-1 adapters."""

from __future__ import annotations

import numpy as np

ACTION_DIM = 60
STATE_DIM = 60
ACTION_PARTS = (
    ("left_ee_pos", slice(0, 3)),
    ("left_ee_aa", slice(3, 6)),
    ("left_gripper", slice(6, 7)),
    ("right_ee_pos", slice(8, 11)),
    ("right_ee_aa", slice(11, 14)),
    ("right_gripper", slice(14, 15)),
    ("waist", slice(16, 17)),
    ("base", slice(17, 20)),
)


def _axis_from_pi(rotation: np.ndarray) -> np.ndarray:
    rot00, rot11, rot22 = rotation[0, 0], rotation[1, 1], rotation[2, 2]
    if rot00 >= rot11 and rot00 >= rot22:
        axis_x = np.sqrt(max((rot00 + 1) / 2, 0))
        if axis_x > 1e-8:
            axis_y = rotation[0, 1] / (2 * axis_x)
            axis_z = rotation[0, 2] / (2 * axis_x)
        else:
            axis_y = axis_z = 0.0
        axis = np.array([axis_x, axis_y, axis_z])
    elif rot11 >= rot22:
        axis_y = np.sqrt(max((rot11 + 1) / 2, 0))
        if axis_y > 1e-8:
            axis_x = rotation[0, 1] / (2 * axis_y)
            axis_z = rotation[1, 2] / (2 * axis_y)
        else:
            axis_x = axis_z = 0.0
        axis = np.array([axis_x, axis_y, axis_z])
    else:
        axis_z = np.sqrt(max((rot22 + 1) / 2, 0))
        if axis_z > 1e-8:
            axis_x = rotation[0, 2] / (2 * axis_z)
            axis_y = rotation[1, 2] / (2 * axis_z)
        else:
            axis_x = axis_y = 0.0
        axis = np.array([axis_x, axis_y, axis_z])
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return axis / norm


def rotm2aa_batch(rotations: np.ndarray) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.float32)
    theta = np.arccos(np.clip((np.einsum("nii->n", rotations) - 1.0) / 2.0, -1.0, 1.0))
    axis_angle = np.zeros((rotations.shape[0], 3))
    near_zero = theta <= 1e-6
    near_pi = np.abs(theta - np.pi) <= 1e-6
    normal = ~(near_zero | near_pi)
    if np.any(normal):
        axis = np.stack(
            [
                rotations[:, 2, 1] - rotations[:, 1, 2],
                rotations[:, 0, 2] - rotations[:, 2, 0],
                rotations[:, 1, 0] - rotations[:, 0, 1],
            ],
            axis=1,
        )
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
        axis_angle[normal] = axis[normal] * theta[normal, None]
    if np.any(near_pi):
        for index in np.where(near_pi)[0]:
            axis_angle[index] = _axis_from_pi(rotations[index]) * theta[index]
    return axis_angle
