# Copyright (C) 2026 Xiaomi Corporation.
# Extracted unchanged from mibot/utils/io.py; Apache-2.0, see licenses/
# xiaomi-robotics-1-APACHE-2.0.txt. Only NumPy is needed for action conversion.
import numpy as np


def _axis_from_pi(rotm: np.ndarray) -> np.ndarray:
    rot00, rot11, rot22 = rotm[0, 0], rotm[1, 1], rotm[2, 2]

    if rot00 >= rot11 and rot00 >= rot22:
        vx = np.sqrt(max((rot00 + 1) / 2, 0))
        if vx > 1e-8:
            vy = rotm[0, 1] / (2 * vx)
            vz = rotm[0, 2] / (2 * vx)
        else:
            vy = vz = 0.0
        axis = np.array([vx, vy, vz])
    elif rot11 >= rot22:
        vy = np.sqrt(max((rot11 + 1) / 2, 0))
        if vy > 1e-8:
            vx = rotm[0, 1] / (2 * vy)
            vz = rotm[1, 2] / (2 * vy)
        else:
            vx = vz = 0.0
        axis = np.array([vx, vy, vz])
    else:
        vz = np.sqrt(max((rot22 + 1) / 2, 0))
        if vz > 1e-8:
            vx = rotm[0, 2] / (2 * vz)
            vy = rotm[1, 2] / (2 * vz)
        else:
            vx = vy = 0.0
        axis = np.array([vx, vy, vz])

    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return axis / norm


def rotm2aa_batch(rotms: np.ndarray) -> np.ndarray:
    rotms = np.asarray(rotms, dtype=np.float32)
    theta = np.arccos(np.clip((np.einsum("nii->n", rotms) - 1.0) / 2.0, -1.0, 1.0))

    axis_angle = np.zeros((rotms.shape[0], 3))
    near_zero = theta <= 1e-6
    near_pi = np.abs(theta - np.pi) <= 1e-6
    normal = ~(near_zero | near_pi)

    if np.any(normal):
        axis = np.stack(
            [
                rotms[:, 2, 1] - rotms[:, 1, 2],
                rotms[:, 0, 2] - rotms[:, 2, 0],
                rotms[:, 1, 0] - rotms[:, 0, 1],
            ],
            axis=1,
        )
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
        axis_angle[normal] = axis[normal] * theta[normal, None]

    if np.any(near_pi):
        for i in np.where(near_pi)[0]:
            axis_angle[i] = _axis_from_pi(rotms[i]) * theta[i]

    return axis_angle
