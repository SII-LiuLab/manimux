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


def interpolate_pose(a, b, weight):
    """Interpolate unit wxyz poses, optionally with trailing tool coordinates.

    Accept single rows or batches (..., D); weight broadcasts over the batch.
    Position/tool coordinates are linear, orientation follows shortest-arc SLERP.
    """
    a, b = np.asarray(a), np.asarray(b)
    weight = np.asarray(weight)[..., None]
    result = (1 - weight) * a + weight * b
    qa, qb = a[..., 3:7], b[..., 3:7]
    dot = np.sum(qa * qb, axis=-1, keepdims=True)
    qb = np.where(dot < 0, -qb, qb)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1)) / np.pi
    # sinc handles coincident/antipodal orientations without a zero denominator.
    q = ((1 - weight) * np.sinc((1 - weight) * angle) * qa
         + weight * np.sinc(weight * angle) * qb) / np.sinc(angle)
    result[..., 3:7] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return result


def interpolate_pose_rows(rows, positions):
    """Sample a unit-quaternion pose trajectory at fractional row positions."""
    positions = np.clip(np.asarray(positions), 0, len(rows) - 1)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(rows) - 1)
    return interpolate_pose(rows[lower], rows[upper], positions - lower)
