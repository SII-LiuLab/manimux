"""Canonical absolute TCP trajectories at the policy/embodiment boundary."""

import uuid

import numpy as np

from manimux.kinematics.poses import matrix_pose
from manimux.types import ActionChunk

POSE_SPACE = "absolute_tcp_pose"


def validate_pose_rows(rows):
    rows = np.array(rows, dtype=float, copy=True)
    if rows.ndim != 2 or not rows.shape[0] or rows.shape[1] != 8 or not np.isfinite(rows).all():
        raise ValueError("TCP trajectory must be finite (H, 8): xyz, wxyz, gripper")
    norm = np.linalg.norm(rows[:, 3:7], axis=1)
    if np.any(norm < 1e-12) or np.any((rows[:, 7] < 0) | (rows[:, 7] > 1)):
        raise ValueError("TCP trajectory has invalid quaternion or normalized gripper")
    rows[:, 3:7] /= norm[:, None]
    return rows


def pose_chunk(context, groups, dt_ns, *, observation_time_ns=None, metadata=None):
    return ActionChunk(
        plan_id=f"pose-{context.request_seq}-{uuid.uuid4().hex[:8]}",
        request_seq=context.request_seq,
        observation_time_ns=(
            context.observation_time_ns if observation_time_ns is None else observation_time_ns
        ),
        created_time_ns=context.created_time_ns,
        action_space=POSE_SPACE,
        dt_ns=dt_ns,
        groups={name: validate_pose_rows(rows) for name, rows in groups.items()},
        metadata={
            "pose_frame": "per_group_robot_base",
            "control_point": "configured_tcp",
            **(metadata or {}),
        },
    )


def target_rows(targets):
    return {
        name: np.stack([np.r_[matrix_pose(pose), grip] for pose, grip in rows])
        for name, rows in targets.items()
    }


def interpolate_pose(a, b, weight):
    """Linear xyz/gripper, shortest-arc unit quaternion interpolation (wxyz)."""
    result = (1 - weight) * a + weight * b
    qa, qb = a[3:7], b[3:7]
    dot = float(np.dot(qa, qb))
    if dot < 0:
        qb, dot = -qb, -dot
    if dot > 0.9995:
        q = (1 - weight) * qa + weight * qb
    else:
        angle = np.arccos(np.clip(dot, -1, 1))
        q = (np.sin((1 - weight) * angle) * qa + np.sin(weight * angle) * qb) / np.sin(angle)
    result[3:7] = q / np.linalg.norm(q)
    return result
