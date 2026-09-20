from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.kinematics.base import ArmKinematics
from manimux.policies.xpolicylab.codec import GroupLayout, decode_action_steps


@dataclass(frozen=True, slots=True)
class AacPreviousAction:
    ee_features: np.ndarray
    chunk_size: int


@dataclass(frozen=True, slots=True)
class EeActionStats:
    groups: tuple[str, ...]
    minimum: np.ndarray
    maximum: np.ndarray

    def normalize(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 4 or values.shape[2:] != (len(self.groups), 7):
            raise ValueError(
                f"AAC EE features must have shape (N, H, {len(self.groups)}, 7), got {values.shape}"
            )
        normalized = values.copy()
        span = self.maximum - self.minimum
        live = span != 0
        pose = normalized[..., :6]
        safe_span = np.where(live, span, 1.0)
        pose[...] = 2.0 * (pose - self.minimum) / safe_span - 1.0
        pose[...] = np.where(live, pose, 0.0)
        return normalized


@dataclass(frozen=True, slots=True)
class AacSelection:
    chunk_size: int
    chunk_id: int
    entropy_elbow: int
    motion_floor: int
    step_entropy: np.ndarray
    chunk_mean_entropy: np.ndarray
    motion_magnitude: np.ndarray

    def metadata(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_id": self.chunk_id,
            "entropy_elbow": self.entropy_elbow,
            "motion_floor": self.motion_floor,
            "step_entropy": self.step_entropy.tolist(),
            "chunk_mean_entropy": self.chunk_mean_entropy.tolist(),
            "motion_magnitude": self.motion_magnitude.tolist(),
            "metric_space": "dual_arm_incremental_ee_action_mean",
        }


def load_ee_action_stats(
    path: str | Path,
    *,
    layouts: tuple[GroupLayout, ...],
) -> EeActionStats:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text())
    if payload.get("format") != "manimux.aac.ee_increment_min_max.v1":
        raise ValueError(f"unsupported AAC EE stats format in {source}")
    groups = tuple(layout.group for layout in layouts)
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, Mapping):
        raise ValueError(f"AAC EE stats must contain a groups mapping: {source}")
    minimum, maximum = [], []
    for group in groups:
        values = raw_groups.get(group)
        if not isinstance(values, Mapping):
            raise ValueError(f"AAC EE stats are missing group {group!r}: {source}")
        low = np.asarray(values.get("min"), dtype=np.float64)
        high = np.asarray(values.get("max"), dtype=np.float64)
        if low.shape != (6,) or high.shape != (6,):
            raise ValueError(f"AAC EE stats for {group!r} must have six min/max values")
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError(f"AAC EE stats for {group!r} must be finite")
        if np.any(high < low):
            raise ValueError(f"AAC EE stats max is below min for {group!r}")
        minimum.append(low)
        maximum.append(high)
    return EeActionStats(
        groups=groups,
        minimum=np.stack(minimum),
        maximum=np.stack(maximum),
    )


def _gaussian_entropy(samples: np.ndarray, *, epsilon: float = 1e-6) -> float:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"Gaussian samples must have shape (N, D), got {samples.shape}")
    count, dimension = samples.shape
    if count <= 1:
        return 0.0
    covariance = np.atleast_2d(np.cov(samples, rowvar=False))
    covariance = covariance + epsilon * np.eye(dimension, dtype=np.float64)
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0:
        return float("-inf")
    return float(0.5 * (dimension * np.log(2.0 * np.pi * np.e) + log_determinant))


def _bernoulli_entropy(samples: np.ndarray) -> float:
    probability = float(np.clip(np.mean(np.asarray(samples)), 1e-9, 1.0 - 1e-9))
    return float(
        -probability * np.log(probability) - (1.0 - probability) * np.log(1.0 - probability)
    )


def ee_pose_increment(previous_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    delta_position = target_pose[:3, 3] - previous_pose[:3, 3]
    delta_rotation = Rotation.from_matrix(target_pose[:3, :3] @ previous_pose[:3, :3].T).as_rotvec()
    return np.concatenate([delta_position, delta_rotation])


def build_ee_candidates(
    candidate_chunks: Sequence[object],
    *,
    layouts: tuple[GroupLayout, ...],
    current_groups: Mapping[str, np.ndarray],
    kinematics: ArmKinematics,
) -> tuple[np.ndarray, list[object]]:
    if len(layouts) != 2:
        raise ValueError("dual-arm AAC requires exactly two group layouts")
    if len(candidate_chunks) <= 1:
        raise ValueError("AAC requires more than one candidate chunk")

    decoded = [decode_action_steps(candidate, layouts=layouts) for candidate in candidate_chunks]
    horizons = {values.shape[0] for candidate in decoded for values in candidate.values()}
    if len(horizons) != 1:
        raise ValueError(f"AAC candidates have mismatched horizons: {sorted(horizons)}")
    horizon = horizons.pop()
    if horizon <= 1:
        raise ValueError("AAC candidates must contain at least two action steps")

    current_poses: list[np.ndarray] = []
    for layout in layouts:
        if layout.arm_dofs != kinematics.num_arm_joints or layout.gripper_dofs != 1:
            raise ValueError(
                "AAC FK requires each group to match the kinematics arm width "
                "and contain exactly one gripper value"
            )
        current = np.asarray(current_groups[layout.group], dtype=np.float64)
        if current.shape != (layout.dim,):
            raise ValueError(
                f"AAC current group {layout.group!r} must have shape "
                f"{(layout.dim,)}, got {current.shape}"
            )
        current_poses.append(kinematics.fk(current[: layout.arm_dofs], float(current[-1])))

    features = np.empty((len(decoded), horizon, 2, 7), dtype=np.float64)
    for candidate_id, groups in enumerate(decoded):
        for arm_id, layout in enumerate(layouts):
            trajectory = groups[layout.group]
            previous_pose = current_poses[arm_id]
            for step, action in enumerate(trajectory):
                pose = kinematics.fk(action[: layout.arm_dofs], float(action[-1]))
                features[candidate_id, step, arm_id, :6] = ee_pose_increment(previous_pose, pose)
                features[candidate_id, step, arm_id, 6] = action[-1]
                previous_pose = pose
    if not np.isfinite(features).all():
        raise ValueError("AAC FK produced non-finite EE features")
    return features, list(candidate_chunks)


def ee_entropy(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(features, dtype=np.float64)
    if candidates.ndim != 4 or candidates.shape[2:] != (2, 7):
        raise ValueError(f"AAC EE features must have shape (N, H, 2, 7), got {candidates.shape}")
    horizon = candidates.shape[1]
    step_entropy = np.empty(horizon, dtype=np.float64)
    for step in range(horizon):
        arm_entropy = []
        for arm in range(2):
            arm_entropy.append(
                _gaussian_entropy(candidates[:, step, arm, :3])
                + _gaussian_entropy(candidates[:, step, arm, 3:6])
                + _bernoulli_entropy(candidates[:, step, arm, 6] >= 0.5)
            )
        step_entropy[step] = float(np.mean(arm_entropy))
    chunk_mean = np.asarray(
        [np.mean(step_entropy[:size]) for size in range(1, horizon + 1)],
        dtype=np.float64,
    )
    return step_entropy, chunk_mean


def entropy_elbow(chunk_mean_entropy: np.ndarray) -> int:
    values = np.asarray(chunk_mean_entropy, dtype=np.float64)
    if values.ndim != 1 or values.size <= 1:
        raise ValueError("AAC entropy elbow requires at least two chunk means")
    return max(int(np.argmax(np.diff(values))) + 1, 2)


def ee_motion_magnitude(features: np.ndarray, *, candidate_id: int = 0) -> np.ndarray:
    candidate = np.asarray(features, dtype=np.float64)[candidate_id]
    magnitudes = []
    for chunk_size in range(2, candidate.shape[0] + 1):
        arm_magnitudes = []
        for arm in range(2):
            increments = candidate[:chunk_size, arm]
            position = np.sum(increments[:, :3], axis=0)
            rotation = Rotation.identity()
            for rotvec in increments[:, 3:6]:
                rotation = Rotation.from_rotvec(rotvec) * rotation
            gripper = candidate[:chunk_size, arm, 6] >= 0.5
            gripper_toggle = float(np.any(np.diff(gripper)))
            arm_magnitudes.append(
                np.linalg.norm(position)
                + np.linalg.norm(rotation.as_rotvec())
                + 0.2 * gripper_toggle
            )
        magnitudes.append(float(np.mean(arm_magnitudes)))
    return np.asarray(magnitudes, dtype=np.float64)


def motion_floor(magnitudes: np.ndarray, threshold: float, horizon: int) -> int:
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("AAC motion threshold must be finite and non-negative")
    for offset, magnitude in enumerate(np.asarray(magnitudes, dtype=np.float64)):
        if magnitude > threshold:
            return offset + 2
    return horizon


def _candidate_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    per_step_arm = np.linalg.norm(source - target, axis=-1)
    return per_step_arm.mean(axis=2)


def _whole_chunk_arm_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = source - target
    flattened = difference.transpose(0, 2, 1, 3).reshape(difference.shape[0], 2, -1)
    return np.linalg.norm(flattened, axis=-1).mean(axis=1)


def select_candidate(
    features: np.ndarray,
    *,
    method: str,
    chunk_size: int,
    previous: AacPreviousAction | None,
    beta: float,
) -> int:
    candidates = np.asarray(features, dtype=np.float64)
    if method == "0":
        return 0
    if method == "mean":
        mean = candidates.mean(axis=0, keepdims=True)
        distances = _whole_chunk_arm_distances(candidates, mean)
        return int(np.argmin(distances))
    if method != "backward":
        raise ValueError("AAC chunk_id_selector must be '0', 'mean', or 'backward'")
    if not 0 < beta <= 1:
        raise ValueError("AAC backward beta must be in (0, 1]")
    if previous is None or previous.chunk_size >= previous.ee_features.shape[1]:
        return 0

    overlap = previous.ee_features[:, previous.chunk_size : previous.chunk_size + chunk_size]
    if overlap.shape[1] == 0:
        return 0
    current = candidates[:, : overlap.shape[1]]
    if current.shape[0] != overlap.shape[0]:
        raise ValueError("AAC backward selector requires matching candidate counts")
    distances = _candidate_distances(current, overlap)
    weights = beta ** np.arange(overlap.shape[1], dtype=np.float64)
    weights /= weights.sum()
    return int(np.argmin(np.sum(distances * weights[None, :], axis=1)))


def select_ee_chunk(
    candidate_chunks: Sequence[object],
    *,
    layouts: tuple[GroupLayout, ...],
    current_groups: Mapping[str, np.ndarray],
    kinematics: ArmKinematics,
    ee_stats: EeActionStats,
    motion_threshold: float = 3.0,
    chunk_id_selector: str = "0",
    previous: AacPreviousAction | None = None,
    backward_beta: float = 0.99,
) -> tuple[object, AacSelection, AacPreviousAction]:
    features, native_candidates = build_ee_candidates(
        candidate_chunks,
        layouts=layouts,
        current_groups=current_groups,
        kinematics=kinematics,
    )
    normalized_features = ee_stats.normalize(features)
    step_entropy, chunk_mean = ee_entropy(normalized_features)
    elbow = entropy_elbow(chunk_mean)
    magnitudes = ee_motion_magnitude(features)
    floor = motion_floor(magnitudes, motion_threshold, features.shape[1])
    chunk_size = max(elbow, floor)
    chunk_id = select_candidate(
        normalized_features,
        method=chunk_id_selector,
        chunk_size=chunk_size,
        previous=previous,
        beta=backward_beta,
    )
    selected = native_candidates[chunk_id]
    if not isinstance(selected, Sequence) or isinstance(selected, str | bytes):
        raise ValueError("AAC native candidate must be an action-step sequence")
    selection = AacSelection(
        chunk_size=chunk_size,
        chunk_id=chunk_id,
        entropy_elbow=elbow,
        motion_floor=floor,
        step_entropy=step_entropy,
        chunk_mean_entropy=chunk_mean,
        motion_magnitude=magnitudes,
    )
    next_previous = AacPreviousAction(
        ee_features=normalized_features.copy(),
        chunk_size=chunk_size,
    )
    return list(selected[:chunk_size]), selection, next_previous
