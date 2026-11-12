"""Shared candidate decoding and timestamp-aligned action references."""

from collections.abc import Sequence

import numpy as np

from manimux.policies.base import action_interval_ns
from manimux.policy_adapter.pose_output import POSE_SPACE, interpolate_pose


def decode_candidates(candidates, context, *, adapter, config, count):
    if not getattr(adapter, "supports_candidate_decoding", False):
        raise ValueError("candidate decoding requires complete absolute-joint source rows")
    if (
        not isinstance(candidates, Sequence) or isinstance(candidates, str | bytes)
        or len(candidates) != count
    ):
        raise ValueError("response must contain the requested number of candidates")
    chunks = [adapter.decode_action(candidate, context) for candidate in candidates]
    for chunk in chunks:
        if (
            chunk.action_space != "joint_position"
            or chunk.horizon_steps != config["policy"]["horizon_policy_steps"]
            or chunk.dt_ns != action_interval_ns(config["policy"])
            or set(chunk.groups) != set(config["robot"]["group_dims"])
            or chunk.source_offset_steps or chunk.hold_from_step
            or chunk.runtime_trajectory is not None or chunk.handoff is not None
        ):
            raise ValueError("candidate decoding requires complete untrimmed absolute-joint chunks")
    return chunks


def align_prediction(rows, origin_ns, times, dt_ns, *, first_step=0):
    """Interpolate valid overlap only; the leading axis of rows is policy time."""
    positions = (times - origin_ns) / dt_ns
    mask = (positions >= first_step) & (positions <= len(rows) - 1)
    positions = positions[mask]
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(rows) - 1)
    alpha = (positions - lower).reshape((-1,) + (1,) * (rows.ndim - 1))
    return mask, (1 - alpha) * rows[lower] + alpha * rows[upper]


def sample_reference(timeline, *, origin_ns, dt_ns, steps, group_order):
    """Sample executable targets on a request's grid, without extending old plans."""
    samples = [timeline.sample(origin_ns + step * dt_ns) for step in range(steps)]
    valid = np.array([sample is not None for sample in samples], dtype=bool)
    if not valid.any():
        return valid, None
    template = next(sample for sample in samples if sample is not None)
    rows = np.zeros((steps, sum(len(template[name]) for name in group_order)))
    for step in np.flatnonzero(valid):
        rows[step] = np.concatenate([samples[step][name] for name in group_order])
    return valid, rows


def sample_action_rows(groups, *, source_origin_ns, source_dt_ns, origin_ns, dt_ns,
                       steps, group_order, action_space):
    """Resample decoded rows on an observation clock, without extrapolating."""
    times = origin_ns + np.arange(steps, dtype=np.int64) * dt_ns
    width = sum(groups[name].shape[1] for name in group_order)
    output = np.zeros((steps, width), dtype=np.float64)
    parts = []
    for name in group_order:
        source = groups[name]
        valid, rows = align_prediction(source, source_origin_ns, times, source_dt_ns)
        if action_space == POSE_SPACE:
            positions = (times[valid] - source_origin_ns) / source_dt_ns
            for i, position in enumerate(positions):
                lower = int(np.floor(position))
                upper = min(lower + 1, len(source) - 1)
                rows[i] = interpolate_pose(source[lower], source[upper], position - lower)
        parts.append(rows)
    output[valid] = np.concatenate(parts, axis=1)
    return valid, output
