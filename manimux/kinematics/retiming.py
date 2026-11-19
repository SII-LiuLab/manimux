"""Shared TCP and gripper timing for policy trajectories."""
from dataclasses import dataclass

import numpy as np


def gripper_transition_weights(protected, steps):
    """Smoothstep from base timing (0) to ordinary retiming (1) outside protection."""
    protected = np.asarray(protected, dtype=bool)
    distance = np.full(len(protected), steps + 1, dtype=float)
    distance[protected] = 0
    # Shift the original mask, so overlapping ramps merge without growing it.
    for offset in range(1, min(steps + 1, len(protected))):
        distance[offset:] = np.minimum(
            distance[offset:], np.where(protected[:-offset], offset, steps + 1)
        )
        distance[:-offset] = np.minimum(
            distance[:-offset], np.where(protected[offset:], offset, steps + 1)
        )
    u = distance / (steps + 1)
    return u * u * (3 - 2 * u)


def retime_knots(distance, protected, dt_s, options, *, speed_protected=None,
                 transition_weights=None):
    """Set interval playback budgets, then equalize speed within that budget.

    Gripper motion and the hard RTC prefix retain the configured playback rate.
    Stationary intervals are excluded from equalization but may be slowed down.
    """
    distance = np.asarray(distance, dtype=float)
    protected = np.asarray(protected, dtype=bool)
    durations = np.full(len(distance), dt_s)
    speed_protected = (protected if speed_protected is None
                       else np.asarray(speed_protected, dtype=bool))
    durations[~speed_protected] /= options["outside_gripper_speed_ratio"]
    moving = ~protected
    if moving.any() and distance[moving].sum() > 1e-12:
        uniform = distance[moving] / distance[moving].mean()
        ratio = (1 - options["strength"]) + options["strength"] * uniform
        # A positive floor avoids deleting low-motion portions; preserve the budget.
        floor = options["min_interval_ratio"]
        excess = np.maximum(ratio - floor, 0)
        ratio = (
            floor + excess * (len(excess) * (1 - floor) / excess.sum())
            if excess.sum()
            else np.ones_like(excess)
        )
        durations[moving] *= ratio
    if transition_weights is not None:
        # Apply after equalization so it cannot undo the boundary taper. Do not
        # renormalize the horizon: transitions intentionally change its duration.
        durations = dt_s + transition_weights * (durations - dt_s)
    return np.r_[0.0, np.cumsum(durations)]


def rotation_distance(a, b):
    return np.arccos(np.clip((np.einsum("...ij,...ij->...", a, b) - 1) / 2, -1, 1))


@dataclass(frozen=True)
class RetimedPath:
    geometry: dict
    arm_lengths: dict
    gripper_events: np.ndarray
    gripper_protected: np.ndarray
    transition_weights: np.ndarray | None
    knots: np.ndarray
    prefix: int


def plan_path_timing(geo, apertures, dt, opt, prefix, *, knots=None):
    """Allocate a common clock from both TCP paths and gripper event streams."""
    h = len(next(iter(apertures.values())))
    increments = []
    protected = np.zeros(h - 1, dtype=bool)
    gripper_events = np.zeros(h - 1, dtype=bool)
    arm_lengths = {}
    for name, transforms in geo.items():
        xyz = transforms[:, :3, 3]
        translation = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        rotation = rotation_distance(transforms[:-1, :3, :3], transforms[1:, :3, :3])
        increments.append(np.hypot(translation, opt["rotation_radius_m"] * rotation))
        arm_lengths[name] = np.r_[0, np.cumsum(translation)]
        aperture_change = np.abs(np.diff(apertures[name]))
        gripper_events |= aperture_change >= opt["gripper_event_delta"]
        protected |= aperture_change >= opt["gripper_motion_delta"]
    protected |= gripper_events
    # Expand from the original motion mask only, without recursively padding.
    for index in np.flatnonzero(protected):
        protected[max(0, index - opt["gripper_protect_before_steps"]):index] = True
    gripper_protected = protected.copy()
    distance = np.max(increments, axis=0)
    protected |= distance < opt["stationary_distance_m"]
    # Keep the hard RTC prefix's original time allocation; it represents overlap.
    protected[:prefix] = True
    speed_protected = gripper_protected.copy()
    speed_protected[:prefix] = True
    transition_weights = (
        gripper_transition_weights(gripper_protected, opt["gripper_transition_steps"])
        if opt["gripper_transition_steps"] else None
    )
    if knots is None:
        knots = retime_knots(
            distance, protected, dt, opt, speed_protected=speed_protected,
            transition_weights=transition_weights,
        )
    return RetimedPath(
        geo, arm_lengths, gripper_events, gripper_protected, transition_weights, knots, prefix,
    )
