"""Time-based causal sampling and offline joint reconstruction shared by replays."""

from __future__ import annotations

import math

import numpy as np


def sampling_strides(target_hz, low_rates_hz):
    rates = (target_hz, *low_rates_hz)
    if any(isinstance(r, bool) or not math.isfinite(r) or r <= 0 for r in rates):
        raise ValueError("Replay rates must be finite and positive")
    if not low_rates_hz or len(set(low_rates_hz)) != len(low_rates_hz):
        raise ValueError("Specify distinct lower replay rates")
    strides = []
    for rate in low_rates_hz:
        ratio = target_hz / rate
        if rate >= target_hz or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("Each lower rate must divide the target rate by an integer > 1")
        strides.append(round(ratio))
    return strides


def resample_joint_streams(raw, *, target_hz, low_rates_hz, max_gap_s):
    """Put timestamped Nx7 streams on one grid; only the six joints interpolate.

    Times are integer Unix ns. Callers validate, deduplicate and order streams.
    Low-rate knots select latest actual samples by time, not every kth input row.
    All curves share end knots and retain missing source intervals as NaN.
    """
    strides = sampling_strides(target_hz, low_rates_hz)
    shared_stride = math.lcm(*strides)
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("Maximum source gap must be finite and positive")
    origin = min(int(ts[0]) for ts, _ in raw.values())
    start = max(int(ts[0]) for ts, _ in raw.values()) - origin
    stop = min(int(ts[-1]) for ts, _ in raw.values()) - origin
    first = math.ceil(start * target_hz / 1e9 / shared_stride - 1e-9) * shared_stride
    last = math.floor(stop * target_hz / 1e9 / shared_stride + 1e-9) * shared_stride
    if last <= first:
        raise ValueError("Recording is too short for shared sampling endpoints")
    ticks = np.arange(first, last + 1, dtype=np.int64)
    time_s = ticks / target_hz
    query_ns = origin + np.rint(time_s * 1e9).astype(np.int64)
    reference = {}
    variants = {f"resample_{i}": {} for i in range(len(strides))}
    for arm, (ts, q) in raw.items():
        index = np.searchsorted(ts, query_ns, side="right") - 1
        left = np.clip(index, 0, len(ts) - 1)
        right = np.minimum(left + 1, len(ts) - 1)
        baseline = q[left].copy()
        gap = (ts[right] - ts[left] > round(max_gap_s * 1e9)) & (query_ns != ts[left])
        baseline[(index < 0) | (query_ns > ts[-1]) | gap] = np.nan
        reference[arm] = baseline
        valid = np.isfinite(baseline[:, :6]).all(axis=1)
        bad_prefix = np.r_[0, np.cumsum(~valid)]
        for method, stride in zip(variants, strides, strict=True):
            knots = np.arange(0, len(ticks), stride)
            output = baseline.copy()  # grippers share the reference, including gaps
            output[:, :6] = np.column_stack([
                np.interp(ticks, ticks[knots], baseline[knots, joint]) for joint in range(6)
            ])
            lo = np.minimum(np.arange(len(ticks)) // stride, len(knots) - 1)
            hi = np.minimum(lo + 1, len(knots) - 1)
            # Never fill an interior missing source region using good low-rate endpoints.
            crosses_gap = bad_prefix[knots[hi] + 1] - bad_prefix[knots[lo]] > 0
            crosses_gap[np.arange(len(ticks)) % stride == 0] = False
            output[~valid | crosses_gap, :6] = np.nan
            variants[method][arm] = output
    valid_endpoints = ticks % shared_stride == 0
    for groups in (reference, *variants.values()):
        for values in groups.values():
            valid_endpoints &= np.isfinite(values).all(axis=1)
    good = np.flatnonzero(valid_endpoints)
    if len(good) < 2:
        raise ValueError("No shared valid sample endpoints for joint comparison")
    keep = slice(good[0], good[-1] + 1)
    return (
        origin, time_s[keep],
        {arm: q[keep] for arm, q in reference.items()},
        {method: {arm: q[keep] for arm, q in groups.items()}
         for method, groups in variants.items()},
    )
