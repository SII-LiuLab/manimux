"""Pure helpers for Physical Intelligence real-time chunking (RTC).

Ported unchanged from yam-abc-reproduce so the two deployments share one
definition of the paper's mask. Implements Eq. 5 of "Real-Time Execution of
Action Chunking Flow Policies" (Black et al., arXiv:2506.07339).


The policy predicts ``H`` actions.  When inference starts after ``s`` actions
from the current chunk have been consumed, ``H - s`` old actions remain.  RTC
uses them as an inpainting condition: the first ``d`` positions are frozen
because the conservative delay estimate says they will certainly execute,
then the guidance weight decays to zero across the rest of the overlap.

This module intentionally contains no torch or transport imports.  The robot
client builds the aligned target/mask here and a flow-policy server applies the
gradient guidance during denoising.
"""

from __future__ import annotations

import numpy as np

RTC_MODE = "pi_guided_v1"


def project_chunk_to_joint_speed(
    actions: np.ndarray,
    *,
    state: np.ndarray,
    joint_indices: np.ndarray | list[int] | tuple[int, ...],
    max_joint_step: float,
    start_step: int = 0,
) -> tuple[np.ndarray, int, float]:
    """Project an action suffix into the controller's per-step speed envelope.

    RTC assumes that ``Acur[t]`` is the action the controller actually consumes.
    A downstream state-dependent clamp breaks that assumption.  This helper makes
    the safety limit part of ``Acur`` itself: starting from the latest measured
    state, every arm-joint target is clipped relative to the preceding projected
    target.  Channels outside ``joint_indices`` are deliberately untouched.

    Returns ``(projected, changed_rows, max_abs_correction)``.  The input is never
    modified, which keeps the helper safe to use in diagnostics and tests.
    """
    chunk = np.asarray(actions, dtype=np.float32)
    current = np.asarray(state, dtype=np.float32).reshape(-1)
    indices = np.asarray(joint_indices, dtype=np.intp).reshape(-1)
    step = float(max_joint_step)
    start = int(start_step)
    if chunk.ndim != 2:
        raise ValueError(f"actions must have shape (H, D), got {chunk.shape}")
    if current.shape != (chunk.shape[1],):
        raise ValueError(f"state must have shape ({chunk.shape[1]},), got {current.shape}")
    if not 0 <= start <= len(chunk):
        raise ValueError(f"start_step must be in [0, {len(chunk)}], got {start}")
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"max_joint_step must be finite and positive, got {step}")
    if len(indices) and (indices.min() < 0 or indices.max() >= chunk.shape[1]):
        raise ValueError(f"joint_indices out of range for action dim {chunk.shape[1]}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("joint_indices must be unique")

    projected = np.array(chunk, copy=True)
    previous = current[indices].copy()
    changed_rows = 0
    max_correction = 0.0
    for row_idx in range(start, len(projected)):
        raw = projected[row_idx, indices].copy()
        safe = np.clip(raw, previous - step, previous + step)
        correction = float(np.max(np.abs(safe - raw))) if len(indices) else 0.0
        if correction > 1e-6:
            changed_rows += 1
            max_correction = max(max_correction, correction)
        projected[row_idx, indices] = safe
        previous = safe
    return projected, changed_rows, max_correction


def soft_mask(horizon: int, executed_steps: int, delay_steps: int) -> np.ndarray:
    """Return the paper's exponential soft mask (Eq. 5), shape ``(H,)``.

    ``delay_steps`` is a conservative forecast, normally the maximum of a
    short history of observed inference delays.  The RTC feasibility constraint
    is ``d <= s <= H - d``.
    """
    h, s, d = int(horizon), int(executed_steps), int(delay_steps)
    if h <= 0:
        raise ValueError(f"horizon must be positive, got {h}")
    if not 0 <= d <= s <= h - d:
        raise ValueError(
            f"RTC requires 0 <= delay <= executed <= horizon-delay; got d={d}, s={s}, H={h}"
        )

    overlap = h - s
    weights = np.zeros(h, dtype=np.float32)
    weights[:d] = 1.0
    denominator = overlap - d + 1
    for i in range(d, overlap):
        c_i = (overlap - i) / denominator
        weights[i] = c_i * np.expm1(c_i) / np.expm1(1.0)
    return weights


def inpainting_condition(
    current_chunk: np.ndarray,
    *,
    executed_steps: int,
    delay_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align the unexecuted old chunk with a newly generated ``H``-step chunk.

    Returns a right-padded action target ``(H, D)`` and the corresponding soft
    weights ``(H,)``.  Padding values are irrelevant because their weights are
    zero, but zeros keep the websocket payload deterministic.
    """
    actions = np.asarray(current_chunk, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"current_chunk must have shape (H, D), got {actions.shape}")
    h = len(actions)
    weights = soft_mask(h, executed_steps, delay_steps)
    targets = np.zeros_like(actions)
    remaining = actions[int(executed_steps) :]
    targets[: len(remaining)] = remaining
    return targets, weights
