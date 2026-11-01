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
