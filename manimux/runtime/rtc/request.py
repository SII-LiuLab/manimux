"""The inference request an RTC runtime sends.

``InferenceRequest`` is part of the core ManiMux contract and is left untouched.
A subclass carries the inpainting condition instead: the policy worker's
``isinstance`` check still passes, the queue still pickles it, and a policy that
does not know about RTC simply ignores the extra fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from manimux.types import InferenceRequest


@dataclass(slots=True)
class RtcInferenceRequest(InferenceRequest):
    """An inference request carrying a real-time-chunking condition.

    ``action_condition`` samples the previous trajectory on the new observation's
    action grid, so index 0 denotes the same time as index 0 of the chunk being
    generated. Unsupported tail rows have zero weight. Values are in raw robot
    units; the policy server normalizes them into the model's action space.

    ``condition_weights`` is the paper's per-step soft mask.
    """

    action_condition: NDArray[np.float64] | None = None
    condition_weights: NDArray[np.float64] | None = None
    rtc_beta: float = 5.0
    # Runtime-only origin of the condition weights, before an adapter's phase shift.
    condition_time_ns: int | None = None
