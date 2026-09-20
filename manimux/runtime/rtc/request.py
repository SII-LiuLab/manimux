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

    ``action_condition`` is the unexecuted tail of the chunk currently being
    executed, left-shifted so index 0 denotes the same controller step as index 0
    of the chunk being generated, and right-padded to the horizon. It is in raw
    robot units; the policy server normalizes it into the model's action space.

    ``condition_weights`` is the paper's per-step soft mask.
    """

    action_condition: NDArray[np.float64] | None = None
    condition_weights: NDArray[np.float64] | None = None
    rtc_beta: float = 5.0
