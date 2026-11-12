"""Forecast how long the next process decode will take.

``expected_decode_s`` has to be known *before* the decode starts: it picks the
IK seed and the first source row, and those in turn decide how many knots the
decode must solve. The value can therefore never be measured in time for its
own use, only forecast from earlier decodes -- the same trick RTC uses for its
inference delay in ``RtcInferenceStrategy._delay_forecast``.
"""

from __future__ import annotations

from collections import deque
from statistics import fmean

FORECAST_MODES = ("max", "mean")


class DecodeForecast:
    """Rolling estimate of the process-decode duration, in seconds.

    ``floor_s`` is the estimate until a decode has been measured and a lower
    bound after that: ``_decode_seed`` chooses its seed on
    ``expected_decode_s > 0``, so a forecast dipping to zero would flip the seed
    source between chunks. A ``size`` of zero holds every measurement out of the
    window, which keeps the configured value in use.
    """

    def __init__(self, *, floor_s: float, size: int, mode: str) -> None:
        self._floor_s = floor_s
        self._mode = mode
        self._samples: deque[float] = deque(maxlen=size)

    @property
    def seconds(self) -> float:
        if not self._samples:
            return self._floor_s
        estimate = max(self._samples) if self._mode == "max" else fmean(self._samples)
        return max(self._floor_s, estimate)

    def observe(self, decode_ms: float) -> None:
        self._samples.append(decode_ms / 1000.0)
