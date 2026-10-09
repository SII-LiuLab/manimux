"""Stateless model-aperture mappings applied before IK and timeline commit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GripperMapping:
    """Resolved stateless part of one user-selected gripper mode.

    ``continuous`` is identity. ``curve`` closes a configurable input band and
    applies a power curve to the remainder. Stateful ``close_latch`` execution
    resolves to identity here and remains owned by the runtime executor.
    """

    mode: str = "continuous"
    deadzone: float | None = None
    exponent: float | None = None

    @classmethod
    def from_options(cls, options: object | None) -> GripperMapping:
        values = {} if options is None else dict(options)
        mode = values.pop("mode", "continuous")
        if mode == "continuous":
            if values:
                raise ValueError(
                    f"continuous gripper mapping does not accept {sorted(values)}"
                )
            return cls()
        if mode != "curve":
            raise ValueError("gripper mapping mode must be 'continuous' or 'curve'")
        if set(values) != {"deadzone", "exponent"}:
            raise ValueError("curve gripper mapping requires only deadzone and exponent")
        deadzone = float(values["deadzone"])
        exponent = float(values["exponent"])
        if not math.isfinite(deadzone) or not 0.0 <= deadzone < 1.0:
            raise ValueError("curve gripper deadzone must be finite and in [0, 1)")
        if not math.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("curve gripper exponent must be finite and positive")
        return cls(mode="curve", deadzone=deadzone, exponent=exponent)

    def map(self, opening):
        values = np.asarray(opening, dtype=np.float64)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("gripper opening must be finite and in [0, 1]")
        if self.mode == "continuous":
            return values.copy()
        assert self.deadzone is not None and self.exponent is not None
        scaled = np.maximum(0.0, (values - self.deadzone) / (1.0 - self.deadzone))
        return scaled**self.exponent

    def inverse(self, opening):
        values = np.asarray(opening, dtype=np.float64)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("gripper opening must be finite and in [0, 1]")
        if self.mode == "continuous":
            return values.copy()
        assert self.deadzone is not None and self.exponent is not None
        return np.where(
            values == 0.0,
            0.0,
            self.deadzone + (1.0 - self.deadzone) * values ** (1.0 / self.exponent),
        )

    def metadata(self) -> dict[str, float | str]:
        values: dict[str, float | str] = {"mode": self.mode}
        if self.mode == "curve":
            assert self.deadzone is not None and self.exponent is not None
            values.update(deadzone=self.deadzone, exponent=self.exponent)
        return values
