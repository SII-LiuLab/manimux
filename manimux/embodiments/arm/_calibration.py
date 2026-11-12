"""Station calibration for absolute joint targets and normalized gripper travel."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from manimux.types import FloatArray


class JointGripperCalibration:
    """Validate device commands and convert travel using measured endpoints.

    The hardware gripper units belong to each controller. Reversed endpoints
    are supported; out-of-range commands and measurements fail without clipping.
    Nominal URDF finger travel is independent of this station calibration.
    """

    def __init__(
        self,
        *,
        channel: str,
        joint_limits: ArrayLike,
        gripper_closed: float,
        gripper_open: float,
    ) -> None:
        if not isinstance(channel, str) or not channel.strip() or channel.startswith("REPLACE"):
            raise ValueError("channel must name the station's actual CAN interface")
        bounds = np.asarray(joint_limits, dtype=float)
        if (
            bounds.shape != (6, 2)
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError("joint_limits must contain six finite lower/upper pairs")
        endpoints = np.asarray([gripper_closed, gripper_open], dtype=float)
        if not np.isfinite(endpoints).all() or endpoints[0] == endpoints[1]:
            raise ValueError("distinct finite closed/open gripper calibration is required")
        self.channel = channel
        self.bounds = bounds.copy()
        self.closed = float(gripper_closed)
        self.open = float(gripper_open)

    def validate(self, targets: Mapping[str, ArrayLike]) -> FloatArray:
        if set(targets) != {self.channel}:
            raise ValueError("command must select exactly this controller's CAN channel")
        q = np.asarray(targets[self.channel], dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("command must contain six finite radians and one opening")
        if np.any(q[:6] < self.bounds[:, 0]) or np.any(q[:6] > self.bounds[:, 1]):
            raise ValueError("joint target is outside configured device bounds")
        if not 0.0 <= q[6] <= 1.0:
            raise ValueError("gripper target must be in [0, 1]")
        return q.copy()

    def opening(self, raw: float) -> float:
        value = (raw - self.closed) / (self.open - self.closed)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("measured gripper is outside its configured calibration")
        return float(value)

    def raw_opening(self, opening: float) -> float:
        return self.closed + opening * (self.open - self.closed)
