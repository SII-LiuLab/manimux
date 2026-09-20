from __future__ import annotations

import numpy as np

from manimux.clock import Clock
from manimux.types import RobotCommand, RobotState, copy_group_vector


class MockDualArmDriver:
    """Small deterministic plant used before hardware or simulator integration."""

    def __init__(
        self,
        group_dims: dict[str, int],
        clock: Clock,
        tracking_gain: float = 0.35,
    ) -> None:
        if not 0 < tracking_gain <= 1:
            raise ValueError("tracking_gain must be in (0, 1]")
        self._clock = clock
        self._tracking_gain = tracking_gain
        self._groups = {
            name: np.zeros(dim, dtype=np.float64) for name, dim in group_dims.items()
        }
        self._target = copy_group_vector(self._groups)
        self._connected = False
        self._stopped = False
        self._sequence = 0

    def connect(self) -> None:
        self._connected = True
        self._stopped = False

    def get_state(self) -> RobotState:
        if not self._connected:
            raise RuntimeError("mock robot is not connected")
        if not self._stopped:
            for name, target in self._target.items():
                self._groups[name] += self._tracking_gain * (target - self._groups[name])
        self._sequence += 1
        return RobotState(
            groups=copy_group_vector(self._groups),
            monotonic_ns=self._clock.now_ns(),
            sequence=self._sequence,
        )

    def send_command(self, command: RobotCommand) -> None:
        if not self._connected or self._stopped:
            raise RuntimeError("mock robot cannot accept commands")
        if set(command.groups) != set(self._groups):
            raise ValueError("command groups do not match mock robot groups")
        for name, values in command.groups.items():
            if values.shape != self._groups[name].shape:
                raise ValueError(f"command group {name!r} has the wrong shape")
        self._target = copy_group_vector(command.groups)

    def home(self) -> None:
        self._target = {name: np.zeros_like(value) for name, value in self._groups.items()}
        self._stopped = False

    def stop(self) -> None:
        self._target = copy_group_vector(self._groups)
        self._stopped = True

    def close(self) -> None:
        self._connected = False
