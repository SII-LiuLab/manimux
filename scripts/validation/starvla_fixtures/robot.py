from __future__ import annotations

import numpy as np

from manimux.clock import Clock
from manimux.embodiments.robot import RobotBase
from manimux.types import RobotCommand, RobotState, copy_group_vector


class RobotDouble(RobotBase):
    """Deterministic plant for runtime tests; never registered by ManiMux."""

    def __init__(
        self,
        group_dims: dict[str, int],
        clock: Clock,
        tracking_gain: float = 0.35,
    ) -> None:
        if not 0 < tracking_gain <= 1:
            raise ValueError("tracking_gain must be in (0, 1]")
        # Joint-only runtime tests do not use FK/IK.
        self._kinematics = None
        self.model = None
        self._clock = clock
        self._tracking_gain = tracking_gain
        self._groups = {name: np.zeros(dim, dtype=np.float64) for name, dim in group_dims.items()}
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


def build_robot(config, clock):
    robot = RobotDouble(config["group_dims"], clock)
    options = config.get("options", {})
    if options.get("offline_model", False):
        from manimux.embodiments.robot.base import RobotModel

        if options.get("cartesian_test_geometry", False):
            raise ValueError("Select either offline robot geometry or Cartesian test geometry")
        robot.model = RobotModel.from_config(config["config"])
        robot._kinematics = robot.model.kinematics
        if {g: len(m.coordinates) for g, m in robot.kinematics.models.items()} != config[
            "group_dims"
        ]:
            raise ValueError("Offline geometry must match the test plant dimensions")
    elif options.get("cartesian_test_geometry", False):
        from manimux.kinematics import RobotKinematics
        from scripts.validation.starvla_fixtures.geometry import CartesianGeometry

        robot._kinematics = RobotKinematics(
            {group: CartesianGeometry(dim) for group, dim in config["group_dims"].items()}
        )
    initial = options.get("initial_groups")
    if initial is not None:
        if set(initial) != set(config["group_dims"]):
            raise ValueError("Initial test state must specify every group")
        for group, width in config["group_dims"].items():
            values = np.asarray(initial[group], dtype=float)
            if values.shape != (width,) or not np.isfinite(values).all():
                raise ValueError("Initial test state has invalid dimensions or values")
            robot._groups[group] = values.copy()
        robot._target = copy_group_vector(robot._groups)
    return robot
