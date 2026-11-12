"""Create YAM components from assembly YAML; construction, FK and RoboGUI do not open CAN."""

import logging
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.arm.yam import YamController
from manimux.embodiments.robot._interrupt import finish_move_before_interrupt
from manimux.embodiments.robot.base import RobotBase, RobotModel

log = logging.getLogger(__name__)


class YamRobot(RobotBase):
    """Each group owns an integrated arm/gripper i2rt session; configuration supplies group names and mounts.

    execute controls all seven coordinates; the separate-tool switch does not control the built-in gripper.
    connect holds the current pose unless move_to_start_on_connect explicitly requests motion.
    home retains the original two phases: home both arms while holding gripper openings, then open the grippers.
    close releases connections; runtime still decides whether to Home before exit.
    """

    def __init__(
        self,
        model,
        *,
        clock=None,
        hardware=None,
        component_hardware=None,
        execute=False,
        move_to_start_on_connect=False,
        start_joints=None,
        start_duration_s=5.0,
        home_duration_s=5.0,
        home_gripper_release_duration_s=1.0,
        home_on_close=False,
    ):
        clock = clock if clock is not None else SystemClock()
        control = {**model.hardware, **(hardware or {})}
        bound = {name: dict(component["hardware"]) for name, component in model.components.items()}
        # A shared station can bind devices outside this robot's assembly.
        for name, options in bound.items():
            options.update((component_hardware or {}).get(name, {}))
        arms = {}
        for name, group in model.groups.items():
            component = model.components[group.arm_name]
            options = dict(bound[group.arm_name])
            controller = YamController(channel=options.pop("channel", None), clock=clock, **options)
            arms[name] = component["class"](controller, kinematics=group.kinematics)
        sensors = {
            name: component["class"](
                name=name, clock=clock, **{**component["options"], **bound[name]}
            )
            for name, component in model.components.items()
            if component["type"] == "sensor"
        }
        super().__init__(
            arm_components=arms,
            models={name: group.kinematics for name, group in model.groups.items()},
            model=model,
            sensors=sensors,
            clock=clock,
            stale_timeout_s=control.get("stale_timeout_s", 0.2),
            execute=execute,
        )
        self._move_to_start = move_to_start_on_connect
        self._start_joints = start_joints
        self._start_duration = start_duration_s
        self._home_duration = home_duration_s
        self._release_duration = home_gripper_release_duration_s

    @classmethod
    def from_config(cls, path, **options):
        return cls(RobotModel.from_config(path), **options)

    def connect(self):
        if self._ready:
            return
        super().connect()
        if self._execute and self._move_to_start:
            self._move(
                self._start_joints,
                self._start_duration,
                "start position",
                interrupt=True,
                parallel=False,
            )

    def _move(self, targets, duration, label, *, interrupt=False, parallel=True):
        # Validate every target before submitting the first arm's motion.
        if not isinstance(targets, Mapping) or set(targets) != set(self.arm_components):
            raise ValueError(f"{label} must specify every arm group")
        targets = {name: np.array(q, dtype=np.float64, copy=True) for name, q in targets.items()}
        for name, arm in self.arm_components.items():
            if targets[name].shape != (arm.num_joints,) or not np.isfinite(targets[name]).all():
                raise ValueError(f"{label}: invalid target for {name}")
        # Keep SDK move_joints and preserve the trajectory algorithm and motion parameters during migration.
        # Wait for all arms before the next phase; refresh feedback to synchronize the existing wrapper's state cache.
        with finish_move_before_interrupt(label, log) as interrupted:
            if parallel:
                with ThreadPoolExecutor(max_workers=len(self.arm_components)) as pool:
                    futures = [
                        pool.submit(
                            arm.controller.move_joints,
                            np.asarray(targets[name]),
                            time_interval_s=duration,
                        )
                        for name, arm in self.arm_components.items()
                    ]
                    for future in futures:
                        future.result()
            else:
                for name, arm in self.arm_components.items():
                    arm.controller.move_joints(np.asarray(targets[name]), time_interval_s=duration)
        self.get_state()
        if interrupted and interrupt:
            raise KeyboardInterrupt

    def stop(self):
        # YAM stop sends a hold target, so observation-only runs must skip it too.
        if self._execute:
            super().stop()

    def home(self):
        if not self._execute:
            return
        with self._lock:
            measured = self.get_state().groups
            targets = {name: np.r_[np.zeros(6), q[6]] for name, q in measured.items()}
            self._move(targets, self._home_duration, "zero home")
            for q in targets.values():
                q[6] = 1.0
            self._move(targets, self._release_duration, "home gripper release")
