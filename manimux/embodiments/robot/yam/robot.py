"""Create YAM components from assembly YAML; construction, FK and RoboGUI do not open CAN."""

import logging
import threading
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
    runtime requests Home on faults; close requires verified Home or explicit physical support.
    """

    # Every runtime exit parks a healthy YAM; policy/config/UI cannot bypass it.
    home_before_close = True

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
        for label, duration in (("start", self._start_duration), ("home", self._home_duration),
                                ("home gripper release", self._release_duration)):
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not np.isfinite(duration) or duration <= 0:
                raise ValueError(f"{label} duration must be finite and positive")
        if self._move_to_start:
            # Reject bad startup targets before connect can enable any actuator.
            self._validated_targets(self._start_joints, "start position")

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
                parallel=True,
            )

    def _validated_targets(self, targets, label):
        if not isinstance(targets, Mapping) or set(targets) != set(self.arm_components):
            raise ValueError(f"{label} must specify every arm group")
        targets = {name: np.array(q, dtype=np.float64, copy=True) for name, q in targets.items()}
        for name, arm in self.arm_components.items():
            if targets[name].shape != (arm.num_joints,) or not np.isfinite(targets[name]).all():
                raise ValueError(f"{label}: invalid target for {name}")
        return targets

    def _move(self, targets, duration, label, *, interrupt=False, parallel=True):
        # Validate every target before submitting the first arm's motion.
        targets = self._validated_targets(targets, label)
        # Preserve linear SDK waypoints while cancelling peer motion on failure.
        for controller in self._controller_open:
            controller.check_control_health()
        cancel = threading.Event()
        with finish_move_before_interrupt(label, log) as interrupted:
            if parallel:
                with ThreadPoolExecutor(max_workers=len(self.arm_components)) as pool:
                    futures = [
                        pool.submit(
                            arm.controller.move_joints,
                            np.asarray(targets[name]),
                            time_interval_s=duration,
                            cancel=cancel,
                        )
                        for name, arm in self.arm_components.items()
                    ]
                    for future in futures:
                        future.result()
            else:
                for name, arm in self.arm_components.items():
                    arm.controller.move_joints(
                        np.asarray(targets[name]), time_interval_s=duration, cancel=cancel,
                    )
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
            # Check every arm before issuing the first recovery motion.
            for controller in self._controller_open:
                controller.check_control_health()
            measured = self.get_state().groups
            targets = {name: np.r_[np.zeros(6), q[6]] for name, q in measured.items()}
            self._move(targets, self._home_duration, "zero home")
            # Do not open a loaded gripper merely because interpolation returned.
            self._wait_for_home()
            for q in targets.values():
                q[6] = 1.0
            self._move(targets, self._release_duration, "home gripper release")
            self._wait_for_home()

    def _wait_for_home(self):
        # Check both arms concurrently so a failed peer is noticed promptly.
        cancel = threading.Event()

        def verify(controller):
            try:
                controller.wait_for_home(cancel=cancel)
            except BaseException:
                cancel.set()
                raise

        with ThreadPoolExecutor(max_workers=len(self._controller_open)) as pool:
            futures = [pool.submit(verify, c) for c in self._controller_open]
            errors = []
            for future in futures:
                try:
                    future.result()
                except BaseException as error:
                    errors.append(error)
            if errors:
                raise BaseExceptionGroup("YAM Home verification failed", errors)
        # Revalidate every arm after the slowest one settles, before release/open.
        for controller in self._controller_open:
            controller.validate_release()

    def home_targets(self, state):
        return {name: np.r_[np.zeros(6), 1.0] for name in state.groups}
