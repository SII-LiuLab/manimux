"""Two independently owned Flexiv Rizon4s controllers in the common runtime."""

import os
import threading
import time
from functools import partial
from types import MappingProxyType

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.arm.flexiv import FlexivController
from manimux.embodiments.end_effector.gripper import GripperCommand
from manimux.embodiments.robot.base import RobotBase, RobotModel


class FlexivRobot(RobotBase):
    def __init__(self, model, *, clock=None, hardware=None, component_hardware=None, execute=False,
                 end_effector_control=True):
        clock = clock if clock is not None else SystemClock()
        bindings = component_hardware or {}
        arms = {}
        tools = {}
        serials = set()
        tool_ports = set()
        if any(c["type"] not in {"arm", "end_effector"} for c in model.components.values()):
            raise ValueError(
                "Flexiv assembly supports arms and grippers; use external camera clients"
            )
        for name, group in model.groups.items():
            component = model.components[group.arm_name]
            controller = FlexivController(
                clock=clock,
                execute=execute,
                **{**component["hardware"], **bindings.get(group.arm_name, {})},
            )
            if controller.channel in serials:
                raise ValueError("Flexiv arms must have distinct robot serials")
            serials.add(controller.channel)
            arms[name] = component["class"](controller, kinematics=group.arm.kinematics)
            if group.end_effector_name is not None:
                tool = model.components[group.end_effector_name]
                tool_hardware = {**tool["hardware"], **bindings.get(group.end_effector_name, {})}
                port = tool_hardware.get("port")
                if not isinstance(port, str) or not port:
                    raise ValueError("each gripper requires a station USB2CAN port")
                port = os.path.realpath(port)
                if port in tool_ports:
                    raise ValueError("each gripper requires a distinct USB2CAN port")
                tool_ports.add(port)
                tools[name] = tool["class"](
                    clock=clock,
                    execute=execute,
                    **tool_hardware,
                )
        settings = {**model.hardware, **(hardware or {})}
        self._home_settings = {"control_hz": 50.0, "peak_velocity_deg_s": 6.0,
                               "tolerance_deg": 0.5, "settle_timeout_s": 5.0,
                               **settings.get("home_motion", {})}
        values = list(self._home_settings.values())
        if not np.isfinite(values).all() or min(values) <= 0 or self._home_settings["control_hz"] > 100:
            raise ValueError("invalid Flexiv home_motion settings")
        self._drag_scale = float(settings.get("drag_stiffness_scale", 0.01))
        self._guiding_motion = {
            "control_hz": 100.0, "damping_ratio": 0.7,
            "max_velocity_deg_s": 6.0, "max_acceleration_deg_s2": 18.0,
            **settings.get("hand_guiding_motion", {}),
        }
        motion = self._guiding_motion
        if (not np.isfinite([self._drag_scale, *motion.values()]).all()
                or not 0 <= self._drag_scale <= 1
                or not 0 < motion["control_hz"] <= 100
                or not 0.3 <= motion["damping_ratio"] <= 0.8
                or min(motion["max_velocity_deg_s"], motion["max_acceleration_deg_s2"]) <= 0):
            raise ValueError("invalid Flexiv hand_guiding_motion settings")
        self._hand_guiding = settings.get("hand_guiding", False) is True
        super().__init__(
            arm_components=arms,
            end_effectors=tools,
            model=model,
            models={name: group.kinematics for name, group in model.groups.items()},
            clock=clock,
            execute=execute,
            end_effector_control=end_effector_control,
            stale_timeout_s=settings.get("stale_timeout_s", 0.2),
        )

    @property
    def recovery_actions(self):
        if not self._execute:
            return ()
        actions = ["clear_error"]
        if self.model.home_joints:
            actions.append("home")
        if self._hand_guiding and all(c.backend == "flexivpy" for c in self._controllers):
            actions.extend(("drag", "release", "release_5s"))
        return tuple(actions)

    @property
    def hand_guiding_actions(self):
        if "drag" not in self.recovery_actions:
            return MappingProxyType({})
        return MappingProxyType({**super().hand_guiding_actions,
                                 "release": self.recover_release,
                                 "release_5s": partial(self.recover_release, duration_s=5.0)})

    @property
    def drag_selections(self):
        selections = {label: (name,) for label, name in
                      (("A", "left_arm"), ("B", "right_arm")) if name in self.arm_components}
        if len(selections) == 2:
            selections["AB"] = ("left_arm", "right_arm")
        return MappingProxyType(selections)

    def _require_idle_recovery(self):
        if not self._execute or self._controller_open or self._end_effector_open:
            raise RuntimeError("recovery requires a disconnected executing robot")
        if any(c._robot is not None for c in self._controllers):
            raise RuntimeError("previous Flexiv recovery cleanup must complete first")

    def _close_controllers(self, controllers):
        errors = []
        for controller in reversed(controllers):
            try:
                controller.close()
                if controller in self._controller_open:
                    self._controller_open.remove(controller)
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Flexiv recovery cleanup failed", errors)

    def clear_errors(self):
        self._require_idle_recovery()
        errors = []
        for controller in self._controllers:
            try:
                controller.clear_errors()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Flexiv fault reset failed", errors)

    def _guide(self, groups, stop, on_active, *, stiffness_scale, follow, duration_s=None):
        self._require_idle_recovery()
        if "drag" not in self.recovery_actions:
            raise RuntimeError("Flexiv hand guiding is not configured for this SDK")
        groups = tuple(dict.fromkeys(groups))
        if not groups or set(groups) - self.arm_components.keys():
            raise ValueError("hand guiding must name configured arm groups")
        if duration_s is not None and (not np.isfinite(duration_s) or duration_s <= 0):
            raise ValueError("hand guiding duration must be positive")
        opened = []
        try:
            for group in groups:
                self._check_cancel(stop)
                controller = self.arm_components[group].controller
                opened.append(controller)
                self._controller_open.append(controller)
                # Recovery owns enabling after all selected arms pass preflight.
                previous = controller.enable_on_connect
                controller.enable_on_connect = False
                try:
                    controller.connect()
                finally:
                    controller.enable_on_connect = previous
                controller.prepare_hand_guiding(
                    stiffness_scale, self._guiding_motion
                )
            # Validate every selected arm before enabling and entering guiding.
            for controller in opened:
                self._check_cancel(stop)
                controller.enable(stop)
            for controller in opened:
                self._check_cancel(stop)
                controller.begin_hand_guiding(stop)
            self._check_cancel(stop)
            if on_active is not None:
                on_active()
            deadline = None if duration_s is None else time.monotonic() + duration_s
            while not stop.is_set():
                for controller in opened:
                    if follow:
                        controller.follow_hand_guiding(stop)
                    else:
                        controller.check_hand_guiding()
                if deadline is not None and time.monotonic() >= deadline:
                    break
                stop.wait(1 / self._guiding_motion["control_hz"])
        except Exception as error:
            try:
                self._close_controllers(opened)
            except Exception as cleanup:
                raise ExceptionGroup("Flexiv hand guiding and cleanup failed", [error, cleanup]) from None
            raise
        self._close_controllers(opened)

    def drag(self, groups, stop, on_active=None):
        self._guide(groups, stop, on_active, stiffness_scale=self._drag_scale, follow=True)

    def recover_release(self, groups, stop, on_active=None, *, duration_s=None):
        self._guide(groups, stop, on_active, stiffness_scale=0.0, follow=False,
                    duration_s=duration_s)

    @staticmethod
    def _check_cancel(stop):
        if stop.is_set():
            raise InterruptedError("Flexiv recovery cancelled")

    def home(self):
        """Home the arms, stop arm motion, then open controlled grippers and verify arrival."""
        if not self._execute:
            return
        stop = getattr(self, "_home_stop", None) or threading.Event()
        self._check_cancel(stop)
        if not self._ready:
            raise RuntimeError("Home requires a connected robot")
        targets = dict(self.model.home_joints)
        if set(targets) != set(self.arm_components):
            raise ValueError("Home requires a target for every configured arm")
        settings = self._home_settings
        state = self.get_state()
        start = {g: state.groups[g][:7].copy() for g in targets}
        # Validate every target before enabling or commanding either arm.
        for group, q in targets.items():
            c = self.arm_components[group].controller
            c.validate_home_commands({c.channel: q})
        duration = max(
            (np.pi / 2) * np.max(np.abs(targets[g] - start[g])) /
            min(np.deg2rad(settings["peak_velocity_deg_s"]),
                self.arm_components[g].controller.max_velocity)
            for g in targets
        )
        try:
            for c in self._controllers:
                c.enable(stop)
            begin = time.monotonic()
            deadline = begin + duration + settings["settle_timeout_s"]
            while True:
                self._check_cancel(stop)
                now = time.monotonic()
                fraction = 1.0 if duration == 0 else min((now - begin) / duration, 1.0)
                weight = 0.5 * (1 - np.cos(np.pi * fraction))
                measured = self.get_state()
                if fraction >= 1 and all(np.max(np.abs(measured.groups[g][:7] - q)) <=
                                         np.deg2rad(settings["tolerance_deg"]) for g, q in targets.items()):
                    break
                if now >= deadline:
                    raise TimeoutError("Flexiv Home did not settle before timeout")
                for group, q in targets.items():
                    self._check_cancel(stop)
                    c = self.arm_components[group].controller
                    c.send_home_commands({c.channel: start[group] + weight * (q - start[group])})
                stop.wait(1 / settings["control_hz"])
            # Leave arms in IDLE before opening the tools; subsequent execution prepares NRT.
            self._close_home_motion()
            if self._end_effector_control and self.end_effectors:
                for tool in self.end_effectors.values():
                    self._check_cancel(stop)
                    tool.send_command(GripperCommand(1.0))
                deadline = time.monotonic() + settings["settle_timeout_s"]
                while True:
                    self._check_cancel(stop)
                    measured = self.get_state()
                    if all(measured.groups[name][-1] >= 0.98 for name in self.end_effectors):
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Flexiv Home grippers did not fully open")
                    stop.wait(1 / settings["control_hz"])
        except Exception as error:
            try:
                self.stop()
            except Exception as cleanup:
                raise ExceptionGroup("Flexiv Home and stop failed", [error, cleanup]) from None
            raise

    def _close_home_motion(self):
        errors = []
        for c in self._controllers:
            try:
                c.stop()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Flexiv Home stop failed", errors)

    def recover_home(self, stop=None):
        self._require_idle_recovery()
        self._home_stop = stop if stop is not None else threading.Event()
        previous_control = self._end_effector_control
        # Explicit idle Return Home opens the grippers independently of policy control.
        self._end_effector_control = True
        try:
            self._check_cancel(self._home_stop)
            flags = {c: c.enable_on_connect for c in self._controllers}
            try:
                for c in self._controllers:
                    c.enable_on_connect = False
                self.connect()
            finally:
                for c, flag in flags.items():
                    c.enable_on_connect = flag
            self.home()
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup:
                raise ExceptionGroup("Flexiv Home and cleanup failed", [error, cleanup]) from None
            raise
        finally:
            self._home_stop = None
            self._end_effector_control = previous_control
        self.close()

    @classmethod
    def from_config(cls, path, **options):
        return cls(RobotModel.from_config(path), **options)


def build_robot(config, clock):
    hz = config.get("control_hz", 50)
    if not np.isfinite(hz) or not 1 <= hz <= 100:
        raise ValueError("Flexiv NRT control_hz must be within 1..100 Hz")
    robot = FlexivRobot.from_config(config["config"], clock=clock, **config.get("options", {}))
    if config["group_dims"] != {
        name: model.num_coordinates for name, model in robot.kinematics.models.items()
    }:
        raise ValueError("group_dims must match the Flexiv assembly")
    return robot
