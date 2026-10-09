"""Tianji–TacCap assembly using public arm and end-effector interfaces."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from manimux.clock import Clock, SystemClock
from manimux.embodiments.arm.tianji.arm import TianjiArm, TianjiArmSettings, TianjiController
from manimux.embodiments.end_effector.gripper import GripperBase
from manimux.embodiments.robot.base import RobotBase, RobotModel
from manimux.kinematics.base import KinematicCoordinate, ManipulatorKinematicsBase
from manimux.types import FloatArray, RobotCommand


@dataclass(frozen=True, slots=True)
class TianjiHomeSettings:
    """Cosine Home motion settings authored in the robot assembly YAML."""

    control_hz: float
    peak_velocity_deg_s: float
    tolerance_deg: float
    settle_timeout_s: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"hardware.home_motion.{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class TianjiArmConfig:
    """One group's complete TCP model and control settings.

    Limits are (lower, upper), seven radians each; ratios are controller percent.
    An injected gripper is exclusively owned by this robot after connect.
    Its Clock must share the robot's domain. No TacCap-specific hardware branch
    exists here; RobotModel loads TacCap-equipped groups from the assembly YAML.
    """

    kinematics: ManipulatorKinematicsBase
    joint_limits: tuple[FloatArray, FloatArray]
    velocity_ratio: int
    acceleration_ratio: int
    gripper: GripperBase | None = None

    def __post_init__(self) -> None:
        settings = TianjiArmSettings(
            self.joint_limits, self.velocity_ratio, self.acceleration_ratio
        )
        object.__setattr__(self, "joint_limits", settings.joint_limits)
        expected = tuple(KinematicCoordinate(f"joint_{i}", "rad") for i in range(1, 8))
        if self.gripper is not None:
            expected += (KinematicCoordinate("gripper", "normalized"),)
        if self.kinematics.coordinates != expected:
            raise ValueError("kinematics layout must match seven joints and optional gripper")


class TianjiTaccapRobot(RobotBase):
    """Bind Tianji arms and optional TacCap-compatible position grippers.

    Native connection, feedback, enabling and batch dispatch belong to the shared
    TianjiController. Generic robot lifecycle and tool coordination are inherited
    from RobotBase. The legacy constructor and per-arm TCP frame conventions are
    retained. Hardware remains disconnected until connect().
    """

    def __init__(
        self,
        *,
        ip: str | None = None,
        arms: Mapping[str, TianjiArmConfig] | None = None,
        model: RobotModel | None = None,
        hardware: Mapping[str, object] | None = None,
        component_hardware: Mapping[str, Mapping] | None = None,
        clock: Clock | None = None,
        stale_timeout_s: float = 0.2,
        ready_timeout_s: float = 3.0,
        execute: bool = True,
        end_effector_control: bool = True,
    ) -> None:
        if model is not None:
            if ip is not None or arms is not None:
                raise ValueError("model cannot be mixed with legacy ip/arms arguments")
            if stale_timeout_s != 0.2 or ready_timeout_s != 3.0:
                raise ValueError("model timeouts must be supplied through hardware")
            self._configure_model(
                model, clock, hardware, component_hardware, execute, end_effector_control
            )
            return
        if hardware is not None or component_hardware is not None:
            raise ValueError("hardware overrides require a robot model")
        if not arms or set(arms) - {"left", "right"}:
            raise ValueError("arms must configure left, right, or both")
        configs = {name: arms[name] for name in ("left", "right") if name in arms}
        for name, cfg in configs.items():
            if cfg.kinematics.base_frame != f"tianji_{name}_base":
                raise ValueError(f"{name} kinematics has the wrong arm base frame")
        clock = clock if clock is not None else SystemClock()
        self.controller = TianjiController(
            ip=ip,
            settings={
                name: TianjiArmSettings(
                    cfg.joint_limits, cfg.velocity_ratio, cfg.acceleration_ratio
                )
                for name, cfg in configs.items()
            },
            clock=clock,
            stale_timeout_s=stale_timeout_s,
            ready_timeout_s=ready_timeout_s,
        )
        self._arms = MappingProxyType(configs)
        super().__init__(
            arm_components={name: TianjiArm(self.controller, name) for name in configs},
            end_effectors={
                name: cfg.gripper for name, cfg in configs.items() if cfg.gripper is not None
            },
            models={name: cfg.kinematics for name, cfg in configs.items()},
            clock=clock,
            stale_timeout_s=stale_timeout_s,
            execute=execute,
            end_effector_control=end_effector_control,
        )

    @property
    def recovery_actions(self) -> tuple[str, ...]:
        # Each action opens the controller itself, so it needs the bound address.
        if not (self._execute and self.controller.ip):
            return ()
        return ("clear_error", "home", "drag")

    @property
    def drag_selections(self) -> Mapping[str, tuple[str, ...]]:
        # RoboGUI uses the Marvin arm labels: A is the left arm, B the right arm.
        by_side = {arm.channel: name for name, arm in self.arm_components.items()}
        selections = {
            label: tuple(by_side[side] for side in sides)
            for label, sides in (("A", ("left",)), ("B", ("right",)), ("AB", ("left", "right")))
            if all(side in by_side for side in sides)
        }
        return MappingProxyType(selections)

    def clear_errors(self) -> None:
        """Clear arm controller faults; call before connect(), which rejects faults."""
        with self._lock:
            if self._controller_open:
                raise RuntimeError("clear_errors requires a disconnected robot")
            self.controller.clear_errors()

    def drag(
        self,
        groups: Sequence[str],
        stop: threading.Event,
        on_active: Callable[[], None] | None = None,
    ) -> None:
        """Hand-guide the named arm groups until stop is set.

        on_active is called once the arms are in drag and may be moved by hand.
        Opens only the shared arm controller; grippers and sensors stay closed.
        Requires execute and a disconnected robot; the arms are disabled on return.
        """
        groups = tuple(dict.fromkeys(groups))
        if not groups or set(groups) - set(self.arm_components):
            raise ValueError("drag groups must name configured arms")
        sides = tuple(self.arm_components[name].channel for name in groups)
        with self._lock:
            if not self._execute:
                raise RuntimeError("drag requires execute=true")
            if self._controller_open or self._end_effector_open:
                raise RuntimeError("drag requires a disconnected robot")
            self.controller.connect()
            try:
                self.controller.drag(tuple(sides), stop, on_active)
            except Exception as error:
                try:
                    self.controller.close()
                except Exception as cleanup_error:
                    raise ExceptionGroup(
                        "drag and cleanup failed", [error, cleanup_error]
                    ) from None
                raise
            self.controller.close()

    def recover_drag(
        self,
        groups: Sequence[str],
        stop: threading.Event,
        on_active: Callable[[], None] | None = None,
    ) -> None:
        """Idle drag; latched faults are cleared first because connect() rejects them."""
        if not self._execute:
            raise RuntimeError("drag requires execute=true")
        self.clear_errors()
        self.drag(groups, stop, on_active)

    def recover_home(self, stop: threading.Event | None = None) -> None:
        """Idle Return Home, also the path after an E-stop: clear faults, then home."""
        previous_control = self._end_effector_control
        self._recovery_home_stop = stop
        # Idle Home restores grippers; this device-specific choice belongs here.
        self._end_effector_control = True
        try:
            self._check_home_cancel()
            self.clear_errors()
            self._check_home_cancel()
            super().recover_home()
        finally:
            self._recovery_home_stop = None
            self._end_effector_control = previous_control

    def _check_home_cancel(self) -> None:
        stop = getattr(self, "_recovery_home_stop", None)
        if stop is not None and stop.is_set():
            raise RuntimeError("Tianji-TacCap Home cancelled")

    def _home_wait(self, period_s: float) -> None:
        stop = getattr(self, "_recovery_home_stop", None)
        if stop is None:
            time.sleep(period_s)
        else:
            stop.wait(period_s)
            self._check_home_cancel()

    def home(self) -> None:
        """Move connected arms to model.home_joints, then fully open the grippers.

        The arms follow the assembly's cosine profile and settle tolerance.
        Grippers open only with end-effector control. No-op without execute.
        """
        if not self._execute:
            return
        self._check_home_cancel()
        targets = {} if self.model is None else dict(self.model.home_joints)
        if not targets:
            raise RuntimeError("Tianji-TacCap Home target is not configured")
        home = self._home_motion
        period_s = 1.0 / home.control_hz
        with self._lock:
            state = self.get_state()
            if set(targets) != set(state.groups):
                raise RuntimeError("Home targets do not match the robot groups")
            start = {name: state.groups[name][: len(q)].copy() for name, q in targets.items()}
            distance = max(float(np.max(np.abs(q - start[name]))) for name, q in targets.items())
            duration_s = (math.pi / 2.0) * distance / math.radians(home.peak_velocity_deg_s)
            deadline = time.monotonic() + duration_s * 2.0 + home.settle_timeout_s
            groups = {name: values.copy() for name, values in state.groups.items()}
            tick = 0
            while True:
                self._check_home_cancel()
                elapsed = tick * period_s
                fraction = (
                    1.0
                    if elapsed >= duration_s
                    else 0.5 * (1.0 - math.cos(math.pi * elapsed / duration_s))
                )
                for name, q in targets.items():
                    groups[name][: len(q)] = start[name] + (q - start[name]) * fraction
                self._check_home_cancel()
                self.send_command(RobotCommand(groups, self._clock.now_ns(), None))
                if fraction >= 1.0:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Tianji-TacCap Home trajectory timed out")
                tick += 1
                self._home_wait(period_s)

            deadline = time.monotonic() + home.settle_timeout_s
            tolerance = math.radians(home.tolerance_deg)
            while True:
                self._check_home_cancel()
                state = self.get_state()
                if all(
                    np.max(np.abs(state.groups[name][: len(q)] - q)) <= tolerance
                    for name, q in targets.items()
                ):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Tianji-TacCap did not reach Home within {home.tolerance_deg:g} degrees"
                    )
                self._check_home_cancel()
                self.send_command(RobotCommand(groups, self._clock.now_ns(), None))
                self._home_wait(period_s)

            if not (self.end_effectors and self._end_effector_control):
                return
            for name in self.end_effectors:
                groups[name][-1] = 1.0
            self._check_home_cancel()
            self.send_command(RobotCommand(groups, self._clock.now_ns(), None))
            deadline = time.monotonic() + home.settle_timeout_s
            while not all(
                self.get_state().groups[name][-1] >= 0.98 for name in self.end_effectors
            ):
                self._check_home_cancel()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Tianji-TacCap grippers did not fully open")
                self._home_wait(period_s)

    @property
    def arms(self) -> Mapping[str, TianjiArmConfig]:
        """Legacy construction settings; arm_components contains the actual arms."""
        return self._arms

    @classmethod
    def from_config(
        cls,
        path: Path | str,
        *,
        clock: Clock | None = None,
        hardware: Mapping[str, object] | None = None,
        component_hardware: Mapping[str, Mapping] | None = None,
        execute: bool = False,
        end_effector_control: bool = False,
    ) -> TianjiTaccapRobot:
        """Assemble configured components without connecting any hardware.

        RobotModel.from_config loads geometry; component constructors store device
        bindings. Missing bindings are reported when connecting or starting devices.
        Both config-based and direct construction keep poses in each arm base.
        Scene placement is excluded from FK/IK.
        """
        return cls(
            model=RobotModel.from_config(path),
            clock=clock,
            hardware=hardware,
            component_hardware=component_hardware,
            execute=execute,
            end_effector_control=end_effector_control,
        )

    def _configure_model(
        self, model, clock, hardware, component_hardware, execute, end_effector_control
    ) -> None:
        clock = clock if clock is not None else SystemClock()
        control = {**model.hardware, **(hardware or {})}
        home_motion = {**model.hardware["home_motion"], **control.pop("home_motion")}
        self._home_motion = TianjiHomeSettings(**home_motion)
        shared_arm_hardware = {
            name: control.pop(name) for name in ("velocity_ratio", "acceleration_ratio")
        }
        # Runtime shaping consumes the rated capability; the SDK only needs percentages.
        control.pop("rated_joint_velocity_rad_s")
        overrides = dict(component_hardware or {})
        for name, options in overrides.items():
            # The rate contract derives runtime limits from the shared ratio only.
            shared = {"velocity_ratio", "acceleration_ratio"}.intersection(options)
            if shared:
                raise ValueError(
                    f"component_hardware.{name} cannot set {sorted(shared)}; set them in "
                    "robot.options.hardware so the runtime limits use the same ratio"
                )
        bound = {name: dict(entry["hardware"]) for name, entry in model.components.items()}
        # 按组件名应用本地绑定；拼错名称时由字典索引直接报错。
        for name, options in overrides.items():
            bound[name].update(options)
        # Both arms share one native controller session; groups retain their YAML order.
        settings, sides = {}, {}
        for name, group in model.groups.items():
            component = model.components[group.arm_name]
            side = component["options"]["side"]
            if side in settings:
                raise ValueError("one controller side cannot be used by multiple groups")
            sides[name] = side
            settings[side] = TianjiArmSettings(
                **{**shared_arm_hardware, **bound[group.arm_name]}
            )
        controller = TianjiController(settings=settings, clock=clock, **control)
        arms, end_effectors, sensors = {}, {}, {}
        configs = {}
        for name, group in model.groups.items():
            arm_class = model.components[group.arm_name]["class"]
            side = sides[name]
            arms[name] = arm_class(controller, side, kinematics=group.arm.kinematics)
            if group.end_effector_name is not None:
                tool_name = group.end_effector_name
                component = model.components[tool_name]
                end_effectors[name] = component["class"](clock=clock, **bound[tool_name])
            cfg = settings[side]
            configs[name] = TianjiArmConfig(
                group.kinematics,
                cfg.joint_limits,
                cfg.velocity_ratio,
                cfg.acceleration_ratio,
                end_effectors.get(name),
            )
        # Sensors stay closed until explicitly started; FK/IK never opens devices.
        for name, component in model.components.items():
            if component["type"] == "sensor":
                options = {**component["options"], **bound[name]}
                sensors[name] = component["class"](name=name, clock=clock, **options)
        super().__init__(
            model=model,
            arm_components=arms,
            end_effectors=end_effectors,
            sensors=sensors,
            models={name: group.kinematics for name, group in model.groups.items()},
            clock=clock,
            stale_timeout_s=control.get("stale_timeout_s", 0.2),
            execute=execute,
            end_effector_control=end_effector_control,
        )
        self.controller = controller
        self._arms = MappingProxyType(configs)


# Compatibility name for existing direct-construction callers.
TianjiRobot = TianjiTaccapRobot
