"""Component-based robot lifecycle and command coordination, without vendor calls."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from manimux.clock import Clock, SystemClock
from manimux.embodiments.arm.base import ArmBase, ArmController, ArmModel
from manimux.embodiments.end_effector.base import EndEffectorModel
from manimux.embodiments.end_effector.gripper import GripperBase, GripperCommand
from manimux.embodiments.sensor.base import SensorBase
from manimux.kinematics.base import FloatArray, ManipulatorKinematicsBase, rigid_transform
from manimux.kinematics.composed import ComposedManipulatorKinematics
from manimux.kinematics.end_effector import Frame, attach_end_effector
from manimux.kinematics.robot import RobotKinematics
from manimux.kinematics.tool import FixedToolGeometry
from manimux.plugins import load_plugin
from manimux.types import RobotCommand, RobotState


class RobotBase(ABC):
    """Assemble arms and supported position-controlled end effectors.

    Each group contains one arm and an optional one-coordinate GripperBase.
    Other end-effector command capabilities require an explicit mapping; this
    class does not infer control semantics from an arbitrary tool state.

    Controllers are connected, sampled and dispatched once per shared session.
    All targets are validated before dispatch. Independent controllers and end
    effectors are not atomic; any dispatch failure attempts to stop all owned
    components. No vendor SDK, smoothing, enabling policy or IK algorithm lives
    here. Those belong to the arm controller, executor and kinematics model.
    """

    @abstractmethod
    def __init__(
        self,
        *,
        arm_components: Mapping[str, ArmBase],
        models: Mapping[str, ManipulatorKinematicsBase],
        model: RobotModel | None = None,
        end_effectors: Mapping[str, GripperBase] | None = None,
        sensors: Mapping[str, SensorBase] | None = None,
        clock: Clock | None = None,
        stale_timeout_s: float = 0.2,
        execute: bool = True,
        end_effector_control: bool = True,
    ) -> None:
        if not arm_components or set(arm_components) != set(models):
            raise ValueError("arm components and kinematic models must have identical groups")
        tools = dict(end_effectors or {})
        if set(tools) - arm_components.keys():
            raise ValueError("end effectors must belong to configured arm groups")
        if len({id(tool) for tool in tools.values()}) != len(tools):
            raise ValueError("each group must have a distinct end effector instance")
        channels = {(id(arm.controller), arm.channel) for arm in arm_components.values()}
        if len(channels) != len(arm_components):
            raise ValueError("each group must have a distinct controller channel")
        if not np.isfinite(stale_timeout_s) or stale_timeout_s <= 0:
            raise ValueError("stale_timeout_s must be finite and positive")
        # Configured robots and action decoding share this exact offline model.
        self.model = model
        self._kinematics = model.kinematics if model is not None else RobotKinematics(models)
        for name, arm in arm_components.items():
            if models[name].num_coordinates != arm.num_joints + (name in tools):
                raise ValueError(f"{name}: model layout does not match its components")
        self._arm_components = MappingProxyType(dict(arm_components))
        self._end_effectors = MappingProxyType(tools)
        sensor_components = dict(sensors or {})
        self.sensors = MappingProxyType(sensor_components)
        self._sensor_open: set[str] = set()
        self._sensor_started: set[str] = set()
        self._controllers = tuple(dict.fromkeys(arm.controller for arm in arm_components.values()))
        self._controller_open: list[ArmController] = []
        self._end_effector_open: set[str] = set()
        self._clock = clock if clock is not None else SystemClock()
        self._stale_ns = int(stale_timeout_s * 1e9)
        self._sequence = 0
        self._ready = False
        # Only an explicit boolean True enables physical command dispatch.
        self._execute = execute is True
        self._end_effector_control = end_effector_control is True
        self._lock = threading.RLock()

    @property
    def arm_components(self) -> Mapping[str, ArmBase]:
        return self._arm_components

    @property
    def end_effectors(self) -> Mapping[str, GripperBase]:
        return self._end_effectors

    @property
    def kinematics(self) -> RobotKinematics:
        """TCP models in each arm base, shared with policy action decoding."""
        return self._kinematics

    def fk(self, configuration: Mapping[str, FloatArray]) -> dict[str, FloatArray]:
        """Return each group's TCP in its own arm base; radians in, metres out."""
        return self.kinematics.fk(configuration)

    def ik(self, targets, seed, *, fixed_coordinates):
        """Solve per-arm-base TCP targets; fixed tool coordinates include aperture."""
        return self.kinematics.ik(targets, seed, fixed_coordinates=fixed_coordinates)

    def connect(self) -> None:
        with self._lock:
            if self._ready:
                return
            if self._controller_open or self._end_effector_open:
                raise RuntimeError("cleanup incomplete; call close before reconnecting")
            try:
                for controller in self._controllers:
                    self._controller_open.append(controller)
                    controller.connect()
                for name, tool in self.end_effectors.items():
                    self._end_effector_open.add(name)
                    tool.connect()
                self._ready = True
                self.get_state()
            except Exception as error:
                try:
                    self.close()
                except Exception as cleanup_error:
                    raise ExceptionGroup(
                        "connect and cleanup failed", [error, cleanup_error]
                    ) from None
                raise

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("robot is not connected")

    def get_state(self) -> RobotState:
        with self._lock:
            self._require_ready()
            feedback = {controller: controller.get_states() for controller in self._controllers}
            groups, timestamps = {}, []
            for name, arm in self.arm_components.items():
                state = feedback[arm.controller][arm.channel]
                if state.joints.shape != (arm.num_joints,):
                    raise ValueError(f"{name}: invalid arm feedback dimension")
                if not 0 <= self._clock.now_ns() - state.monotonic_ns <= self._stale_ns:
                    raise RuntimeError(f"{name}: stale arm feedback or clock mismatch")
                groups[name] = state.joints.copy()
                timestamps.append(state.monotonic_ns)
                if name in self.end_effectors:
                    tool_state = self.end_effectors[name].get_state()
                    tool_ns = int(tool_state.timestamp * 1e9)
                    if not 0 <= self._clock.now_ns() - tool_ns <= self._stale_ns:
                        raise RuntimeError(f"{name}: stale gripper feedback or clock mismatch")
                    groups[name] = np.r_[groups[name], tool_state.opening]
                    timestamps.append(tool_ns)
            self._sequence += 1
            return RobotState(groups, min(timestamps), self._sequence)

    def send_command(self, command: RobotCommand) -> None:
        with self._lock:
            self._require_ready()
            if set(command.groups) != set(self.arm_components):
                raise ValueError("command must contain all configured groups, without extras")
            batches = {controller: {} for controller in self._controllers}
            tool_commands = {}
            for name, arm in self.arm_components.items():
                q = np.array(command.groups[name], dtype=float, copy=True)
                if (
                    q.shape != (self.kinematics.models[name].num_coordinates,)
                    or not np.isfinite(q).all()
                ):
                    raise ValueError(f"{name}: invalid command dimension/values")
                batches[arm.controller][arm.channel] = q[: arm.num_joints]
                if name in self.end_effectors:
                    tool_commands[name] = GripperCommand(float(q[arm.num_joints]))
            # A read-only experiment still checks action shapes, but never enables
            # an arm or end effector. FK/IK and measured observations remain usable.
            if not self._execute:
                return
            for controller, batch in batches.items():
                controller.validate_commands(batch)
            try:
                self.get_state()
                for controller, batch in batches.items():
                    controller.send_commands(batch)
                if self._end_effector_control:
                    for name, tool_command in tool_commands.items():
                        self.end_effectors[name].send_command(tool_command)
            except Exception as error:
                try:
                    self.stop()
                except Exception as stop_error:
                    raise ExceptionGroup("command and stop failed", [error, stop_error]) from None
                raise

    def home(self) -> None:
        raise NotImplementedError("home trajectory not configured; use the runtime motion planner")

    def start_sensors(self) -> None:
        """Explicitly start owned sensors; connect() does not claim camera devices."""
        with self._lock:
            if self._sensor_open - self._sensor_started:
                raise RuntimeError(
                    "sensor cleanup incomplete; call close_sensors before restarting"
                )
            try:
                for name, sensor in self.sensors.items():
                    if name not in self._sensor_open:
                        self._sensor_open.add(name)
                        sensor.start()
                        self._sensor_started.add(name)
            except Exception as error:
                try:
                    self.close_sensors()
                except Exception as cleanup_error:
                    raise ExceptionGroup(
                        "sensor startup and cleanup failed", [error, cleanup_error]
                    ) from None
                raise

    def read_sensors(self):
        with self._lock:
            if self._sensor_started != self.sensors.keys():
                raise RuntimeError("robot sensors are not started")
            return {name: sensor.read() for name, sensor in self.sensors.items()}

    def close_sensors(self) -> None:
        with self._lock:
            errors = []
            for name in tuple(self._sensor_open):
                self._sensor_started.discard(name)
                try:
                    self.sensors[name].close()
                    self._sensor_open.remove(name)
                except Exception as error:
                    errors.append(error)
            if errors:
                raise ExceptionGroup("sensor cleanup incomplete", errors)

    def stop(self) -> None:
        with self._lock:
            errors = []
            for controller in self._controller_open:
                try:
                    controller.stop()
                except Exception as error:
                    errors.append(error)
            for name in tuple(self._end_effector_open):
                try:
                    self.end_effectors[name].stop()
                except Exception as error:
                    errors.append(error)
            if errors:
                raise ExceptionGroup("robot stop incomplete", errors)

    def close(self) -> None:
        with self._lock:
            self._ready = False
            errors = []
            try:
                self.stop()
            except Exception as error:
                errors.append(error)
            for name in tuple(self._end_effector_open):
                try:
                    self.end_effectors[name].close()
                    self._end_effector_open.remove(name)
                except Exception as error:
                    errors.append(error)
            for controller in tuple(reversed(self._controller_open)):
                try:
                    controller.close()
                    self._controller_open.remove(controller)
                except Exception as error:
                    errors.append(error)
            try:
                self.close_sensors()
            except Exception as error:
                errors.append(error)
            if errors:
                raise ExceptionGroup("robot cleanup incomplete", errors)


# These data classes belong to robot assembly. They neither connect devices nor
# replace a component's official solver; no separate description/assembly layer
# is needed to share them with Viewer and policy adapters.
@dataclass(frozen=True, slots=True)
class MountedGroup:
    arm_name: str
    arm: ArmModel
    end_effector_name: str | None
    end_effector: EndEffectorModel | None
    base_transform: FloatArray  # Scene placement only; never included in control FK/IK.
    mount: Frame
    kinematics: ManipulatorKinematicsBase

    def visual_end_effector(self):
        if self.end_effector is None:
            return None
        visual = self.end_effector.visual
        # The component provides its own tool geometry; installation comes from
        # the robot config. Render the same calibrated fixed TCP as the solver.
        geometry = self.end_effector.geometry
        transform = geometry.tcp_transform(np.asarray(visual.spec.rest_inputs))
        tcp = Frame(
            xyz=tuple(transform[:3, 3]),
            rpy=tuple(Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")),
        )
        spec = visual.spec.model_copy(update={"mount": self.mount, "tcp": tcp})
        return replace(visual, spec=spec)

    def visual_urdf(self) -> Path:
        return attach_end_effector(
            self.arm.urdf_path,
            self.arm.flange_link,
            self.visual_end_effector(),
        )

    def visual_configuration(self, configuration: FloatArray) -> FloatArray:
        q = np.asarray(configuration, dtype=np.float64)
        if q.shape != (self.kinematics.num_coordinates,) or not np.isfinite(q).all():
            raise ValueError("visual configuration must match the full group layout")
        width = len(self.arm.coordinates)
        if self.end_effector is None:
            return q.copy()
        return np.r_[q[:width], self.end_effector.visual.joint_positions(q[width:])]


def _read_mapping(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _mount(value) -> Frame:
    # Frame 负责读取 xyz/rpy；这里只补充刚体变换的数值约束。
    frame = Frame.model_validate(value)
    # Reuse the common rigid-transform validator, including finite values.
    rigid_transform(frame.matrix(), "mount")
    return frame


def _resource_path(value: str, base: Path) -> Path:
    if value.startswith("package://"):
        package, resource = value.removeprefix("package://").split("/", 1)
        path = Path(str(files(package).joinpath(resource)))
    else:
        path = (base / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@dataclass(frozen=True, slots=True)
class RobotModel:
    """Offline assembly shared by control, adapters and Viewer.

    Current groups support an arm mounted on the robot base with an optional
    flange-mounted end effector. Sensors may declare an unknown mount (null):
    they can capture images, but no spatial camera transform is inferred.
    """

    name: str
    root_frame: str
    groups: Mapping[str, MountedGroup]
    kinematics: RobotKinematics
    components: Mapping[str, dict]
    hardware: Mapping[str, object]
    static_assets: tuple[tuple[str, Path, Frame], ...]
    config_path: Path

    @classmethod
    def from_config(cls, path: Path | str) -> RobotModel:
        source = Path(path).expanduser().resolve()
        spec = _read_mapping(source)
        name, root = spec["name"], spec["root_frame"]
        declarations = spec["components"]
        components, arm_models, tool_models, mounts = {}, {}, {}, {}
        for component_name, entry in declarations.items():
            component_source = (source.parent / entry["config"]).resolve()
            component = _read_mapping(component_source)
            component["options"] = {**component.get("options", {}), **entry.get("options", {})}
            component["hardware"] = {**component.get("hardware", {}), **entry.get("hardware", {})}
            component["parent"] = entry["parent"]
            component["mount"] = entry.get("mount")
            component["source"] = component_source
            factory = load_plugin(
                component["implementation"], group="manimux.components", builtins={}
            )
            kind = component["type"]
            component["class"] = factory
            # 安装关系只解析一次，分组装配复用同一个 Frame。
            if kind in {"arm", "end_effector"} or entry.get("mount") is not None:
                mounts[component_name] = _mount(entry.get("mount"))
            if kind == "arm":
                if entry["parent"] != root:
                    raise ValueError("current arm groups must mount directly to root_frame")
                arm_models[component_name] = factory.load_model(**component["options"])
            elif kind == "end_effector":
                tool_models[component_name] = factory.load_model(
                    **component["options"],
                    base_frame=f"{component_name}.base",
                    tcp_frame=f"{component_name}.tcp",
                )
            components[component_name] = component
        groups, used_arms, used_tools = {}, set(), set()
        for group_name, group in spec["groups"].items():
            arm_name, tool_name = group["arm"], group.get("end_effector")
            if arm_name not in arm_models or arm_name in used_arms:
                raise ValueError("each group must select a distinct configured arm")
            arm = arm_models[arm_name]
            base_transform = mounts[arm_name].matrix()
            base_transform.setflags(write=False)
            used_arms.add(arm_name)
            tool = None
            mount = Frame()
            if tool_name is not None:
                if tool_name not in tool_models or tool_name in used_tools:
                    raise ValueError("each group must select a distinct configured end effector")
                if components[tool_name]["parent"] != f"{arm_name}.flange":
                    raise ValueError("end effector parent must be the group's arm flange")
                tool = tool_models[tool_name]
                mount = mounts[tool_name]
                used_tools.add(tool_name)
            geometry = (
                tool.geometry
                if tool
                else FixedToolGeometry(
                    np.eye(4), base_frame=f"{arm_name}.flange", tcp_frame=f"{arm_name}.flange"
                )
            )
            kin = ComposedManipulatorKinematics(
                arm.kinematics,
                geometry,
                arm_coordinates=arm.coordinates,
                mount=mount.matrix(),
                # Control uses each arm base. base_transform is display metadata only.
                base_frame=arm.base_frame,
                position_tolerance=None,  # Preserve the official solver's acceptance rules.
            )
            groups[group_name] = MountedGroup(
                arm_name, arm, tool_name, tool, base_transform, mount, kin
            )
        if used_arms != set(arm_models) or used_tools != set(tool_models):
            raise ValueError("all arm and end-effector components must belong to a group")
        static = []
        for asset_name, entry in spec.get("static_assets", {}).items():
            static.append(
                (
                    asset_name,
                    _resource_path(entry["urdf"], source.parent),
                    _mount(entry["mount"]),
                )
            )
        kin = RobotKinematics({key: value.kinematics for key, value in groups.items()})
        return cls(
            name,
            root,
            MappingProxyType(groups),
            kin,
            MappingProxyType(components),
            MappingProxyType(dict(spec.get("hardware", {}))),
            tuple(static),
            source,
        )
