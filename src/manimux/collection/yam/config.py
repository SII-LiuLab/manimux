"""Configuration dataclasses and a minimal YAML loader.

Deliberately plain: no ``_target_`` reflection magic. Configs are dataclasses;
the loader fills them from YAML dicts, ignoring unknown keys so a config file
can carry extra documentation fields without breaking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# --- configurable option sets ---
CONTROLLER_TYPE_OPTIONS = [
    "yam_lead_left",
    "yam_lead_right",
    "passive_gello_left",
    "passive_gello_right",
    "mobile_gello_left",
    "mobile_gello_right",
]
ROBOT_TYPE_OPTIONS = ["yam_left", "yam_right"]
GRIPPER_OPTIONS = ["linear_4310", "linear_3507", "crank_4310", "flexible_4310"]
CAMERA_TYPE_OPTIONS = [
    {"value": "realsense", "label": "RealSense"},
    {"value": "orbbec", "label": "Orbbec Gemini RGB"},
    {"value": "decxin_mono", "label": "Decxin Mono"},
    {"value": "decxin_stereo", "label": "Decxin Stereo"},
]
CAMERA_ROLE_OPTIONS = ["top", "left", "right", "wrist", "gemini305", "gemini335"]
FORMAT_OPTIONS = ["default"]

# Which CAN bus each device identity maps to. The operator picks controller/robot
# *types* in the GUI; the udev channel names stay a station-setup detail derived here.
# A controller (leader/teaching arm) drives a robot (follower arm).
_CONTROLLER_CHANNELS = {
    "yam_lead_left": "can_lead_l",
    "yam_lead_right": "can_lead_r",
    # Passive GELLO leaders (desk or mobile) reuse the same leader CAN buses.
    "passive_gello_left": "can_lead_l",
    "passive_gello_right": "can_lead_r",
    "mobile_gello_left": "can_lead_l",
    "mobile_gello_right": "can_lead_r",
}
_ROBOT_CHANNELS = {"yam_left": "can_left", "yam_right": "can_right"}

# Passive-GELLO joint directions (arm joints then gripper), copied from
# common/hardware/src/common_hardware/robots/passive_gello.py (lab42); keep them in
# lockstep with its _REGULAR_JOINT_SIGNS / _MOBILE_JOINT_SIGNS. A desk GELLO's two arms
# share one table; the mobile rig's right arm is mirrored, so each side has its own.
# The gripper element is inert: the trigger normalizes by |angle| (see
# passive_gello._normalize_gripper).
_REGULAR_JOINT_SIGNS = [-1, -1, 1, 1, -1, -1, 1]
_MOBILE_JOINT_SIGNS = {
    "left": [-1, 1, 1, 1, -1, -1, -1],
    "right": [-1, -1, -1, -1, -1, -1, 1],
}


# Standard udev channel names, offered in the GUI's channel pickers alongside whatever
# CAN interfaces are actually up on the station (see can_bus.list_can_interfaces).
CAN_CHANNEL_OPTIONS = ["can_left", "can_right", "can_lead_l", "can_lead_r"]


def controller_channel(controller_type: str) -> str:
    return _CONTROLLER_CHANNELS.get(controller_type, "can_lead_l")


def controller_channel_for(controller: ControllerConfig) -> str:
    """The leader's explicit ``channel``, else the type-derived default."""
    return (controller.channel or "").strip() or controller_channel(controller.type)


def robot_channel_for(robot: RobotUnitConfig) -> str:
    """The follower's explicit ``channel``, else the type-derived default."""
    return (robot.channel or "").strip() or robot_channel(robot.type)


def check_channel_conflicts(
    robots: list[RobotUnitConfig], controllers: list[ControllerConfig]
) -> None:
    """Raise if two devices resolve to the same CAN bus. Each device opens its own
    ``can.interface.Bus``, so a collision surfaces as intermittent dropped frames
    rather than a clean failure."""
    used: dict[str, list[str]] = {}
    for r in robots:
        used.setdefault(robot_channel_for(r), []).append(f"robot {r.type}")
    for c in controllers:
        used.setdefault(controller_channel_for(c), []).append(f"controller {c.type}")
    clashes = {ch: who for ch, who in used.items() if len(who) > 1}
    if clashes:
        detail = "; ".join(f"{ch} <- {', '.join(who)}" for ch, who in sorted(clashes.items()))
        raise ValueError(
            f"CAN channel assigned to more than one device ({detail}). "
            f"Give each robot and controller its own bus."
        )


def is_passive_gello(controller_type: str) -> bool:
    """Whether a controller type is a passive-encoder GELLO (no motors) rather
    than a motorized YAM lead arm. Drives which leader driver is built. Both the
    desk (``passive_gello_*``) and mobile (``mobile_gello_*``) GELLOs are passive."""
    return controller_type.startswith(("passive_gello", "mobile_gello"))


def gello_variant(controller_type: str) -> tuple[str, str | None]:
    """``(gello_type, side)`` for a passive-GELLO controller type, mirroring the fleet
    driver's ``--type``/``--side``: ``("mobile", "left"|"right")`` for a mobile rig,
    ``("regular", None)`` for a desk GELLO."""
    if controller_type.startswith("mobile_gello"):
        return "mobile", ("right" if controller_type.endswith("right") else "left")
    return "regular", None


def leader_joint_signs_for(
    controller: ControllerConfig, station_signs: list[int] | None
) -> list[int] | None:
    """Per-joint directions for one leader: the controller's own ``joint_signs`` first,
    then the station-wide ``leader_joint_signs``, then the variant's built-in table. A
    mobile rig inverts the last two: its sides need different tables, which one
    station-wide list can't express, so ``_MOBILE_JOINT_SIGNS`` wins over
    ``station_signs``."""
    if controller.joint_signs:
        return list(controller.joint_signs)
    gello_type, side = gello_variant(controller.type)
    if gello_type == "mobile":
        return list(_MOBILE_JOINT_SIGNS[side])
    return station_signs if station_signs else list(_REGULAR_JOINT_SIGNS)


def robot_channel(robot_type: str) -> str:
    return _ROBOT_CHANNELS.get(robot_type, "can_left")


def _from_dict(cls, data: dict[str, Any]):
    """Build a dataclass from a dict, ignoring keys that aren't fields."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class RobotUnitConfig:
    """A follower robot arm and the gripper mounted on it.

    ``channel`` overrides the CAN bus derived from ``type`` (see ``robot_channel_for``);
    unset keeps the udev default. ``gripper_limits`` pins this arm's gripper
    ``[closed, open]`` motor range (rad). Left unset, i2rt re-measures it on every build
    by driving the motor into both stops (``needs_calibration``), and a blocked or stiff
    gripper measures a collapsed span that maps the whole trigger travel onto almost no
    motion. Pin it to keep the travel repeatable; see ``docs/hardware.md``.
    """

    type: str = "yam_left"
    gripper: str = "linear_4310"
    gripper_limits: list[float] | None = None
    channel: str | None = None


@dataclass
class ControllerConfig:
    """A teleop controller (leader / teaching arm) and the robot it controls
    (by robot ``type``).

    ``channel`` overrides the CAN bus derived from ``type`` (see ``controller_channel_for``);
    ``joint_signs`` overrides this leader's directions (see ``leader_joint_signs_for``).
    """

    type: str = "yam_lead_left"
    controls: str = "yam_left"
    joint_signs: list[int] | None = None
    channel: str | None = None


def _default_robots() -> list[RobotUnitConfig]:
    return [
        RobotUnitConfig("yam_left", "linear_4310"),
        RobotUnitConfig("yam_right", "linear_4310"),
    ]


def _default_controllers() -> list[ControllerConfig]:
    return [
        ControllerConfig("yam_lead_left", "yam_left"),
        ControllerConfig("yam_lead_right", "yam_right"),
    ]


@dataclass
class RobotConfig:
    type: str = "yam"  # "yam" | "mock"
    arm_name: str = "left"  # schema prefix for the single follower arm
    arm_type: str = "yam"  # i2rt ArmType: "yam" | "yam_pro" | "yam_ultra" | "big_yam"
    num_arm_joints: int = 6  # arm DOFs, excluding the gripper
    gripper_type: str = "linear_4310"  # follower's powered gripper (fallback)
    leader_gripper_type: str = "yam_teaching_handle"
    # ``analog`` mirrors the trigger continuously. ``toggle`` treats each full
    # squeeze as one edge: close, release, squeeze again to open.
    leader_gripper_mode: str = "analog"
    gripper_close_duration_s: float = 0.0  # toggle close ramp; 0 keeps immediate closing
    gripper_force_limit: float = 50.0  # follower blocked-gripper force limit (N)
    ee_mass: float | None = None  # override link_6 mass (kg) for i2rt gravity comp
    follower_channel: str = "can_left"
    leader_channel: str = "can_lead_l"
    # Bilateral force feedback: leader PD gain scale (0 = free-floating leader on
    # gravity comp; >0 commands the leader toward the follower pose for haptics).
    # Ignored for passive-GELLO leaders, which have no motors to drive.
    bilateral_kp: float = 0.0
    # Passive-GELLO leader calibration (used only for a passive-GELLO controller
    # type). Arm encoders are device ids 0..num_arm_joints-1; the gripper is a
    # separate encoder. ``leader_joint_signs`` has length num_arm_joints+1 (arm
    # joints then gripper), applied as ``angle = sign*raw``. It is the station-wide
    # default; ``leader_joint_signs_for`` resolves the per-leader override.
    # The home zero is stored in each encoder's EEPROM (``calibrate_gello.py zero``),
    # so there is no software offset. ``leader_gripper`` is
    # ``[device_id, closed_rad, open_rad]``.
    leader_joint_signs: list[int] | None = None
    leader_gripper: list[float] | None = None
    # Operator-configurable robots + controllers (GUI Station rail). Leader->follower
    # is an identity map (both are YAM arms), so no per-joint signs/offsets are
    # needed. The runtime drives robots[0] and the controller bound to it.
    robots: list[RobotUnitConfig] = field(default_factory=_default_robots)
    controllers: list[ControllerConfig] = field(default_factory=_default_controllers)

    def active_robot(self) -> RobotUnitConfig:
        return self.robots[0] if self.robots else RobotUnitConfig()

    def controller_for(self, robot: RobotUnitConfig) -> ControllerConfig:
        """The controller that drives ``robot`` (matched by robot type), falling
        back to the first controller if the binding can't be resolved."""
        for c in self.controllers:
            if c.controls == robot.type:
                return c
        return self.controllers[0] if self.controllers else ControllerConfig()


@dataclass
class CameraConfig:
    name: str
    type: str  # "decxin_mono" | "decxin_stereo" | "realsense" | "orbbec" | "mock"
    role: str  # unique stream prefix, e.g. "top", "left", "right", "gemini305"
    mode: str = "mono"  # "mono" | "stereo"
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    enable_depth: bool = False
    exposure_us: int | None = None  # fixed RGB exposure (us); None = leave sensor defaults
    white_balance: int | None = None  # fixed WB (K); None = leave sensor defaults


def validate_camera_configs(cameras: list[CameraConfig]) -> None:
    """Names index live frames; roles index saved streams. Both must be unique."""
    for field_name in ("name", "role"):
        values = [getattr(c, field_name) for c in cameras]
        duplicates = sorted({v for v in values if values.count(v) > 1})
        if duplicates:
            raise ValueError(f"duplicate camera {field_name}(s): {duplicates}")


@dataclass
class StationConfig:
    collector: str = "yam"
    execution_mode: str = "synchronous"
    manimux_config: str = "configs/collection/yam/control.yaml"
    robot: RobotConfig = field(default_factory=RobotConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    control_hz: float = 30.0
    save_root: str = "data/episodes"
    task_name: str = "pick_and_place"
    data_format: str = "default"  # default on-disk format for new episodes
    # Pose the followers ramp to before a policy takes over, so its first observation
    # is in-distribution. Per-arm ``[joints..., gripper]`` concatenated in ``robots``
    # order -- the same layout as the policy state vector, so 14 values for a 2-arm YAM
    # (gripper normalized [0, 1]). This is the default the GUI's Deploy tab pre-fills
    # and the CLI's ``--home-pose`` falls back to; clearing the field skips homing.
    deploy_home_pose: list[float] | None = None

    def __post_init__(self):
        if self.collector != "yam":
            raise ValueError("YAM station requires collector: yam")
        if self.execution_mode not in {"synchronous", "threaded"}:
            raise ValueError("execution_mode must be synchronous or threaded")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_station_config(
    station_path: str | Path,
    cameras_path: str | Path | None = None,
) -> StationConfig:
    """Load a station YAML (and optional cameras YAML) into a StationConfig."""
    raw = load_yaml(station_path)

    raw_robot = raw.get("robot", {})
    robot = _from_dict(RobotConfig, raw_robot)
    if raw_robot.get("controllers"):
        robot.controllers = [_from_dict(ControllerConfig, c) for c in raw_robot["controllers"]]
    if raw_robot.get("robots"):
        robot.robots = [_from_dict(RobotUnitConfig, r) for r in raw_robot["robots"]]
    _types = [r.type for r in robot.robots]
    _dups = sorted({t for t in _types if _types.count(t) > 1})
    if _dups:
        raise ValueError(
            f"duplicate robot type(s) {_dups} in config; each robot needs a distinct type"
        )
    check_channel_conflicts(robot.robots, robot.controllers)

    cam_dicts = raw.get("cameras")
    if not cam_dicts:
        cpath = cameras_path or raw.get("cameras_config")
        if cpath:
            craw = load_yaml(cpath)
            cam_dicts = craw.get("cameras", craw if isinstance(craw, list) else [])
        else:
            cam_dicts = []
    cameras = [_from_dict(CameraConfig, c) for c in cam_dicts]
    validate_camera_configs(cameras)

    station = _from_dict(StationConfig, raw)
    station.robot = robot
    station.cameras = cameras
    _apply_control_profile(station, raw)
    return station


def _apply_control_profile(station: StationConfig, raw: dict) -> None:
    from manimux.config import load_config

    runtime = load_config(station.manimux_config)
    if runtime.control_profile is None:
        return
    frequency = 1.0 / runtime.policy.effective_action_dt_s
    if "control_hz" in raw and not math.isclose(station.control_hz, frequency, rel_tol=1e-9):
        raise ValueError("station control_hz conflicts with control_profile")
    station.control_hz = frequency
    raw_robot = raw.get("robot", {})
    shared = runtime.robot.options
    if runtime.execution.motion_limits is not None:
        closing_velocity = runtime.execution.motion_limits.gripper.max_closing_velocity
        duration = 0.0 if closing_velocity is None else 1.0 / closing_velocity
        if "gripper_close_duration_s" in raw_robot and not math.isclose(
            station.robot.gripper_close_duration_s, duration, rel_tol=1e-9
        ):
            raise ValueError("station gripper_close_duration_s conflicts with control_profile")
        station.robot.gripper_close_duration_s = duration
    for setting in ("arm_type", "gripper_force_limit", "ee_mass"):
        left = shared.get("left_hardware_options", {}).get(setting)
        right = shared.get("right_hardware_options", {}).get(setting)
        if left != right:
            raise ValueError(f"YAM collection requires matching left/right {setting}")
        if left is not None:
            if setting in raw_robot and getattr(station.robot, setting) != left:
                raise ValueError(f"station robot.{setting} conflicts with control_profile")
            setattr(station.robot, setting, left)
    for robot in station.robot.robots:
        side = robot.type.removeprefix("yam_")
        hardware = shared.get(f"{side}_hardware_options", {})
        raw_unit = next(
            (unit for unit in raw_robot.get("robots", []) if unit.get("type") == robot.type), {}
        )
        values = {
            "channel": shared.get(f"{side}_channel"),
            "gripper": hardware.get("gripper_type"),
            "gripper_limits": hardware.get("gripper_limits_override"),
        }
        for setting, value in values.items():
            if value is not None:
                if setting in raw_unit and getattr(robot, setting) != value:
                    raise ValueError(
                        f"station {robot.type}.{setting} conflicts with control_profile"
                    )
                setattr(robot, setting, value)
    check_channel_conflicts(station.robot.robots, station.robot.controllers)


def _mode_for_camera(cam_type: str) -> str:
    return "stereo" if cam_type == "decxin_stereo" else "mono"


def apply_station_form(base: StationConfig, form: dict[str, Any]) -> StationConfig:
    """Overlay the operator-editable fields from the GUI Station rail onto a base
    config, preserving station calibration (joint signs/offsets, arm_type,
    num_arm_joints, control_hz, camera serials/resolution) that the rail doesn't expose.

    Editable: robots (type + gripper), controllers (type + controlled robot),
    camera list (name/role/type), output format, save_root, task name.
    """
    import copy

    cfg = copy.deepcopy(base)

    robots = form.get("robots") or []
    if robots:
        # The rail doesn't expose gripper_limits, so carry the base config's pin, but
        # only while the gripper type is unchanged: another gripper's range silently
        # mis-maps the trigger travel (same reasoning as camera serials below).
        base_robots = {rb.type: rb for rb in base.robot.robots}
        new_robots = []
        for r in robots:
            rtype = r.get("type", "yam_left")
            gripper = r.get("gripper", "linear_4310")
            prev = base_robots.get(rtype)
            new_robots.append(
                RobotUnitConfig(
                    type=rtype,
                    gripper=gripper,
                    gripper_limits=(
                        prev.gripper_limits if (prev and prev.gripper == gripper) else None
                    ),
                    # The rail edits this one, so blank means "back to the type default".
                    channel=(r.get("channel") or "").strip() or None,
                )
            )
        cfg.robot.robots = new_robots

    controllers = form.get("controllers") or []
    if controllers:
        # The rail doesn't expose joint_signs, so carry the base config's override by type.
        base_signs = {c.type: c.joint_signs for c in base.robot.controllers}
        cfg.robot.controllers = [
            ControllerConfig(
                type=c.get("type", "yam_lead_left"),
                controls=c.get("controls", "yam_left"),
                joint_signs=base_signs.get(c.get("type", "yam_lead_left")),
                channel=(c.get("channel") or "").strip() or None,
            )
            for c in controllers
        ]

    # The rail can reassign buses, so re-check before the session tries to open them.
    check_channel_conflicts(cfg.robot.robots, cfg.robot.controllers)

    cams = form.get("cameras")
    if cams is not None:
        by_name = {c.name: c for c in base.cameras}
        new_cams = []
        for c in cams:
            name = c.get("name", "cam")
            prev = by_name.get(name)
            cam_type = c.get("type", "realsense")
            # A serial identifies a device within one family (decxin: ID_SERIAL_SHORT;
            # realsense: S/N), so a base serial is only meaningful if the type is
            # unchanged. When the rail switches a camera to a different type, carrying
            # the old serial would point the new driver at a non-existent device (a
            # masked open failure); drop it so the driver falls back to discovery.
            keep = prev if (prev and prev.type == cam_type) else None
            # The rail now exposes the serial (picked from live hardware via
            # /api/cameras/detect), so a form-supplied serial wins over the base's.
            form_serial = (c.get("serial") or "").strip() or None
            new_cams.append(
                CameraConfig(
                    name=name,
                    role=(c.get("role") or (prev.role if prev else None) or name),
                    type=cam_type,
                    mode=_mode_for_camera(cam_type),
                    serial=(form_serial if form_serial else (keep.serial if keep else None)),
                    width=(keep.width if keep else 640),
                    height=(keep.height if keep else 480),
                    fps=(keep.fps if keep else 30),
                    enable_depth=(keep.enable_depth if keep else False),
                    exposure_us=(keep.exposure_us if keep else None),
                    white_balance=(keep.white_balance if keep else None),
                )
            )
        validate_camera_configs(new_cams)
        cfg.cameras = new_cams

    if form.get("data_format"):
        cfg.data_format = form["data_format"]
    if form.get("save_root"):
        cfg.save_root = form["save_root"]
    if form.get("task_name") is not None:
        # Blank is meaningful: it must clear a stale/default task so the physical
        # record button cannot silently label a new episode with the previous task.
        cfg.task_name = form["task_name"].strip()
    return cfg


_CAMERAS_YAML_HEADER = """\
# Camera roster for a YAM-ABC-Reproduce station.
#
# type:    decxin_mono | decxin_stereo | realsense | orbbec | mock
# role:    unique stream prefix (e.g. top, left, right, gemini305, gemini335)
# mode:    mono | stereo
# serial:  USB serial for UVC/Orbbec; SDK S/N for RealSense
#
# Managed by the GUI Station rail: Preview detects live serials and saves them here.
# `yam_abc_reproduce cameras` also lists connected devices with their serials.
"""


def _camera_to_dict(cam: CameraConfig) -> dict[str, Any]:
    """One cameras.yaml entry, fields ordered for readability; ``enable_depth`` is
    RealSense-only so it's omitted for other types."""
    d: dict[str, Any] = {
        "name": cam.name,
        "type": cam.type,
        "role": cam.role,
        "mode": cam.mode,
        "serial": cam.serial,
        "width": cam.width,
        "height": cam.height,
        "fps": cam.fps,
    }
    if cam.type == "realsense":
        d["enable_depth"] = cam.enable_depth
        if cam.exposure_us is not None:
            d["exposure_us"] = cam.exposure_us
        if cam.white_balance is not None:
            d["white_balance"] = cam.white_balance
    return d


def dump_cameras_yaml(cameras: list[CameraConfig]) -> str:
    """Serialize a camera roster to cameras.yaml text (header + ``cameras:`` list)."""
    body = yaml.safe_dump(
        {"cameras": [_camera_to_dict(c) for c in cameras]},
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    return _CAMERAS_YAML_HEADER + "\n" + body


def save_cameras_yaml(path: str | Path, cameras: list[CameraConfig]) -> bool:
    """Write the roster to ``path`` iff it differs from what's already there (so a
    repeated Preview click doesn't churn the file). Returns True if written."""
    text = dump_cameras_yaml(cameras)
    p = Path(path)
    try:
        if p.exists() and p.read_text() == text:
            return False
    except OSError:
        pass
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return True
