"""Tianji Marvin dual-arm RobotDriver.

Maps ManiMux's canonical groups onto one Marvin controller (both arms share one
UDP link) and, when mounted, the UMI follower grippers on their own USB links.

* ``left_arm`` is arm A, ``right_arm`` is arm B.
* A group is 7 joints in radians, followed by the gripper aperture (0 closed,
  1 open) when ``end_effector`` is ``umi_follower``.
* ``execute: false`` (the default) is read-only: the arms never leave their
  current mode, no joint target is sent and the gripper motors stay unpowered.
  Commands are still validated against the joint limits.

With ``execute: true`` every command is also checked against the previous one
(``max_joint_rate_deg_s``) and against the measured joints (sustained tracking
error), and a controlled gripper's protection faults stop the arms.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from manimux.clock import Clock
from manimux.config import RobotConfig
from manimux.kinematics.tianji import JOINT_LIMITS_DEG, NUM_ARM_JOINTS
from manimux.robots._interrupt import finish_move_before_interrupt
from manimux.types import RobotCommand, RobotState

from . import sdk
from .gripper import GripperSettings, TianjiGrippers
from .marvin import ARM_INDEX, STATE_ERROR, STATE_POSITION, MarvinSession, describe_error

log = logging.getLogger("manimux.robots.tianji")

FloatArray = NDArray[np.float64]

GROUP_ORDER = ("left_arm", "right_arm")
ARM_OF_GROUP = {"left_arm": "A", "right_arm": "B"}
END_EFFECTOR_INPUTS = {"umi_follower": 1, "none": 0}

DEFAULT_ROBOT_IP = "192.168.1.190"
DEFAULT_HOME_JOINTS_DEG = {
    "left_arm": (90.0, -90.0, -90.0, -90.0, 0.0, 0.0, 0.0),
    "right_arm": (-90.0, -90.0, 90.0, -90.0, 0.0, 0.0, 0.0),
}
HOME_RATE_HZ = 250.0


def _positive(options: Mapping[str, object], key: str, default: float | None) -> float | None:
    value = options.get(key, default)
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"robot.options.{key} must be a positive number")
    return number


def _ratio(options: Mapping[str, object], key: str, default: int | None) -> int | None:
    value = options.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError(f"robot.options.{key} must be an integer percentage in [1, 100]")
    return value


def _joints(value: object, what: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (NUM_ARM_JOINTS,) or not np.isfinite(array).all():
        raise ValueError(f"{what} must be 7 finite joint angles in degrees")
    return array


class TianjiDualArmDriver:
    GROUP_ORDER = GROUP_ORDER

    #: Every key ``robot.options`` may carry; anything else is rejected.
    OPTIONS = frozenset(
        {
            "robot_ip",
            "execute",
            "active_arms",
            "end_effector",
            "sdk_root",
            "vel_ratio",
            "acc_ratio",
            "max_joint_rate_deg_s",
            "limit_margin_deg",
            "joint_limits_deg",
            "max_tracking_error_deg",
            "max_tracking_error_s",
            "max_feedback_age_s",
            "home_joints_deg",
            "home_waypoints_deg",
            "home_speed_deg_s",
            "home_on_close",
            "left_gripper_sn",
            "right_gripper_sn",
            "gripper_hz",
            "gripper_kp",
            "gripper_kd",
            "gripper_grip_margin",
            "gripper_max_torque_nm",
            "gripper_stale_ms",
            "gripper_read_period_s",
        }
    )

    def __init__(self, config: RobotConfig, clock: Clock) -> None:
        options = dict(config.options)
        unknown = sorted(set(options) - self.OPTIONS)
        if unknown:
            raise ValueError(
                f"unknown robot.options for tianji_dual: {', '.join(unknown)}; "
                f"known keys are {', '.join(sorted(self.OPTIONS))}"
            )
        end_effector = str(options.get("end_effector", "umi_follower"))
        if end_effector not in END_EFFECTOR_INPUTS:
            raise ValueError(
                f"tianji_dual supports end_effector {sorted(END_EFFECTOR_INPUTS)}, "
                f"got {end_effector!r}"
            )
        self._width = NUM_ARM_JOINTS + END_EFFECTOR_INPUTS[end_effector]
        if tuple(config.group_dims) != GROUP_ORDER or any(
            config.group_dims[name] != self._width for name in GROUP_ORDER
        ):
            raise ValueError(
                f"tianji_dual with end_effector {end_effector} requires left_arm and "
                f"right_arm groups of dimension {self._width}"
            )
        self._clock = clock
        self._gripper = end_effector == "umi_follower"
        self._ip = str(options.get("robot_ip", DEFAULT_ROBOT_IP))
        execute = options.get("execute", False)
        if not isinstance(execute, bool):
            raise ValueError("robot.options.execute must be true or false")
        self._execute = execute
        active = options.get("active_arms", list(GROUP_ORDER))
        if (
            not isinstance(active, list | tuple)
            or not active
            or len(set(active)) != len(active)
            or any(name not in GROUP_ORDER for name in active)
        ):
            raise ValueError("robot.options.active_arms must list left_arm and/or right_arm")
        self._active = tuple(name for name in GROUP_ORDER if name in active)
        sdk_root = options.get("sdk_root")
        self._sdk_root = None if sdk_root is None else str(sdk_root)

        self._vel_ratio = _ratio(options, "vel_ratio", None)
        self._acc_ratio = _ratio(options, "acc_ratio", 100)
        self._max_rate_deg_s = _positive(options, "max_joint_rate_deg_s", None)
        if self._execute and (self._vel_ratio is None or self._max_rate_deg_s is None):
            raise ValueError(
                "robot.options.execute requires vel_ratio and max_joint_rate_deg_s, "
                "matching the controller speed confirmed on this robot"
            )
        margin = float(options.get("limit_margin_deg", 5.0))  # type: ignore[arg-type]
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("robot.options.limit_margin_deg must be non-negative")
        self._lower, self._upper = self._joint_limits(options.get("joint_limits_deg", {}), margin)
        self._max_tracking_deg = _positive(options, "max_tracking_error_deg", 5.0) or 5.0
        self._max_tracking_s = _positive(options, "max_tracking_error_s", 0.5) or 0.5
        self._max_feedback_age_s = _positive(options, "max_feedback_age_s", 0.1) or 0.1
        home = options.get("home_joints_deg", DEFAULT_HOME_JOINTS_DEG)
        if not isinstance(home, Mapping) or set(home) != set(GROUP_ORDER):
            raise ValueError("robot.options.home_joints_deg must map left_arm and right_arm")
        self._home = {name: _joints(home[name], f"home_joints_deg.{name}") for name in GROUP_ORDER}
        waypoints = options.get("home_waypoints_deg", {})
        if not isinstance(waypoints, Mapping) or not set(waypoints) <= set(GROUP_ORDER):
            raise ValueError("robot.options.home_waypoints_deg must map arm groups to joint lists")
        self._waypoints = {
            name: [_joints(point, f"home_waypoints_deg.{name}") for point in points]
            for name, points in waypoints.items()
        }
        self._home_speed_deg_s = _positive(options, "home_speed_deg_s", 6.0) or 6.0
        self._gripper_serials = {
            "A": None
            if options.get("left_gripper_sn") is None
            else str(options["left_gripper_sn"]),
            "B": None
            if options.get("right_gripper_sn") is None
            else str(options["right_gripper_sn"]),
        }
        self._gripper_settings = GripperSettings(
            hz=int(options.get("gripper_hz", 100)),  # type: ignore[call-overload]
            kp=float(options.get("gripper_kp", 8.0)),  # type: ignore[arg-type]
            kd=float(options.get("gripper_kd", 0.3)),  # type: ignore[arg-type]
            grip_margin=float(options.get("gripper_grip_margin", 0.036)),  # type: ignore[arg-type]
            max_torque_nm=float(options.get("gripper_max_torque_nm", 1.0)),  # type: ignore[arg-type]
            stale_ms=float(options.get("gripper_stale_ms", 200.0)),  # type: ignore[arg-type]
            read_period_s=float(options.get("gripper_read_period_s", 0.05)),  # type: ignore[arg-type]
        )

        self._session: MarvinSession | None = None
        self._grippers: TianjiGrippers | None = None
        self._prepared: list[str] = []
        self._measured_deg: dict[str, FloatArray] = {}
        self._last_serials: tuple[int, int] | None = None
        self._last_serials_ns = 0
        self._last_sent: tuple[int, dict[str, FloatArray]] | None = None
        self._tracking_since_ns: int | None = None
        self._sequence = 0

    @staticmethod
    def _joint_limits(
        overrides: object, margin: float
    ) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
        if not isinstance(overrides, Mapping) or not set(overrides) <= set(GROUP_ORDER):
            raise ValueError("robot.options.joint_limits_deg must map arm groups to overrides")
        lower, upper = {}, {}
        for name in GROUP_ORDER:
            low = np.array([limit[0] for limit in JOINT_LIMITS_DEG], dtype=np.float64)
            high = np.array([limit[1] for limit in JOINT_LIMITS_DEG], dtype=np.float64)
            per_joint = overrides.get(name, {})
            if not isinstance(per_joint, Mapping):
                raise ValueError(f"joint_limits_deg.{name} must map joint numbers 1-7 to [lo, hi]")
            for joint, bounds in per_joint.items():
                index = int(joint) - 1
                pair = np.asarray(bounds, dtype=np.float64)
                if not 0 <= index < NUM_ARM_JOINTS or pair.shape != (2,) or pair[0] >= pair[1]:
                    raise ValueError(f"joint_limits_deg.{name}.{joint} must be [lo, hi] in degrees")
                # An override may only narrow the controller's own limits.
                low[index] = max(low[index], pair[0])
                high[index] = min(high[index], pair[1])
            low += margin
            high -= margin
            if np.any(low >= high):
                raise ValueError(f"limit_margin_deg leaves no travel on {name}")
            lower[name], upper[name] = low, high
        return lower, upper

    # ---------- lifecycle ----------

    def connect(self) -> None:
        if self._session is not None:
            return
        taccap = sdk.load_taccap() if self._gripper else None  # must precede the arm SDK
        session = MarvinSession(sdk.load_marvin_robot(self._sdk_root), self._ip)
        try:
            session.connect()
            self._session = session
            feedback = session.read()
            for name in self._active:
                arm = feedback[ARM_INDEX[ARM_OF_GROUP[name]]]
                if arm.err_code or arm.cur_state == STATE_ERROR:
                    raise RuntimeError(
                        f"{name} (arm {ARM_OF_GROUP[name]}) reports {describe_error(arm.err_code)} "
                        f"in state {arm.cur_state}; clear it on the controller first"
                    )
            if self._execute:
                assert self._vel_ratio is not None and self._acc_ratio is not None
                for name in self._active:
                    session.prepare_position_mode(
                        ARM_OF_GROUP[name], self._vel_ratio, self._acc_ratio
                    )
                    self._prepared.append(name)
            if taccap is not None:
                grippers = TianjiGrippers(
                    taccap,
                    self._gripper_serials,
                    control_arms=[ARM_OF_GROUP[name] for name in self._active]
                    if self._execute
                    else (),
                    settings=self._gripper_settings,
                )
                grippers.start()
                self._grippers = grippers
            log.info(
                "Tianji connected to %s (controller version %s, %s, arms %s)",
                self._ip,
                session.version,
                "EXECUTE" if self._execute else "read-only",
                ", ".join(self._active),
            )
        except BaseException as primary_error:
            cleanup_error: BaseException | None = None
            try:
                if self._session is None:
                    session.close()
                else:
                    self.close()
            except BaseException as exc:
                cleanup_error = exc
            if cleanup_error is not None:
                raise BaseExceptionGroup(
                    "Tianji connection failed and cleanup was incomplete",
                    [primary_error, cleanup_error],
                ) from None
            raise

    def _require_session(self) -> MarvinSession:
        if self._session is None:
            raise RuntimeError("Tianji dual-arm driver is not connected")
        return self._session

    def get_state(self) -> RobotState:
        session = self._require_session()
        feedback = session.read()
        now = self._clock.now_ns()
        serials = (feedback[0].frame_serial, feedback[1].frame_serial)
        if serials != self._last_serials:
            self._last_serials, self._last_serials_ns = serials, now
        elif now - self._last_serials_ns > self._max_feedback_age_s * 1e9:
            raise RuntimeError(
                f"Marvin feedback stalled for {(now - self._last_serials_ns) / 1e9:.3f} s "
                f"(frame serials {serials})"
            )
        apertures = self._grippers.apertures() if self._grippers is not None else {}
        groups: dict[str, FloatArray] = {}
        for name in GROUP_ORDER:
            arm = feedback[ARM_INDEX[ARM_OF_GROUP[name]]]
            if arm.err_code or arm.cur_state == STATE_ERROR:
                raise RuntimeError(
                    f"{name} reports {describe_error(arm.err_code)} in state {arm.cur_state}"
                )
            if self._execute and name in self._active and arm.cur_state != STATE_POSITION:
                raise RuntimeError(
                    f"{name} left position mode (state {arm.cur_state}); emergency stop?"
                )
            if arm.joints_deg.shape != (NUM_ARM_JOINTS,) or not np.isfinite(arm.joints_deg).all():
                raise RuntimeError(f"{name} returned invalid joint feedback {arm.joints_deg}")
            self._measured_deg[name] = arm.joints_deg
            values = np.radians(arm.joints_deg)
            if self._gripper:
                values = np.append(values, apertures[ARM_OF_GROUP[name]])
            groups[name] = values
        self._sequence += 1
        return RobotState(groups=groups, monotonic_ns=now, sequence=self._sequence)

    def send_command(self, command: RobotCommand) -> None:
        session = self._require_session()
        targets: dict[str, FloatArray] = {}
        for name in GROUP_ORDER:
            if name not in command.groups:
                raise ValueError(f"Tianji command is missing group {name}")
            values = np.asarray(command.groups[name], dtype=np.float64)
            if values.shape != (self._width,) or not np.isfinite(values).all():
                raise ValueError(f"Tianji {name} command must be {self._width} finite values")
            targets[name] = values
        joints_deg = {name: np.degrees(targets[name][:NUM_ARM_JOINTS]) for name in self._active}
        for name, joints in joints_deg.items():
            outside = np.flatnonzero((joints < self._lower[name]) | (joints > self._upper[name]))
            if outside.size:
                index = int(outside[0])
                raise ValueError(
                    f"{name} J{index + 1} command {joints[index]:.2f} deg is outside "
                    f"[{self._lower[name][index]:.1f}, {self._upper[name][index]:.1f}]"
                )
        if not self._execute:
            return

        missing = [name for name in self._active if name not in self._measured_deg]
        if missing:
            raise RuntimeError("read the robot state before commanding it")
        if self._last_sent is None:
            for name in self._active:
                jump = float(np.max(np.abs(joints_deg[name] - self._measured_deg[name])))
                if jump > self._max_tracking_deg:
                    raise RuntimeError(
                        f"{name} first command is {jump:.2f} deg from the measured joints"
                    )
        else:
            dt = (command.monotonic_ns - self._last_sent[0]) / 1e9
            if dt <= 0:
                self._halt()
                raise RuntimeError("Tianji command timestamps must increase")
            assert self._max_rate_deg_s is not None
            for name in self._active:
                step = float(np.max(np.abs(joints_deg[name] - self._last_sent[1][name])))
                limit = self._max_rate_deg_s * dt * 1.05
                if step > limit:
                    self._halt()
                    raise RuntimeError(
                        f"{name} joint step {step:.3f} deg exceeds {limit:.3f} deg "
                        f"in {dt * 1e3:.1f} ms"
                    )
        worst_name, worst = max(
            (
                (name, float(np.max(np.abs(joints_deg[name] - self._measured_deg[name]))))
                for name in self._active
            ),
            key=lambda item: item[1],
        )
        if worst > self._max_tracking_deg:
            if self._tracking_since_ns is None:
                self._tracking_since_ns = command.monotonic_ns
            elif command.monotonic_ns - self._tracking_since_ns > self._max_tracking_s * 1e9:
                self._halt()
                raise RuntimeError(
                    f"{worst_name} tracking error {worst:.2f} deg stayed above "
                    f"{self._max_tracking_deg:.2f} deg for more than {self._max_tracking_s:.2f} s"
                )
        else:
            self._tracking_since_ns = None
        if self._grippers is not None:
            fault = self._grippers.check()
            if fault is not None:
                self._halt()
                raise RuntimeError(f"gripper protection: {fault}")
        session.send_joints({ARM_OF_GROUP[name]: joints for name, joints in joints_deg.items()})
        if self._grippers is not None:
            for name in self._active:
                self._grippers.set_target(ARM_OF_GROUP[name], float(targets[name][NUM_ARM_JOINTS]))
        self._last_sent = (command.monotonic_ns, joints_deg)

    def _halt(self) -> None:
        """Soft-stop the active arms and hold the grippers; used on command faults."""

        session = self._session
        if session is None:
            return
        for name in self._active:
            try:
                session.stop_running(ARM_OF_GROUP[name])
            except Exception:  # noqa: BLE001 - already failing; keep stopping the rest
                log.exception("stop_running failed on %s", name)
        if self._grippers is not None:
            try:
                self._grippers.hold()
            except Exception:  # noqa: BLE001
                log.exception("gripper hold failed")

    def home(self) -> None:
        self._require_session()
        if not self._execute:
            log.info("Tianji is read-only; skipping home")
            return
        stages: list[dict[str, FloatArray]] = []
        depth = max((len(self._waypoints.get(name, [])) for name in self._active), default=0)
        for index in range(depth):
            stage = {
                name: self._waypoints[name][index]
                for name in self._active
                if index < len(self._waypoints.get(name, []))
            }
            stages.append(stage)
        stages.append({name: self._home[name] for name in self._active})
        for stage in stages:
            self._move_joints(stage, what="Tianji home")

    def _move_joints(self, targets: Mapping[str, FloatArray], *, what: str) -> None:
        """Cosine-eased joint move shared by every axis, with tracking checks."""

        session = self._require_session()
        for name, target in targets.items():
            if np.any(target < self._lower[name]) or np.any(target > self._upper[name]):
                raise ValueError(f"{what} target for {name} is outside the joint limits")
        feedback = session.read()
        start = {name: feedback[ARM_INDEX[ARM_OF_GROUP[name]]].joints_deg for name in targets}
        delta = {name: targets[name] - start[name] for name in targets}
        distance = max(float(np.max(np.abs(value))) for value in delta.values())
        self._last_sent = None
        if distance <= 1e-3:
            return
        duration = (math.pi / 2.0) * distance / self._home_speed_deg_s
        period = 1.0 / HOME_RATE_HZ
        with finish_move_before_interrupt(what, log) as interrupted:
            started = time.monotonic()
            tick = 0
            while True:
                elapsed = tick * period
                fraction = (
                    1.0
                    if elapsed >= duration
                    else 0.5 * (1.0 - math.cos(math.pi * elapsed / duration))
                )
                command = {name: start[name] + delta[name] * fraction for name in targets}
                session.send_joints({ARM_OF_GROUP[name]: value for name, value in command.items()})
                tick += 1
                if tick % 25 == 0 or fraction >= 1.0:
                    feedback = session.read()
                    for name, value in command.items():
                        measured = feedback[ARM_INDEX[ARM_OF_GROUP[name]]].joints_deg
                        error = float(np.max(np.abs(value - measured)))
                        if error > self._max_tracking_deg:
                            self._halt()
                            raise RuntimeError(
                                f"{what}: {name} tracking error {error:.2f} deg (obstructed?)"
                            )
                if fraction >= 1.0:
                    break
                if time.monotonic() > started + 2.0 * duration + 5.0:
                    self._halt()
                    raise RuntimeError(f"{what} timed out")
                time.sleep(max(0.0, started + tick * period - time.monotonic()))
        if interrupted:
            log.info("%s finished; the deferred Ctrl-C now applies.", what)

    def stop(self) -> None:
        session = self._session
        if session is None or not self._execute:
            return
        feedback = session.read()
        session.send_joints(
            {
                ARM_OF_GROUP[name]: feedback[ARM_INDEX[ARM_OF_GROUP[name]]].joints_deg
                for name in self._active
            }
        )
        if self._grippers is not None:
            self._grippers.hold()
        self._last_sent = None

    def close(self) -> None:
        errors: list[BaseException] = []
        grippers, self._grippers = self._grippers, None
        if grippers is not None:
            try:
                grippers.close()
            except BaseException as exc:
                errors.append(exc)
        session, self._session = self._session, None
        if session is not None:
            for name in reversed(self._prepared):
                try:
                    if not session.disable(ARM_OF_GROUP[name]):
                        errors.append(
                            RuntimeError(
                                f"{name} did not confirm servo-off; the motors may still be "
                                "energised"
                            )
                        )
                except BaseException as exc:
                    errors.append(exc)
            try:
                session.close()
            except BaseException as exc:
                errors.append(exc)
        self._prepared.clear()
        self._measured_deg.clear()
        self._last_serials = None
        self._last_sent = None
        self._tracking_since_ns = None
        if errors:
            raise RuntimeError("Tianji shutdown did not complete safely") from errors[0]


def build_robot(config: RobotConfig, clock: Clock) -> TianjiDualArmDriver:
    return TianjiDualArmDriver(config, clock)
