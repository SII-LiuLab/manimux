"""Tianji Marvin dual-arm RobotDriver.

The driver owns what is specific to this body and works at whatever rate the
ManiMux runtime calls it. Joint limits, velocity and acceleration envelopes,
interpolation and step shaping are not here: they come from the control
profile's ``command_safety`` and ``motion_limits`` and are enforced by the
shared runtime, like every other robot.

Body-specific responsibilities:

* SDK call sequences, as in CalibWrist's real-robot path and the tianji-control
  drivers it calls: controller link self-check, position mode with speed-ratio
  readback, batched joint targets, soft stop, confirmed servo-off.
* Units and layout: ``left_arm`` is arm A, ``right_arm`` is arm B; a group is 7
  joints in radians, followed by the gripper aperture (0 closed, 1 open) with
  the ``umi_follower`` end effector.
* Hardware faults: controller error codes, an active arm leaving position mode,
  feedback frames that stop advancing for ``max_feedback_age_s``, SDK FK that
  disagrees with the DH model (checked at connect), gripper protection faults,
  and a command further than ``max_tracking_error_deg`` from the measured joints.
* ``execute: false`` (the default) is read-only: nothing is enabled or sent and
  the gripper motors stay unpowered. ``gripper_control`` needs ``execute`` with
  both arms active.
* ``home()`` moves the active arms through ``home_waypoints_deg`` to
  ``home_joints_deg`` on tianji-control's cosine-eased joint move.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.clock import Clock
from manimux.config import RobotConfig
from manimux.kinematics.tianji import NUM_ARM_JOINTS, TianjiKinematics
from manimux.robots._interrupt import finish_move_before_interrupt
from manimux.types import RobotCommand, RobotState

from . import sdk
from .gripper import GripperSettings, TianjiGrippers
from .marvin import (
    ARM_INDEX,
    STATE_POSITION,
    ArmFeedback,
    FloatArray,
    MarvinArmModel,
    MarvinSession,
    describe_error,
)

log = logging.getLogger("manimux.robots.tianji")

GROUP_ORDER = ("left_arm", "right_arm")
ARM_OF_GROUP = {"left_arm": "A", "right_arm": "B"}
END_EFFECTOR_INPUTS = {"umi_follower": 1, "none": 0}

DEFAULT_ROBOT_IP = "192.168.1.190"
# tianji-control config.HOME_JOINTS.
DEFAULT_HOME_JOINTS_DEG = {
    "left_arm": (90.0, -90.0, -90.0, -90.0, 0.0, 0.0, 0.0),
    "right_arm": (-90.0, -90.0, 90.0, -90.0, 0.0, 0.0, 0.0),
}
HOME_RATE_HZ = 250.0
FK_TOLERANCE_MM = 0.01
FK_TOLERANCE_DEG = 0.01


def _flag(options: Mapping[str, object], key: str) -> bool:
    value = options.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"robot.options.{key} must be true or false")
    return value


def _positive(options: Mapping[str, object], key: str, default: float) -> float:
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"robot.options.{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"robot.options.{key} must be positive")
    return number


def _percentage(options: Mapping[str, object], key: str, default: int | None) -> int | None:
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
            "gripper_control",
            "end_effector",
            "sdk_root",
            "vel_ratio",
            "acc_ratio",
            "max_tracking_error_deg",
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
            "gripper_read_hz",
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
        self._execute = _flag(options, "execute")
        _flag(options, "home_on_close")  # read by the runtime; validated here
        active = options.get("active_arms", list(GROUP_ORDER))
        if (
            not isinstance(active, list | tuple)
            or not active
            or len(set(active)) != len(active)
            or any(name not in GROUP_ORDER for name in active)
        ):
            raise ValueError("robot.options.active_arms must list left_arm and/or right_arm")
        self._active = tuple(name for name in GROUP_ORDER if name in active)
        self._gripper_control = _flag(options, "gripper_control")
        if self._gripper_control and (
            not self._execute or not self._gripper or self._active != GROUP_ORDER
        ):
            raise ValueError(
                "gripper_control requires execute with both arms active and the "
                "umi_follower end effector"
            )
        sdk_root = options.get("sdk_root")
        self._sdk_root = None if sdk_root is None else str(sdk_root)
        self._vel_ratio = _percentage(options, "vel_ratio", None)
        self._acc_ratio = _percentage(options, "acc_ratio", 100)
        if self._execute and self._vel_ratio is None:
            raise ValueError(
                "robot.options.execute requires vel_ratio, the controller speed percentage"
            )
        self._max_tracking_deg = _positive(options, "max_tracking_error_deg", 5.0)
        self._max_feedback_age_s = _positive(options, "max_feedback_age_s", 0.1)
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
        self._home_speed_deg_s = _positive(options, "home_speed_deg_s", 6.0)
        self._gripper_serials = {
            "A": None
            if options.get("left_gripper_sn") is None
            else str(options["left_gripper_sn"]),
            "B": None
            if options.get("right_gripper_sn") is None
            else str(options["right_gripper_sn"]),
        }
        self._gripper_settings = GripperSettings(
            hz=int(_positive(options, "gripper_hz", 100)),
            kp=_positive(options, "gripper_kp", 8.0),
            kd=_positive(options, "gripper_kd", 0.3),
            grip_margin=_positive(options, "gripper_grip_margin", 0.036),
            max_torque_nm=_positive(options, "gripper_max_torque_nm", 1.0),
            stale_ms=_positive(options, "gripper_stale_ms", 200.0),
            read_hz=_positive(options, "gripper_read_hz", 30.0),
        )
        self._dh = TianjiKinematics()

        self._session: MarvinSession | None = None
        self._models: dict[str, MarvinArmModel] = {}
        self._grippers: TianjiGrippers | None = None
        self._prepared: list[str] = []
        self._last_serials: tuple[int, int] | None = None
        self._last_serials_ns = 0
        self._measured_deg: dict[str, FloatArray] = {}
        self._sequence = 0

    # ---------- lifecycle ----------

    def connect(self) -> None:
        if self._session is not None:
            return
        # xense.taccap must be imported before the Marvin bindings.
        taccap = sdk.load_taccap() if self._gripper else None
        fx_kine = sdk.load_marvin_kine(self._sdk_root)
        config_path = sdk.kine_config(self._sdk_root)
        models = {
            name: MarvinArmModel(fx_kine, ARM_INDEX[ARM_OF_GROUP[name]], config_path)
            for name in GROUP_ORDER
        }
        session = MarvinSession(sdk.load_marvin_robot(self._sdk_root), self._ip)
        try:
            session.connect()
            self._session = session
            self._models = models
            self._check_kinematics(session.read())
            if taccap is not None:
                grippers = TianjiGrippers(
                    taccap,
                    self._gripper_serials,
                    control=self._gripper_control,
                    settings=self._gripper_settings,
                )
                grippers.start()
                self._grippers = grippers
            if self._execute:
                assert self._vel_ratio is not None and self._acc_ratio is not None
                for name in self._active:
                    session.prepare_position_mode(
                        ARM_OF_GROUP[name], self._vel_ratio, self._acc_ratio
                    )
                    self._prepared.append(name)
            log.info(
                "Tianji connected to %s (controller version %s, %s, active %s)",
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

    def _check_kinematics(self, feedback: tuple[ArmFeedback, ArmFeedback]) -> None:
        """The SDK's FK and the DH model must agree before anything relies on them."""

        for name, arm in zip(GROUP_ORDER, feedback, strict=True):
            q_deg = arm.joints_deg
            sdk_fk = self._models[name].fk_mm(q_deg)
            dh_fk = self._dh.flange(np.radians(q_deg))
            if sdk_fk.shape != (4, 4) or not np.isfinite(sdk_fk).all():
                raise ValueError(f"{name}: invalid SDK FK matrix")
            position_error = float(np.linalg.norm(sdk_fk[:3, 3] - dh_fk[:3, 3] * 1e3))
            rotation_error = float(
                np.degrees(Rotation.from_matrix(sdk_fk[:3, :3].T @ dh_fk[:3, :3]).magnitude())
            )
            if position_error > FK_TOLERANCE_MM or rotation_error > FK_TOLERANCE_DEG:
                raise ValueError(
                    f"{name}: SDK FK versus DH mismatch ({position_error:.4f} mm, "
                    f"{rotation_error:.4f} deg)"
                )

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
                f"robot feedback did not advance for {(now - self._last_serials_ns) / 1e9:.3f} s"
            )
        if any(arm.err_code for arm in feedback):
            raise RuntimeError(
                "robot error: "
                + "; ".join(
                    f"{name} state {arm.cur_state} {describe_error(arm.err_code)}"
                    for name, arm in zip(GROUP_ORDER, feedback, strict=True)
                )
            )
        apertures = self._grippers.apertures() if self._grippers is not None else {}
        groups: dict[str, FloatArray] = {}
        for name, arm in zip(GROUP_ORDER, feedback, strict=True):
            if self._execute and name in self._active and arm.cur_state != STATE_POSITION:
                raise RuntimeError(
                    f"{name} left position mode (state {arm.cur_state}); emergency stop?"
                )
            if arm.joints_deg.shape != (NUM_ARM_JOINTS,) or not np.isfinite(arm.joints_deg).all():
                raise ValueError(f"{name}: invalid joint feedback")
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
            if self._gripper and not 0.0 <= values[NUM_ARM_JOINTS] <= 1.0:
                raise ValueError(f"Tianji {name} gripper command is outside [0, 1]")
            targets[name] = values
        if not self._execute:
            return

        joints_deg = {name: np.degrees(targets[name][:NUM_ARM_JOINTS]) for name in self._active}
        for name, joints in joints_deg.items():
            measured = self._measured_deg.get(name)
            if measured is None:
                raise RuntimeError("read the robot state before commanding it")
            deviation = np.abs(joints - measured)
            joint = int(np.argmax(deviation))
            if deviation[joint] > self._max_tracking_deg:
                self.stop()
                raise RuntimeError(
                    f"{ARM_OF_GROUP[name]} tracking error {deviation[joint]:.3f} deg on "
                    f"J{joint + 1} exceeds {self._max_tracking_deg:.3f} deg (all joints: "
                    + " ".join(f"J{k + 1}={v:.2f}" for k, v in enumerate(deviation))
                    + ")"
                )
        if self._grippers is not None:
            fault = self._grippers.check()
            if fault:
                self.stop()
                raise RuntimeError(f"gripper protection: {fault}")
        session.send_joints({ARM_OF_GROUP[name]: joints for name, joints in joints_deg.items()})
        if self._grippers is not None and self._gripper_control:
            for name in GROUP_ORDER:
                self._grippers.set_target(ARM_OF_GROUP[name], float(targets[name][NUM_ARM_JOINTS]))

    def home(self) -> None:
        self._require_session()
        if not self._execute:
            log.info("Tianji is read-only; skipping home")
            return
        depth = max((len(self._waypoints.get(name, [])) for name in self._active), default=0)
        stages = [
            {
                name: self._waypoints[name][index]
                for name in self._active
                if index < len(self._waypoints.get(name, []))
            }
            for index in range(depth)
        ]
        stages.append({name: self._home[name] for name in self._active})
        for stage in stages:
            self._move_joints(stage)

    def _move_joints(self, targets: Mapping[str, FloatArray]) -> None:
        """tianji-control ``move_to_joints``: one cosine-eased time base for every axis."""

        session = self._require_session()
        for name, target in targets.items():
            model = self._models[name]
            if np.any(target < model.lim_n) or np.any(target > model.lim_p):
                raise ValueError(f"home target for {name} is outside the controller limits")
        feedback = session.read()
        start = {name: feedback[ARM_INDEX[ARM_OF_GROUP[name]]].joints_deg for name in targets}
        delta = {name: targets[name] - start[name] for name in targets}
        distance = max(float(np.max(np.abs(value))) for value in delta.values())
        if distance <= 1e-3:
            return
        duration = (math.pi / 2.0) * distance / self._home_speed_deg_s
        period = 1.0 / HOME_RATE_HZ
        deadline = time.monotonic() + duration * 2.0 + 5.0
        with finish_move_before_interrupt("Tianji home", log) as interrupted:
            tick = 0
            while True:
                elapsed = tick * period
                if elapsed >= duration:
                    fraction = 1.0
                else:
                    fraction = 0.5 * (1.0 - math.cos(math.pi * elapsed / duration))
                command = {name: start[name] + delta[name] * fraction for name in targets}
                session.send_joints({ARM_OF_GROUP[name]: value for name, value in command.items()})
                tick += 1
                if fraction >= 1.0:
                    break
                if time.monotonic() > deadline:
                    self.stop()
                    raise RuntimeError("Tianji home timed out")
                if tick % 25 == 0:  # tracking check every 0.1 s
                    feedback = session.read()
                    for name, value in command.items():
                        measured = feedback[ARM_INDEX[ARM_OF_GROUP[name]]].joints_deg
                        error = float(np.max(np.abs(value - measured)))
                        if error > self._max_tracking_deg:
                            self.stop()
                            raise RuntimeError(
                                f"Tianji home: arm {ARM_OF_GROUP[name]} tracking error "
                                f"{error:.2f} deg exceeds the limit (obstructed?)"
                            )
                time.sleep(period)
        if interrupted:
            log.info("Tianji home finished; the deferred Ctrl-C now applies.")

    def stop(self) -> None:
        """Stop the gripper motors and soft-stop the active arms."""

        if self._grippers is not None:
            self._grippers.stop_control()
        session = self._session
        if session is None or not self._execute:
            return
        for name in self._active:
            try:
                session.stop_running(ARM_OF_GROUP[name])
            except Exception:  # noqa: BLE001 - keep stopping the other arm
                log.exception("stop_running failed on %s", name)

    def close(self) -> None:
        grippers, self._grippers = self._grippers, None
        session, self._session = self._session, None
        try:
            if grippers is not None:
                grippers.stop_control()
            if session is not None:
                for name in self._prepared:
                    if not session.disable(ARM_OF_GROUP[name]):
                        log.warning(
                            "arm %s servo-off not confirmed; motors may still be energised. "
                            "Check the controller state, and cut power if needed.",
                            ARM_OF_GROUP[name],
                        )
        finally:
            try:
                if session is not None:
                    session.close()
            finally:
                if grippers is not None:
                    grippers.close()
                self._prepared.clear()
                self._models.clear()
                self._last_serials = None
                self._measured_deg.clear()


def build_robot(config: RobotConfig, clock: Clock) -> TianjiDualArmDriver:
    return TianjiDualArmDriver(config, clock)
