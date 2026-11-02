"""YAM uses the installed i2rt directly; each command contains six joints and the built-in gripper."""

import logging
import threading
import time
from enum import Enum

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.arm.base import ArmBase, ArmController, ArmModel, ArmState
from manimux.timing import stage

from .kinematics import YamManipulatorKinematics, visual_configuration

log = logging.getLogger(__name__)


class YamController(ArmController):
    """Each CAN channel owns an SDK instance; construction stores parameters and connect opens the device.

    The SDK manages motor loops and gripper normalization; this layer adapts the ManiMux lifecycle and state interface.
    State timestamps are host times after SDK reads complete, not motor sampling times.
    """

    def __init__(
        self, *, channel, clock=None, arm_type="yam", gripper_type="linear_4310",
        feedback_timeout_s=0.2, release_home_tolerance_rad=0.05,
        release_velocity_tolerance_rad_s=0.1,
        home_settle_duration_s=0.25, move_intervals=50, **hardware_options
    ):
        self.channel = channel
        self.clock = clock if clock is not None else SystemClock()
        self.arm_type = arm_type
        self.gripper_type = gripper_type
        # Old station/config snapshots may still carry this removed deadline.
        # Never forward it as an unknown option into the SDK constructor.
        if "home_settle_timeout_s" in hardware_options:
            hardware_options.pop("home_settle_timeout_s")
            log.warning("home_settle_timeout_s is retired; Home waits for measured arrival")
        self.hardware_options = hardware_options
        self.robot = None
        self.sequence = 0
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.release_home_tolerance_rad = float(release_home_tolerance_rad)
        self.release_velocity_tolerance_rad_s = float(release_velocity_tolerance_rad_s)
        self.home_settle_duration_s = float(home_settle_duration_s)
        if not all(np.isfinite(x) and x > 0 for x in (
            self.feedback_timeout_s, self.release_home_tolerance_rad,
            self.release_velocity_tolerance_rad_s,
            self.home_settle_duration_s,
        )):
            raise ValueError("YAM feedback/release tolerances must be finite and positive")
        self._supported_release = False
        if type(move_intervals) is not int or move_intervals < 1:
            raise ValueError("YAM move_intervals must be a positive integer")
        self.move_intervals = move_intervals

    def connect(self):
        if self.robot is None:
            self._supported_release = False
            from i2rt.robots.get_robot import get_yam_robot
            from i2rt.robots.utils import ArmType, GripperType

            self.robot = get_yam_robot(
                channel=self.channel,
                arm_type=ArmType.from_string_name(self.arm_type),
                gripper_type=GripperType.from_string_name(self.gripper_type),
                **self.hardware_options,
            )

    def get_states(self):
        # Copy current SDK feedback; never substitute the last command for measured positions.
        with stage(f"yam.{self.channel}.sdk_get_joint_pos"):
            sdk_joints = self.robot.get_joint_pos()
        joints = sdk_joints.copy()
        self.sequence += 1
        return {self.channel: ArmState(joints, self.clock.now_ns(), self.sequence)}

    def send_commands(self, targets):
        # The SDK may clip the input array in place; the caller retains ownership of the command.
        target = targets[self.channel].copy()
        with stage(f"yam.{self.channel}.sdk_command_joint_pos"):
            self.robot.command_joint_pos(target)

    def move_joints(self, target, *, time_interval_s, cancel=None):
        """Retain i2rt's linear waypoints with health checks at every submission.

        The SDK helper itself does not stop interpolation after its CAN thread
        fails. The assembly supplies a shared cancellation event for both arms.
        """
        cancel = cancel if cancel is not None else threading.Event()
        try:
            start = self.check_control_health().pos.copy()
            target = np.array(target, dtype=float, copy=True)
            if (target.shape != start.shape or not np.isfinite(target).all()
                    or not np.isfinite(time_interval_s) or time_interval_s <= 0):
                raise ValueError("Invalid YAM move target or duration")
            started = time.monotonic()
            for index in range(self.move_intervals + 1):
                if cancel.is_set():
                    raise RuntimeError("YAM move cancelled after peer-arm failure")
                self.check_control_health()
                if cancel.is_set():
                    raise RuntimeError("YAM move cancelled after peer-arm failure")
                alpha = index / self.move_intervals
                self.robot.command_joint_pos((1 - alpha) * start + alpha * target)
                deadline = started + (index + 1) * time_interval_s / self.move_intervals
                time.sleep(max(0.0, deadline - time.monotonic()))
        except BaseException:
            cancel.set()
            raise

    def sent_command_snapshots(self):
        """Read cached CAN-send evidence; an unpatched SDK has no evidence."""
        getter = getattr(self.robot, "get_sent_command_snapshot", None)
        sample = getter() if callable(getter) else None
        return {} if sample is None else {self.channel: sample}

    def runtime_metadata(self):
        """Read connected SDK settings and calibrated bounds without device I/O."""
        from manimux.recording.provenance import (
            installed_package_provenance,
            loaded_module_provenance,
        )

        result = {
            "sdk": installed_package_provenance("i2rt"),
            "connected": self.robot is not None,
            "channel": self.channel,
            "arm_type": self.arm_type,
            "gripper_type": self.gripper_type,
            "captured_monotonic_ns": self.clock.now_ns(),
            "effective": {},
            "unavailable": {},
        }
        if self.robot is None:
            result["unavailable"]["effective"] = "controller_not_connected"
            return result
        result["sdk"]["loaded_robot_module"] = loaded_module_provenance(type(self.robot).__module__)
        method = getattr(self.robot, "get_robot_info", None)
        if callable(method):
            try:
                info = method()
            except Exception as exc:
                info = {}
                result["unavailable"]["sdk_robot_info"] = type(exc).__name__
        else:
            info = {}
            result["unavailable"]["sdk_robot_info"] = "sdk_method_not_available"
        fields = {
            "kp": "kp",
            "kd": "kd",
            "grav_comp_kd": "grav_comp_kd",
            "joint_limits_rad": "joint_limits",
            "gripper_limits_raw_rad": "gripper_limits",
            "gripper_index": "gripper_index",
            "gripper_force_limit_n": "limit_gripper_effort",
            "gravity_comp_factor": "gravity_comp_factor",
            "use_coulomb_friction": "use_coulomb_friction",
            "coulomb_friction": "coulomb_friction",
            "enable_auto_recovery": "enable_auto_recovery",
        }
        for output, key in fields.items():
            result["effective"][output] = _metadata_value(info.get(key))
            if key not in info:
                result["unavailable"][output] = "sdk_field_not_available"
        chain = getattr(self.robot, "motor_chain", None)
        limiter = getattr(self.robot, "_gripper_force_limiter", None)
        attributes = {
            "use_gravity_comp": (self.robot, "use_gravity_comp"),
            "clip_motor_torque_nm": (self.robot, "_clip_motor_torque"),
            "motor_offset_rad": (chain, "motor_offset"),
            "motor_direction": (chain, "motor_direction"),
            "last_reported_communication_hz": (chain, "comm_freq"),
            "gripper_effort_average_window_s": (limiter, "average_torque_window"),
            "gripper_clog_effort_threshold_nm": (limiter, "clog_force_threshold"),
            "gripper_clog_speed_threshold": (limiter, "clog_speed_threshold"),
        }
        for output, (owner, key) in attributes.items():
            if owner is None or not hasattr(owner, key):
                result["effective"][output] = None
                result["unavailable"][output] = "sdk_field_not_available"
            else:
                result["effective"][output] = _metadata_value(getattr(owner, key))
        result["gripper_calibration"] = {
            "effective_closed_open_raw_rad": result["effective"]["gripper_limits_raw_rad"],
            "source": "connected_sdk_gripper_limits",
            "procedure": None,
            "procedure_reason": "sdk_does_not_report_calibration_or_override_origin",
        }
        return result

    def stop(self):
        if self.robot is not None:
            self.check_control_health()
            self.robot.command_joint_pos(self.robot.get_joint_pos().copy())

    def check_control_health(self):
        """Require advancing CAN feedback AND SDK state, not refreshed read timestamps."""
        native = self.robot
        if native is None:
            raise RuntimeError(f"{self.channel}: no connected YAM controller")
        chain = native.motor_chain
        with chain.state_lock:
            previous_can = chain.state
        with native._state_lock:
            previous_joint = native._joint_state
        deadline = time.monotonic() + self.feedback_timeout_s
        while True:
            if (not native._server_thread.is_alive() or native._stop_event.is_set()
                    or not chain.running):
                raise RuntimeError(f"{self.channel}: YAM control loop is not healthy")
            with chain.state_lock:
                current_can = chain.state
                faults = [s.error_code for s in current_can if s.error_code != "0x1"]
            if faults:
                raise RuntimeError(f"{self.channel}: motor fault {faults}")
            with native._state_lock:
                current_joint = native._joint_state
            if current_can is not previous_can and current_joint is not previous_joint:
                if (not np.isfinite(current_joint.pos).all()
                        or not np.isfinite(current_joint.vel).all()):
                    raise RuntimeError(f"{self.channel}: nonfinite YAM feedback")
                return current_joint
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{self.channel}: CAN or SDK feedback stopped advancing")
            time.sleep(0.002)

    def validate_release(self):
        if self.robot is None or self._supported_release:
            return
        state = self.check_control_health()
        detail = self._home_error(state)
        if detail:
            raise RuntimeError(detail)

    def _home_error(self, state):
        positions, speeds = np.abs(state.pos[:6]), np.abs(state.vel[:6])
        if (positions.max() <= self.release_home_tolerance_rad
                and speeds.max() <= self.release_velocity_tolerance_rad_s):
            return ""
        return (
            f"{self.channel}: YAM is not stationary at zero Home; keep control alive; "
            f"max_position_error_rad={positions.max():.6f} "
            f"(joint={int(positions.argmax()) + 1}, limit={self.release_home_tolerance_rad}); "
            f"max_speed_rad_s={speeds.max():.6f} "
            f"(joint={int(speeds.argmax()) + 1}, limit={self.release_velocity_tolerance_rad_s})"
        )

    def wait_for_home(self, *, cancel=None):
        """Observe fresh feedback while the last Home target remains active.

        No new target, automatic retry or threshold relaxation is permitted here.
        There is no arrival deadline. Hardware-health failures still propagate
        immediately; elapsed settling time alone never authorizes release.
        """
        next_report = time.monotonic() + 5.0
        stable_since = None
        while True:
            if cancel is not None and cancel.is_set():
                raise RuntimeError(f"{self.channel}: Home verification cancelled after peer-arm failure")
            state = self.check_control_health()
            now = time.monotonic()
            detail = self._home_error(state)
            if detail:
                stable_since = None
            else:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= self.home_settle_duration_s:
                    return
            if now >= next_report:
                log.warning(
                    "%s: still returning Home; control retained; %s",
                    self.channel, detail or "waiting for continuous stationary feedback",
                )
                next_report = now + 5.0
            time.sleep(0.01)

    def confirm_supported_release(self):
        self._supported_release = True

    def close(self):
        if self.robot is None:
            return
        while True:
            try:
                self.validate_release()
                break
            except Exception as error:
                # Recheck after Home; a late fault cannot inherit permission
                # from an earlier recovery attempt.
                from manimux.embodiments.robot.release import (
                    ReleaseDecision,
                    require_supported_release,
                )
                if require_supported_release(str(error)) is ReleaseDecision.SUPPORTED_RELEASE:
                    self.confirm_supported_release()
                    break
        native = self.robot
        # The installed i2rt.close only joins service threads. Preserve the shutdown order: both loops
        # must end before releasing CAN so the motor loop cannot access a closed socket.
        native._stop_event.set()
        native._server_thread.join(timeout=2.0)
        if native._server_thread.is_alive():
            raise RuntimeError("i2rt robot server thread did not stop; CAN left open")
        chain = native.motor_chain
        chain.running = False
        threads = [
            thread
            for thread in threading.enumerate()
            if getattr(getattr(thread, "_target", None), "__self__", None) is chain
        ]
        for thread in threads:
            thread.join(timeout=2.0)
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError("i2rt motor control thread did not stop; CAN left open")
        chain.close()
        self.robot = None


def _metadata_value(value):
    """Convert copied SDK values to JSON without serializing device objects."""
    if isinstance(value, Enum):
        return _metadata_value(value.value)
    if isinstance(value, np.ndarray):
        return _metadata_value(value.tolist())
    if isinstance(value, np.generic):
        return _metadata_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


class YamArm(ArmBase):
    """All seven coordinates are controlled; end_effector: null retains the component's built-in gripper."""

    num_joints = 7

    def __init__(self, controller, *, kinematics):
        self._controller = controller
        self._kinematics = kinematics

    @classmethod
    def load_model(cls, **options):
        model = YamManipulatorKinematics(**options)
        return ArmModel(
            model,
            model.coordinates,
            model.solver._assets_root / "i2rt/robot_models/arm/yam/yam.urdf",
            None,  # The integrated component provides a complete TCP, without a flange for a separate tool.
            base_frame=model.base_frame,
            visual_mapping=visual_configuration,
        )

    @property
    def controller(self):
        return self._controller

    @property
    def channel(self):
        return self._controller.channel

    @property
    def kinematics(self):
        return self._kinematics
