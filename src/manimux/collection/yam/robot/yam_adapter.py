"""Leader input and button semantics copied from YAM-ABC-Reproduce."""

from __future__ import annotations

import time
from threading import Event

import numpy as np

from .interface import RobotInterface, TeleopAgent


class YamLeaderArm:
    """Passive YAM lead arm read through i2rt.

    Backdrivability: with ``bilateral_kp == 0`` the PD gains are zeroed so the arm
    floats on i2rt's gravity compensation (exactly as i2rt's bimanual_lead_follower
    leader does). With ``bilateral_kp > 0`` the leader is given proportional gains
    and can be commanded toward the follower pose for force feedback.
    """

    def __init__(
        self,
        channel: str,
        num_arm_joints: int = 6,
        arm_type: str = "yam",
        gripper_type: str = "yam_teaching_handle",
        ee_mass: float | None = None,
        bilateral_kp: float = 0.0,
    ):
        from manimux.robots.yam.arm import YAMRobot

        self._handle = YAMRobot(
            channel=channel, arm_type=arm_type, gripper_type=gripper_type, ee_mass=ee_mass,
        )
        self._robot = self._handle.robot
        self._n = num_arm_joints
        # Remember the arm's native kp so bilateral scaling is relative to it.
        self._native_kp = np.asarray(getattr(self._robot, "_kp", np.zeros(self._n)), dtype=float)
        kp = self._native_kp * bilateral_kp if bilateral_kp > 0 else np.zeros(self._n)
        self._robot.update_kp_kd(kp=kp, kd=np.zeros(self._n))

    def close(self):
        self._handle.close()

    def get_state(self) -> tuple[np.ndarray, float, list[bool]]:
        """One same-bus read -> (arm_joints, gripper_norm, [top, second] buttons).

        Mirrors i2rt minimum_gello's ``YAMLeaderRobot.get_info``: the teaching-handle
        trigger is ``1 - position`` and ``io_inputs`` are the two button bits."""
        arm = np.asarray(self._robot.get_joint_pos(), dtype=np.float64).reshape(-1)[: self._n]
        enc = self._robot.motor_chain.get_same_bus_device_states()[0]
        gripper = float(np.clip(1.0 - enc.position, 0.0, 1.0))
        # Teaching-handle button polarity varies between units (some idle high,
        # some idle low). Learn the idle level from the first read (assumes no
        # button held during startup) and report "pressed" as deviation from it.
        raw = [bool(b) for b in enc.io_inputs]
        if not hasattr(self, "_btn_idle") or self._btn_idle is None or len(self._btn_idle) != len(raw):
            self._btn_idle = raw
        buttons = [r != i for r, i in zip(raw, self._btn_idle)]
        return arm, gripper, buttons

    def command_arm(self, arm_joints: np.ndarray) -> None:
        """Command the leader arm joints (bilateral force feedback only)."""
        self._robot.command_joint_pos(np.asarray(arm_joints, dtype=np.float64).reshape(-1))


class YamLeaderPolicy(TeleopAgent):
    """Identity map the lead arm onto the follower and apply it each tick.

    Leader and follower are the same YAM geometry, so the mapping is 1:1.
    """

    _GRIPPER_PRESS_THRESHOLD = 0.35
    _GRIPPER_RELEASE_THRESHOLD = 0.65

    def __init__(
        self,
        leader: YamLeaderArm,
        follower: RobotInterface,
        bilateral_kp: float = 0.0,
        gripper_mode: str = "analog",
        control_hz: float = 30.0,
        gripper_close_duration_s: float = 0.0,
    ):
        if gripper_mode not in {"analog", "toggle"}:
            raise ValueError("gripper_mode must be 'analog' or 'toggle'")
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if gripper_close_duration_s < 0:
            raise ValueError("gripper_close_duration_s must be non-negative")
        self._leader = leader
        self._follower = follower
        self._bilateral_kp = bilateral_kp
        self._gripper_mode = gripper_mode
        self._gripper_close_step = (
            1.0
            if gripper_close_duration_s == 0
            else 1.0 / (control_hz * gripper_close_duration_s)
        )
        self._n = follower.num_dofs() - 1
        # Leader state cached by read_inputs() so act() reads the bus only once/tick.
        self._cached: tuple[np.ndarray, float] | None = None
        self._toggle_gripper_target: float | None = None
        self._toggle_gripper_command: float | None = None
        self._toggle_gripper_armed = False

    def _gripper_command(self, trigger: float) -> float:
        trigger = float(np.clip(trigger, 0.0, 1.0))
        if self._gripper_mode == "analog":
            return trigger
        if self._toggle_gripper_target is None:
            current = np.asarray(self._follower.get_joint_pos(), dtype=np.float64).reshape(-1)
            self._toggle_gripper_target = 1.0 if current[-1] >= 0.5 else 0.0
            self._toggle_gripper_command = float(np.clip(current[-1], 0.0, 1.0))
        if trigger >= self._GRIPPER_RELEASE_THRESHOLD:
            self._toggle_gripper_armed = True
        elif trigger <= self._GRIPPER_PRESS_THRESHOLD and self._toggle_gripper_armed:
            self._toggle_gripper_target = 1.0 - self._toggle_gripper_target
            self._toggle_gripper_armed = False
        if self._toggle_gripper_target < self._toggle_gripper_command:
            self._toggle_gripper_command = max(
                self._toggle_gripper_target,
                self._toggle_gripper_command - self._gripper_close_step,
            )
        else:
            self._toggle_gripper_command = self._toggle_gripper_target
        return self._toggle_gripper_command

    def _read_leader(self) -> tuple[np.ndarray, float, list[bool]]:
        arm, trigger, buttons = self._require_leader().get_state()
        self._cached = (arm, self._gripper_command(trigger))
        return arm, trigger, buttons

    def _require_leader(self):
        if self._leader is None:
            raise RuntimeError(
                "this arm's leader was released for autonomy — Reset Session and "
                "Start Teleop to rebuild it"
            )
        return self._leader

    def _target(self) -> np.ndarray:
        if self._cached is None:
            arm, gripper, _ = self._read_leader()
        else:
            arm, gripper = self._cached
        return np.concatenate([arm, [float(np.clip(gripper, 0.0, 1.0))]])

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        cmd = self._target()  # identity: follower target == leader pose
        self._follower.command_joint_pos(cmd)
        if self._bilateral_kp > 0:
            # Push the leader toward the follower's current arm pose (haptics).
            self._leader.command_arm(np.asarray(obs["joint_pos"], dtype=np.float64).reshape(-1))
        self._cached = None
        return cmd

    def read_inputs(self) -> tuple[list[bool], float] | None:
        _, gripper, buttons = self._read_leader()
        return buttons, gripper

    def leader_raw(self) -> tuple[np.ndarray, np.ndarray] | None:
        """``(raw, cal)`` leader joint angles in radians for the live readout, or None for
        a leader without a raw readout (``YamLeaderArm`` reports through i2rt, already
        calibrated)."""
        angles = getattr(self._leader, "leader_angles", None)
        return angles() if callable(angles) else None

    def engage(self, abort: Event | None = None, steps: int = 33, dt: float = 0.03) -> None:
        """Ease the follower from its current pose to the leader's pose before live
        tracking, so it never snaps. Blocks ~steps*dt s (~1 s). Aborts early when
        ``abort`` (the loop's E-STOP event) is set, so a stop interrupts the ramp
        instead of it commanding the follower all the way to the leader."""
        arm, trigger, _ = self._require_leader().get_state()
        start = np.asarray(self._follower.get_joint_pos(), dtype=np.float64).reshape(-1)
        if self._gripper_mode == "toggle":
            self._toggle_gripper_target = 1.0 if start[-1] >= 0.5 else 0.0
            self._toggle_gripper_command = float(np.clip(start[-1], 0.0, 1.0))
            self._toggle_gripper_armed = trigger >= self._GRIPPER_RELEASE_THRESHOLD
            gripper = self._toggle_gripper_target
        else:
            gripper = float(np.clip(trigger, 0.0, 1.0))
        target = np.concatenate([arm, [gripper]])
        for i in range(1, steps + 1):
            if abort is not None and abort.is_set():
                return
            alpha = i / steps
            self._follower.command_joint_pos(start * (1.0 - alpha) + target * alpha)
            time.sleep(dt)

    def release_leader(self) -> bool:
        leader, self._leader, self._cached = self._leader, None, None
        close = getattr(leader, "close", None) or getattr(leader, "stop", None)
        if close is not None:
            close()
        return True

    def stop(self) -> None:
        self._follower.stop()
        # Release the leader too (e.g. a passive GELLO's CAN reader thread/bus).
        leader_stop = getattr(self._leader, "stop", None)
        if callable(leader_stop):
            try:
                leader_stop()
            except Exception:
                pass
