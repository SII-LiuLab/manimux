"""YAM RobotDriver plugin for ManiMux's canonical grouped commands."""

from __future__ import annotations

import contextlib
import logging
import signal
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from manimux.clock import Clock
from manimux.config import RobotConfig
from manimux.types import RobotCommand, RobotState

log = logging.getLogger("manimux.robots.yam")


@contextlib.contextmanager
def _finish_move_before_interrupt(what: str) -> Iterator[list[int]]:
    """Let a blocking arm move finish before Ctrl-C takes effect.

    i2rt drives ``move_joints`` with ``time.sleep`` in a loop, so a SIGINT lands
    as ``KeyboardInterrupt`` inside that sleep: the move stops with the arm
    mid-air and the ``close()`` that follows zeroes torques, dropping it. Defer
    the first interrupt until the move returns. A second Ctrl-C restores the
    default handler and aborts immediately, so a genuinely stuck move is still
    escapable.
    """
    if threading.current_thread() is not threading.main_thread():
        yield []
        return

    pending: list[int] = []

    def _handler(signum: int, frame: Any) -> None:
        if pending:
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        pending.append(signum)
        log.warning(
            "Ctrl-C received; finishing %s first (press Ctrl-C again to abort now).",
            what,
        )

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handler)
    try:
        yield pending
    finally:
        with contextlib.suppress(ValueError, TypeError):
            signal.signal(signal.SIGINT, previous)


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAM config must be a mapping: {path}")
    return value


class YamDualArmDriver:
    """Map ManiMux joint-position groups onto the existing i2rt YAM wrapper."""

    GROUP_ORDER = ("left_arm", "right_arm")

    #: Every key ``robot.options`` may carry. The mapping is free-form, so an
    #: unlisted key -- a typo, or one left over from a retired feature -- would
    #: otherwise be ignored in silence while the arms still move.
    OPTIONS = frozenset(
        {
            "right_config",
            "start_duration_s",
            "start_joints",
            "home_duration_s",
            "home_gripper_release_duration_s",
            "move_to_start_on_connect",
            "home_on_close",
        }
    )

    def __init__(self, config: RobotConfig, clock: Clock) -> None:
        if config.config is None:
            raise ValueError("yam_dual requires robot.config for the left arm")
        if tuple(config.group_dims) != self.GROUP_ORDER or any(
            config.group_dims[name] != 7 for name in self.GROUP_ORDER
        ):
            raise ValueError("yam_dual requires left_arm and right_arm groups of dimension 7")
        right_config = config.options.get("right_config")
        if not isinstance(right_config, str) or not right_config:
            raise ValueError("yam_dual requires robot.options.right_config")
        self._left_path = config.config
        self._right_path = Path(right_config)
        self._clock = clock
        self._home_duration_s = float(config.options.get("home_duration_s", 5.0))
        self._home_gripper_release_duration_s = float(
            config.options.get("home_gripper_release_duration_s", 1.0)
        )
        self._start_duration_s = float(config.options.get("start_duration_s", 5.0))
        start_joints = config.options.get("start_joints")
        self._start_joints = (
            None if start_joints is None else np.asarray(start_joints, dtype=np.float64)
        )
        self._move_to_start_on_connect = bool(config.options.get("move_to_start_on_connect", False))
        unknown = sorted(set(config.options) - self.OPTIONS)
        if unknown:
            raise ValueError(
                f"unknown robot.options for yam_dual: {', '.join(unknown)}; "
                f"known keys are {', '.join(sorted(self.OPTIONS))}"
            )
        if (
            self._home_duration_s <= 0
            or self._home_gripper_release_duration_s <= 0
            or self._start_duration_s <= 0
        ):
            raise ValueError("YAM move durations must be positive")
        self._robot: Any | None = None
        self._sequence = 0

    def connect(self) -> None:
        left_cfg = _load_mapping(self._left_path)
        right_cfg = _load_mapping(self._right_path)
        left_start = np.asarray(left_cfg.get("agent", {}).get("start_joints"), dtype=np.float64)
        right_start = np.asarray(right_cfg.get("agent", {}).get("start_joints"), dtype=np.float64)
        if self._start_joints is not None:
            if self._start_joints.shape != (14,) or not np.isfinite(self._start_joints).all():
                raise ValueError("robot.options.start_joints must contain 14 finite values")
            left_start = self._start_joints[:7]
            right_start = self._start_joints[7:]
        if left_start.shape != (7,) or right_start.shape != (7,):
            raise ValueError("YAM requires 7-value agent.start_joints in both configs")
        if not np.isfinite(left_start).all() or not np.isfinite(right_start).all():
            raise ValueError("YAM start joints must be finite")
        if self._robot is not None:
            return

        from manimux.robots.yam.arm import YAMRobot
        from manimux.robots.yam.base import BimanualRobot

        left_channel = left_cfg.get("robot", {}).get("channel")
        right_channel = right_cfg.get("robot", {}).get("channel")
        if not isinstance(left_channel, str) or not isinstance(right_channel, str):
            raise ValueError("both YAM configs must define robot.channel")
        left_robot = None
        right_robot = None
        try:
            left_robot = YAMRobot(channel=left_channel)
            right_robot = YAMRobot(channel=right_channel)
            self._robot = BimanualRobot(left_robot, right_robot)
            if self._move_to_start_on_connect:
                self._move_joints(
                    np.concatenate([left_start, right_start]),
                    duration_s=self._start_duration_s,
                    transition="start position",
                    parallel=True,
                )
        except BaseException as primary_error:
            cleanup_error: BaseException | None = None
            if self._robot is not None:
                try:
                    self.close()
                except BaseException as exc:
                    cleanup_error = exc
            else:
                for arm in (right_robot, left_robot):
                    if arm is not None:
                        try:
                            arm.close()
                        except BaseException as exc:
                            if cleanup_error is None:
                                cleanup_error = exc
            if cleanup_error is not None:
                raise BaseExceptionGroup(
                    "YAM connection failed and cleanup was incomplete",
                    [primary_error, cleanup_error],
                ) from None
            raise

    def _require_robot(self) -> Any:
        if self._robot is None:
            raise RuntimeError("YAM dual-arm driver is not connected")
        return self._robot

    def get_state(self) -> RobotState:
        joints = np.asarray(self._require_robot().get_joint_state(), dtype=np.float64)
        if joints.shape != (14,) or not np.isfinite(joints).all():
            raise RuntimeError(f"YAM returned invalid joint state shape {joints.shape}")
        self._sequence += 1
        return RobotState(
            groups={"left_arm": joints[:7], "right_arm": joints[7:]},
            monotonic_ns=self._clock.now_ns(),
            sequence=self._sequence,
        )

    def send_command(self, command: RobotCommand) -> None:
        joints = np.concatenate([command.groups[name] for name in self.GROUP_ORDER])
        if joints.shape != (14,) or not np.isfinite(joints).all():
            raise ValueError("YAM command must contain two finite 7-value groups")
        self._require_robot().command_joint_state(joints)

    def home(self) -> None:
        achieved = np.asarray(self._require_robot().get_joint_state(), dtype=np.float64)
        if achieved.shape != (14,) or not np.isfinite(achieved).all():
            raise RuntimeError("YAM returned invalid joint state before homing")

        # The native YAM command is 7-D per arm, so an arm-only home command is
        # represented by six zero arm joints plus the currently achieved gripper
        # position.  Keeping the gripper unchanged prevents a grasped object from
        # being released while the arms are still returning home.
        arm_home = np.zeros(14, dtype=np.float64)
        arm_home[6] = achieved[6]
        arm_home[13] = achieved[13]
        self._move_joints(
            arm_home,
            duration_s=self._home_duration_s,
            transition="zero home",
            parallel=True,
            reraise_interrupt=False,
        )

        # Only after both arms have reached home do we release both grippers.
        released_home = arm_home.copy()
        released_home[[6, 13]] = 1.0
        self._move_joints(
            released_home,
            duration_s=self._home_gripper_release_duration_s,
            transition="home gripper release",
            parallel=True,
            reraise_interrupt=False,
        )

    def _move_joints(
        self,
        target: np.ndarray,
        *,
        duration_s: float,
        transition: str,
        parallel: bool,
        reraise_interrupt: bool = True,
    ) -> None:
        robot = self._require_robot()
        if target.shape != (14,) or not np.isfinite(target).all():
            raise ValueError(f"YAM {transition} target must contain 14 finite joints")
        arm_targets = tuple(
            zip(
                (robot._robot_l, robot._robot_r),
                (target[:7], target[7:]),
                strict=True,
            )
        )

        def move_one(arm: Any, arm_target: np.ndarray) -> None:
            native = getattr(arm, "robot", None)
            move_joints = getattr(native, "move_joints", None)
            if not callable(move_joints):
                raise RuntimeError("YAM i2rt backend does not expose move_joints")
            move_joints(arm_target, time_interval_s=duration_s)

        with _finish_move_before_interrupt(transition) as interrupted:
            if parallel:
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="yam-start") as pool:
                    futures = [
                        pool.submit(move_one, arm, arm_target) for arm, arm_target in arm_targets
                    ]
                    for future in futures:
                        future.result()
            else:
                for arm, arm_target in arm_targets:
                    move_one(arm, arm_target)
        if interrupted:
            log.info("%s finished; honouring the deferred Ctrl-C.", transition)
            if reraise_interrupt:
                raise KeyboardInterrupt

    def stop(self) -> None:
        if self._robot is None:
            return
        achieved = np.asarray(self._robot.get_joint_state(), dtype=np.float64)
        if achieved.shape != (14,) or not np.isfinite(achieved).all():
            raise RuntimeError("YAM returned invalid joint state while stopping")
        self._robot.command_joint_state(achieved)

    def close(self) -> None:
        if self._robot is None:
            return
        robot = self._robot
        cleanup_error: BaseException | None = None
        for arm in (robot._robot_l, robot._robot_r):
            try:
                arm.close()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        self._robot = None
        if cleanup_error is not None:
            raise RuntimeError("YAM shutdown did not complete safely") from cleanup_error


def build_robot(config: RobotConfig, clock: Clock) -> YamDualArmDriver:
    return YamDualArmDriver(config, clock)
