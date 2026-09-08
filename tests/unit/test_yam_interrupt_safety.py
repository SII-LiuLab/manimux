"""A Ctrl-C must never leave the arms released mid-air.

i2rt drives ``move_joints`` with ``time.sleep`` in a loop, so a SIGINT used to
land inside that sleep, stop the homing move half-way, and let the following
``close()`` zero the torques while the arm was still up.
"""

from __future__ import annotations

import os
import signal
import threading
import time

import numpy as np
import pytest

from manimux.clock import SystemClock
from manimux.config import load_config
from manimux.robots.yam import YamDualArmDriver


class _SleepingArm:
    """Stand-in for the i2rt backend: moving takes real time, in small sleeps."""

    def __init__(self, position: np.ndarray, steps: int = 20) -> None:
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.steps = steps
        self.completed_moves: list[np.ndarray] = []
        self.closed = False

    def move_joints(self, target: np.ndarray, time_interval_s: float) -> None:
        for _ in range(self.steps):
            time.sleep(time_interval_s / self.steps)
        self.position = np.asarray(target, dtype=np.float64).copy()
        self.completed_moves.append(self.position.copy())

    def close(self) -> None:
        self.closed = True


class _Arm:
    def __init__(self, position: np.ndarray) -> None:
        self.robot = _SleepingArm(position)

    def get_joint_state(self) -> np.ndarray:
        return self.robot.position.copy()

    def close(self) -> None:
        self.robot.close()


class _Bimanual:
    def __init__(self) -> None:
        left = np.linspace(-0.5, 0.5, 7)
        right = np.linspace(0.5, -0.5, 7)
        left[-1] = 0.25
        right[-1] = 0.75
        self._robot_l = _Arm(left)
        self._robot_r = _Arm(right)

    def get_joint_state(self) -> np.ndarray:
        return np.concatenate([self._robot_l.get_joint_state(), self._robot_r.get_joint_state()])

    def command_joint_state(self, command: np.ndarray) -> None:
        self._robot_l.robot.position = np.asarray(command[:7], dtype=np.float64).copy()
        self._robot_r.robot.position = np.asarray(command[7:], dtype=np.float64).copy()


def _live_driver() -> tuple[YamDualArmDriver, _Bimanual]:
    config = load_config("configs/molmoact2/yam/infra/manimux.yaml")
    config.robot.options["home_duration_s"] = 0.4
    config.robot.options["home_gripper_release_duration_s"] = 0.1
    config.robot.options["start_duration_s"] = 0.4
    driver = YamDualArmDriver(config.robot, SystemClock())
    backend = _Bimanual()
    driver._robot = backend
    return driver, backend


def _interrupt_after(delay_s: float, count: int = 1) -> threading.Thread:
    def _fire() -> None:
        for _ in range(count):
            time.sleep(delay_s)
            os.kill(os.getpid(), signal.SIGINT)

    thread = threading.Thread(target=_fire, daemon=True)
    thread.start()
    return thread


HOME = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_ctrl_c_during_homing_still_reaches_zero_home() -> None:
    driver, backend = _live_driver()
    original = signal.getsignal(signal.SIGINT)

    thread = _interrupt_after(0.1)
    try:
        driver.home()  # must NOT raise: homing is the last cleanup step
    finally:
        thread.join(timeout=2.0)
        signal.signal(signal.SIGINT, original)

    for arm in (backend._robot_l, backend._robot_r):
        assert len(arm.robot.completed_moves) == 2, "the two-stage home did not complete"
        np.testing.assert_allclose(arm.robot.position, HOME)
    np.testing.assert_allclose(
        backend._robot_l.robot.completed_moves[0],
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]),
    )
    np.testing.assert_allclose(
        backend._robot_r.robot.completed_moves[0],
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75]),
    )
    assert signal.getsignal(signal.SIGINT) is original


def test_ctrl_c_during_start_move_finishes_then_propagates() -> None:
    driver, backend = _live_driver()
    original = signal.getsignal(signal.SIGINT)

    thread = _interrupt_after(0.1)
    try:
        with pytest.raises(KeyboardInterrupt):
            driver._move_joints(
                np.concatenate([HOME, HOME]),
                duration_s=0.4,
                transition="start pose",
                parallel=True,
            )
    finally:
        thread.join(timeout=2.0)
        signal.signal(signal.SIGINT, original)

    # The abort is honoured, but only after both arms actually arrived.
    for arm in (backend._robot_l, backend._robot_r):
        assert arm.robot.completed_moves
        np.testing.assert_allclose(arm.robot.position, HOME)


def test_second_ctrl_c_aborts_a_stuck_move() -> None:
    driver, backend = _live_driver()
    backend._robot_l.robot.steps = 400
    backend._robot_r.robot.steps = 400
    original = signal.getsignal(signal.SIGINT)

    thread = _interrupt_after(0.15, count=2)
    try:
        with pytest.raises(KeyboardInterrupt):
            driver._move_joints(
                np.concatenate([HOME, HOME]),
                duration_s=8.0,
                transition="zero home",
                parallel=False,
                reraise_interrupt=False,
            )
    finally:
        thread.join(timeout=3.0)
        signal.signal(signal.SIGINT, original)

    assert not backend._robot_l.robot.completed_moves, "a second Ctrl-C must abort immediately"
