"""Marvin controller session and per-arm SDK kinematics.

Mirrors tianji-control's ``drivers/arm_driver.py`` (``RobotConnection``,
``ArmDriver``, ``send_joint_commands``) and ``algos/ik_solver.py``
(``ArmIK`` construction), which CalibWrist's real-robot path calls. Joint values
stay in the SDK's degrees here; the driver converts at its boundary. SDK calls
share one lock because the binding shares a single state buffer.
"""

from __future__ import annotations

import contextlib
import io
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger("manimux.robots.tianji")

FloatArray = NDArray[np.float64]

# cur_state values (vendor python_doc_contrl.md).
STATE_DISABLED = 0
STATE_POSITION = 1
STATE_ERROR = 100

ARM_INDEX = {"A": 0, "B": 1}

# J6/J7 self-interference table measured on this hardware (ik_solver.BD67_REAL).
BD67_REAL = [
    [0.0, -1.025, 110.5],
    [0.0, 1.025, 110.5],
    [0.0, -1.025, -110.5],
    [0.0, 1.025, -110.5],
]

# Controller err_code names from the vendor README.
ERROR_NAMES = {
    1: "bus topology fault",
    2: "servo fault",
    3: "PVT fault",
    4: "position-mode request failed",
    5: "entering position mode failed",
    6: "torque-mode request failed",
    7: "entering torque mode failed",
    8: "servo-on request failed",
    9: "servo on failed",
    10: "servo-off request failed",
    11: "servo off failed",
    12: "internal error",
    13: "emergency stop",
    14: "floating base configured without IMU",
    15: "PDO not working",
}


def describe_error(code: int) -> str:
    return f"err_code {code} ({ERROR_NAMES.get(code, 'unknown')})"


@dataclass(frozen=True, slots=True)
class ArmFeedback:
    cur_state: int
    err_code: int
    frame_serial: int
    joints_deg: FloatArray
    vel_ratio: int
    acc_ratio: int


class MarvinArmModel:
    """One arm's SDK kinematics and limits, initialised as ``ArmIK`` does."""

    def __init__(self, fx_kine: ModuleType, arm_index: int, config_path: Path) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            kine = fx_kine.Marvin_Kine()
        kine.log_switch(0)
        cfg = kine.load_config(arm_type=arm_index, config_path=str(config_path))
        if not cfg:
            raise RuntimeError(f"load_config failed: {config_path}")
        if not kine.initial_kine(
            robot_type=cfg["TYPE"][arm_index],
            dh=cfg["DH"][arm_index],
            pnva=cfg["PNVA"][arm_index],
            j67=BD67_REAL,
        ):
            raise RuntimeError("initial_kine failed")
        pnva = cfg["PNVA"][arm_index]
        self.lim_n = np.array([row[1] for row in pnva], dtype=np.float64)
        self.lim_p = np.array([row[0] for row in pnva], dtype=np.float64)
        self.vmax = np.array([row[2] for row in pnva], dtype=np.float64)
        self._kine = kine

    def fk_mm(self, joints_deg: Sequence[float] | FloatArray) -> FloatArray:
        """Flange pose from the vendor library, millimetres."""

        return np.asarray(
            self._kine.fk(joints=[float(value) for value in joints_deg]), dtype=np.float64
        )


class MarvinSession:
    def __init__(self, fx_robot: ModuleType, ip: str) -> None:
        self._fx = fx_robot
        self.ip = ip
        self.version: int | None = None
        self._robot: Any | None = None
        self._dcss: Any | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        if self._robot is not None:
            return
        with contextlib.redirect_stdout(io.StringIO()):
            robot = self._fx.Marvin_Robot()
        if not robot.connect(self.ip):
            raise RuntimeError(
                f"cannot link to the Marvin controller at {self.ip}: the port is taken. "
                "Is MarvinPlatform still open? Disconnect does not release the port; "
                "quit the application. Check: ss -lunp | grep 4730"
            )
        self._robot = robot
        self._dcss = self._fx.DCSS()
        try:
            robot.log_switch("0")
            robot.local_log_switch("0")
            # VERSION reads 0 right after a controller restart.
            _, version = robot.get_param("int", "VERSION")
            if not version:
                raise RuntimeError("the Marvin controller has not finished booting (VERSION 0)")
            self.version = int(version)
            # Confirm the UDP data channel is actually flowing, not just connected.
            frames, last = 0, None
            for _ in range(5):
                serial = self.read()[0].frame_serial
                if serial and serial != last:
                    frames, last = frames + 1, serial
                time.sleep(0.01)
            if frames < 3:
                raise RuntimeError(
                    f"realtime frames are not refreshing (changed {frames} times in 5 samples): "
                    "the controller is still starting, or a firewall blocks UDP"
                )
        except BaseException:
            self.close()
            raise

    def _require(self) -> Any:
        if self._robot is None:
            raise RuntimeError("Marvin session is not connected")
        return self._robot

    def read(self) -> tuple[ArmFeedback, ArmFeedback]:
        robot = self._require()
        with self._lock:
            data = robot.subscribe(self._dcss)
        feedback = []
        for index in (0, 1):
            state = data["states"][index]
            output = data["outputs"][index]
            command = data["inputs"][index]
            feedback.append(
                ArmFeedback(
                    cur_state=int(state["cur_state"]),
                    err_code=int(state["err_code"]),
                    frame_serial=int(output["frame_serial"]),
                    joints_deg=np.asarray(output["fb_joint_pos"], dtype=np.float64),
                    vel_ratio=int(command["joint_vel_ratio"]),
                    acc_ratio=int(command["joint_acc_ratio"]),
                )
            )
        return feedback[0], feedback[1]

    def prepare_position_mode(self, arm: str, vel_ratio: int, acc_ratio: int) -> None:
        """``ArmDriver.prepare`` for position mode."""

        robot = self._require()
        index = ARM_INDEX[arm]
        feedback = self.read()[index]
        if feedback.err_code or feedback.cur_state == STATE_ERROR:
            raise RuntimeError(
                f"arm {arm} has an error (state {feedback.cur_state}, "
                f"{describe_error(feedback.err_code)}); clear it first"
            )
        # The vel/acc ratio must be set before the mode switch.
        with self._lock:
            robot.clear_set()
            robot.set_vel_acc(arm=arm, velRatio=int(vel_ratio), AccRatio=int(acc_ratio))
            robot.send_cmd()
        time.sleep(0.2)
        self._check_vel_ratio(arm, vel_ratio, "after set_vel_acc")
        with self._lock:
            robot.clear_set()
            robot.set_state(arm=arm, state=STATE_POSITION)
            robot.send_cmd()
        time.sleep(1.0)  # servo response time; do not shorten
        feedback = self.read()[index]
        if feedback.cur_state != STATE_POSITION:
            raise RuntimeError(
                f"arm {arm} mode switch failed: expected {STATE_POSITION}, got {feedback.cur_state}"
            )
        self._check_vel_ratio(arm, vel_ratio, "after set_state")

    def _check_vel_ratio(self, arm: str, expected: int, when: str) -> None:
        feedback = self.read()[ARM_INDEX[arm]]
        live, acc = feedback.vel_ratio, feedback.acc_ratio
        if not 1 <= live <= 100:
            log.warning(
                "arm %s speed ratio readback unavailable (%s read vel=%r acc=%r); the "
                "controller speed behind the %d%% limits is unverified",
                arm,
                when,
                live,
                acc,
                expected,
            )
            return
        if live != int(expected):
            raise RuntimeError(
                f"arm {arm} speed ratio mismatch ({when}): sent {expected}%, read back "
                f"{live}%. The command limits assume {expected}%; do not run the robot."
            )
        log.info("arm %s speed ratio confirmed (%s): vel %d%% / acc %d%%", arm, when, live, acc)

    def send_joints(self, joints_deg: Mapping[str, Sequence[float] | FloatArray]) -> None:
        """``send_joint_commands``: several arms inside one clear_set/send_cmd pair."""

        if not joints_deg:
            return
        robot = self._require()
        with self._lock:
            robot.clear_set()
            for arm, joints in joints_deg.items():
                robot.set_joint_cmd_pose(arm=arm, joints=[float(value) for value in joints])
            robot.send_cmd()

    def stop_running(self, arm: str) -> None:
        robot = self._require()
        with self._lock:
            robot.stop_running(arm)

    def disable(self, arm: str) -> bool:
        """``ArmDriver.disable``: servo off, confirmed within three attempts.

        Returns False only when the controller kept reporting another state; a
        read that fails (connection already gone) cannot confirm and ends the
        attempt without raising, as in the original.
        """

        robot = self._require()
        for _ in range(3):
            with self._lock:
                robot.clear_set()
                robot.set_state(arm=arm, state=STATE_DISABLED)
                robot.send_cmd()
            time.sleep(0.3)
            try:
                if self.read()[ARM_INDEX[arm]].cur_state == STATE_DISABLED:
                    return True
            except Exception:  # noqa: BLE001 - cannot confirm and must not raise here
                return True
        return False

    def close(self) -> None:
        robot, self._robot = self._robot, None
        if robot is not None:
            with self._lock:
                robot.release_robot()
