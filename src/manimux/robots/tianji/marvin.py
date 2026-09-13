"""One Marvin controller session: a single UDP link shared by arms A and B.

A typed layer over the vendor ``Marvin_Robot`` binding. Joint values stay in the
SDK's degrees here; the driver converts at its boundary. All SDK calls go
through one lock because the binding shares a single state buffer.
"""

from __future__ import annotations

import contextlib
import io
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger("manimux.robots.tianji")

# cur_state values (vendor python_doc_contrl.md).
STATE_DISABLED = 0
STATE_POSITION = 1
STATE_ERROR = 100

ARM_INDEX = {"A": 0, "B": 1}

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
    joints_deg: NDArray[np.float64]
    vel_ratio: int
    acc_ratio: int


class MarvinSession:
    def __init__(self, fx_robot: ModuleType, ip: str) -> None:
        self._fx = fx_robot
        self.ip = ip
        self.version: int | None = None
        self._robot: Any | None = None
        self._dcss: Any | None = None
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._robot is not None

    def connect(self, *, ready_samples: int = 5, sample_interval_s: float = 0.01) -> None:
        if self._robot is not None:
            return
        with contextlib.redirect_stdout(io.StringIO()):
            robot = self._fx.Marvin_Robot()
        if not robot.connect(self.ip):
            raise RuntimeError(
                f"cannot link to the Marvin controller at {self.ip}: the UDP port is taken "
                "or the controller is unreachable. Quit MarvinPlatform completely "
                "(Disconnect does not free the port) and check `ss -lunp | grep 4730`."
            )
        self._robot = robot
        self._dcss = self._fx.DCSS()
        try:
            robot.log_switch("0")
            robot.local_log_switch("0")
            _, version = robot.get_param("int", "VERSION")
            if not version:
                raise RuntimeError(
                    "the Marvin controller is still booting (VERSION reads 0); retry shortly"
                )
            self.version = int(version)
            serials = set()
            for _ in range(ready_samples):
                serials.add(self.read()[0].frame_serial)
                time.sleep(sample_interval_s)
            if len(serials) < min(3, ready_samples):
                raise RuntimeError(
                    f"Marvin feedback frames are not advancing ({len(serials)} distinct in "
                    f"{ready_samples} samples): the controller is still starting or UDP is "
                    "blocked by a firewall"
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
            state, output, command = (
                data["states"][index],
                data["outputs"][index],
                data["inputs"][index],
            )
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

    def prepare_position_mode(
        self, arm: str, vel_ratio: int, acc_ratio: int, *, settle_s: float = 1.0
    ) -> None:
        """Set the speed/acceleration percentages, then enter position-follow mode."""

        robot = self._require()
        index = ARM_INDEX[arm]
        feedback = self.read()[index]
        if feedback.err_code or feedback.cur_state == STATE_ERROR:
            raise RuntimeError(
                f"arm {arm} reports {describe_error(feedback.err_code)} "
                f"(state {feedback.cur_state}); clear it on the controller first"
            )
        # The ratios must be set before the mode switch.
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
        time.sleep(settle_s)
        feedback = self.read()[index]
        if feedback.cur_state != STATE_POSITION:
            raise RuntimeError(
                f"arm {arm} did not enter position mode (state {feedback.cur_state}, "
                f"{describe_error(feedback.err_code)})"
            )
        self._check_vel_ratio(arm, vel_ratio, "after set_state")

    def _check_vel_ratio(self, arm: str, expected: int, when: str) -> None:
        live = self.read()[ARM_INDEX[arm]].vel_ratio
        if not 1 <= live <= 100:
            log.warning(
                "arm %s speed ratio readback unavailable %s (read %r); the controller speed "
                "behind the command limits is unconfirmed",
                arm,
                when,
                live,
            )
            return
        if live != int(expected):
            raise RuntimeError(
                f"arm {arm} speed ratio mismatch {when}: requested {expected}%, "
                f"controller reports {live}%"
            )

    def send_joints(self, joints_deg: Mapping[str, Sequence[float] | NDArray[np.float64]]) -> None:
        """Send position targets (degrees) for several arms in one batch."""

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

    def disable(self, arm: str, *, attempts: int = 3, wait_s: float = 0.3) -> bool:
        """Switch one arm to servo-off and return whether the controller confirmed it."""

        robot = self._require()
        for _ in range(attempts):
            with self._lock:
                robot.clear_set()
                robot.set_state(arm=arm, state=STATE_DISABLED)
                robot.send_cmd()
            time.sleep(wait_s)
            if self.read()[ARM_INDEX[arm]].cur_state == STATE_DISABLED:
                return True
        return False

    def close(self) -> None:
        robot, self._robot = self._robot, None
        if robot is not None:
            with self._lock:
                robot.release_robot()
