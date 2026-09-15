"""Managed teleop joint drag, isolated in teleop's Python environment.

Matches project/teleop/scripts/set_state.py --state drag --tool umi:
250 Hz, K=1, D=0.3, and a 15 deg/s reference tracker. A/B/AB share one
RobotConnection. No controller is imported or connected in the runtime process.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

DEFAULT_TELEOP_ROOT = Path(__file__).resolve().parents[5] / "teleop"
_PREFIX = "MANIMUX_DRAG "


class TeleopDragProcess:
    def __init__(self, root: Path, robot_ip: str) -> None:
        self.root = root.expanduser().resolve()
        self.robot_ip = robot_ip
        self.arm = ""
        self.state = "idle"
        self.error = ""
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._output: deque[str] = deque(maxlen=12)
        self._ready = threading.Event()
        self._started_at = 0.0
        self.cleanup_failed = False

    @property
    def busy(self) -> bool:
        return self._process is not None or self.cleanup_failed

    def start(self, arm: str) -> None:
        if self.busy:
            raise RuntimeError("Exit the current drag before starting another one")
        if arm not in {"A", "B", "AB"}:
            raise ValueError("Drag arm must be A, B or AB")
        python = self.root / ".venv/bin/python3"
        for path in (
            python,
            self.root / "drivers/arm_driver.py",
            self.root / "configs/tool/umi.yaml",
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Teleop drag requires {path}")
        self._ready.clear()
        self._output.clear()
        self.error = ""
        self._process = subprocess.Popen(
            [
                str(python),
                "-u",
                str(Path(__file__).resolve()),
                "--teleop-root",
                str(self.root),
                "--arm",
                arm,
                "--tool",
                "umi",
                "--robot-ip",
                self.robot_ip,
            ],
            cwd=self.root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self.arm = arm
        self.state = "starting"
        self._started_at = time.monotonic()
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            line = line.strip()
            self._output.append(line)
            if line == _PREFIX + '{"state": "active"}':
                self._ready.set()
            elif line == _PREFIX + '{"state": "cleanup_failed"}':
                self.cleanup_failed = True

    def stop(self) -> None:
        if self._process is not None and self.state != "stopping":
            self.state = "stopping"
            # EOF also reaches the child if this service crashes. The child
            # finishes mode-switch/servo-off cleanup instead of being killed.
            assert self._process.stdin is not None
            self._process.stdin.close()

    def poll(self) -> None:
        process = self._process
        if process is None:
            return
        code = process.poll()
        if code is not None:
            assert self._reader is not None
            self._reader.join(timeout=1)
            if process.stdout is not None:
                process.stdout.close()
            if process.stdin is not None:
                process.stdin.close()
            expected = self.state == "stopping"
            self._process = None
            self.state = "idle" if expected and code == 0 else "error"
            if self.state == "error":
                self.error = "\n".join(self._output) or f"Drag exited with code {code}"
            self.arm = ""
        elif self.state == "starting":
            if self._ready.is_set():
                self.state = "active"
            elif time.monotonic() - self._started_at > 20:
                self.error = "Drag startup timed out; waiting for servo-off"
                self.stop()

    def close(self) -> None:
        self.stop()
        if self._process is not None:
            # Keep the controller reserved until teleop has left drag and
            # disabled the arms. Never fall through into another rollout.
            self._process.wait(timeout=10)
            self.poll()


def _drag_loop(
    api: Any, conn: Any, drivers: dict[str, Any], tools: dict[str, Any], stop: threading.Event
) -> None:
    """The teleop set-state joint-drag algorithm, with paired cleanup."""
    touched: list[str] = []
    cleanup_errors: list[Exception] = []
    try:
        for arm, driver in drivers.items():
            if stop.is_set():
                return
            # set-state uses this same checked reset: E-stop faults can remain
            # latched after the physical button is released. Never enter torque
            # mode until every selected arm reads clean, including paired drag.
            ok, state = api.ensure_clear(conn, driver, arm)
            if not ok or state["err"] or state["cur"] == api.STATE_ERROR:
                raise RuntimeError(
                    f"Arm {arm} controller fault remains (err_code {state['err']}). "
                    "Release the physical E-stop and resolve the controller fault, then retry."
                )
        if stop.is_set():
            return
        conn.robot.clear_set()
        for arm, tool in tools.items():
            # Track attempts before send/readback: even a partially successful
            # mode switch must leave both selected arms disabled on failure.
            touched.append(arm)
            drivers[arm].engaged = True
            conn.robot.set_state(arm=arm, state=api.STATE_TORQUE)
            conn.robot.set_impedance_type(arm=arm, type=1)
            conn.robot.set_tool(
                arm=arm, kineParams=tool.kine_params(), dynamicParams=tool.dynamic_params()
            )
            conn.robot.set_joint_kd_params(arm=arm, K=[1.0] * 7, D=[0.3] * 7)
        conn.robot.send_cmd()
        time.sleep(0.5)
        for arm, driver in drivers.items():
            state = driver.state()
            if state["cur"] != api.STATE_TORQUE or state["err"]:
                raise RuntimeError(f"Arm {arm} failed to enter torque mode")
        if stop.is_set():
            return
        conn.robot.clear_set()
        for arm in drivers:
            conn.robot.set_drag_space(arm=arm, dgType=1)
        conn.robot.send_cmd()
        time.sleep(0.2)
        feedback = conn.subscribe()
        for arm, driver in drivers.items():
            if feedback["inputs"][driver.idx]["drag_sp_type"] != 1:
                raise RuntimeError(f"Arm {arm} drag readback failed")
        print(_PREFIX + json.dumps({"state": "active"}), flush=True)
        commands = {arm: list(driver.joints()) for arm, driver in drivers.items()}
        for arm, joints in commands.items():
            if len(joints) != 7 or not all(math.isfinite(q) for q in joints):
                raise RuntimeError(f"Arm {arm} returned invalid initial joint feedback")
        last_frames: dict[str, tuple[int, float]] = {}
        while not stop.is_set():
            for arm, driver in drivers.items():
                state = driver.state()
                if state["err"] or state["cur"] != api.STATE_TORQUE:
                    raise RuntimeError(f"Arm {arm} left drag torque mode or faulted")
                now = time.monotonic()
                frame, changed = last_frames.get(arm, (state["frame"], now))
                if frame != state["frame"]:
                    changed = now
                last_frames[arm] = (state["frame"], changed)
                if now - changed > 0.1:
                    raise RuntimeError(f"Arm {arm} feedback stopped refreshing")
                measured = state["q"]
                if len(measured) != 7 or not all(math.isfinite(q) for q in measured):
                    raise RuntimeError(f"Arm {arm} returned invalid joint feedback")
                commands[arm] = [
                    c + max(-15.0 / 250.0, min(15.0 / 250.0, m - c))
                    for c, m in zip(commands[arm], measured, strict=True)
                ]
            api.send_joint_commands(conn, commands)
            stop.wait(1.0 / 250.0)
    finally:
        for arm in touched:
            try:
                conn.robot.clear_set()
                conn.robot.set_drag_space(arm=arm, dgType=0)
                conn.robot.send_cmd()
            except Exception as exc:
                cleanup_errors.append(exc)
        if touched:
            time.sleep(0.5)
        for arm in touched:
            try:
                drivers[arm].disable()
                if drivers[arm].state()["cur"] != api.STATE_DISABLED:
                    raise RuntimeError(f"Arm {arm} servo-off was not confirmed")
            except Exception as exc:
                cleanup_errors.append(exc)
        try:
            conn.close()
        except Exception as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            print(_PREFIX + json.dumps({"state": "cleanup_failed"}), flush=True)
            raise ExceptionGroup("Drag cleanup failed", cleanup_errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teleop-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B", "AB"), required=True)
    parser.add_argument("--tool", choices=("umi",), required=True)
    parser.add_argument("--robot-ip", required=True)
    args = parser.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def watch_parent() -> None:
        sys.stdin.read()
        stop.set()

    threading.Thread(target=watch_parent, daemon=True).start()
    sys.path.insert(0, str(args.teleop_root.resolve()))
    config = importlib.import_module("config")
    tool_api = importlib.import_module("drivers.tool_config")
    tools = {
        arm: tool_api.load_tool_config(args.tool, arm, str(args.teleop_root))[0] for arm in args.arm
    }
    # Validate all selected tools before opening the hardware connection.
    for tool in tools.values():
        if not all(math.isfinite(x) for x in tool.kine_params() + tool.dynamic_params()):
            raise ValueError("UMI tool parameters must be finite")
    api = importlib.import_module("drivers.arm_driver")
    if stop.is_set():
        return
    conn = api.RobotConnection(args.robot_ip)
    try:
        drivers = {arm: api.ArmDriver(conn, arm, config) for arm in args.arm}
    except BaseException:
        conn.close()
        raise
    _drag_loop(api, conn, drivers, tools, stop)


if __name__ == "__main__":
    main()
