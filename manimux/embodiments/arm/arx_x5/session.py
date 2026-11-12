"""Own the official X5 binding in a process with an explicit resource lifetime."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal

import numpy as np

from manimux.types import FloatArray

SessionOperation = Literal["state", "send", "stop", "close"]
SDKSample = tuple[list[float], int, int]
JointGripperTarget = tuple[list[float], float]

_PROTECT_MODE = 2
_POSITION_CONTROL_MODE = 5


def _accepted(value: bool, operation: str) -> None:
    if value is not True:
        raise RuntimeError(f"ARX SDK rejected {operation}")


class VendorX5Session:
    """X5 (2023) only; never instantiate the upstream active SingleArm wrapper."""

    def __init__(self, channel: str, urdf_path: Path, control_parameters: Sequence[float]) -> None:
        import arx_x5_python

        self.native = arx_x5_python.InterfacesPy(str(urdf_path), channel, 0)
        # Set PROTECT before starting the vendor control loop. GO_HOME is mode 1
        # and is never selected by this adapter, including during cleanup.
        _accepted(self.native.set_arm_status(_PROTECT_MODE), "protect mode")
        self.native.arx_x(*control_parameters)

    def sample(self) -> FloatArray:
        q = np.array(self.native.get_joint_positions(), dtype=float, copy=True)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError("ARX SDK must return six measured joints and a gripper encoder")
        return q

    def send(self, joints: list[float], gripper: float) -> None:
        _accepted(self.native.set_joint_positions(joints), "joint target")
        _accepted(self.native.set_catch(gripper), "gripper target")
        _accepted(self.native.set_arm_status(_POSITION_CONTROL_MODE), "position mode")

    def stop(self) -> None:
        _accepted(self.native.set_arm_status(_PROTECT_MODE), "protect mode")


def run_session(
    connection: Connection,
    channel: str,
    urdf_path: Path,
    control_parameters: Sequence[float],
    poll_interval_s: float,
) -> None:
    """RPC handles carry copied data; the child owns every native SDK resource.

    The SDK exposes no motor receipt timestamp. A sample records completion of
    an SDK read, not a proven new device measurement. Cached RPC reads retain it.
    os._exit releases process resources without invoking an opaque destructor.
    """
    session = None
    try:
        session = VendorX5Session(channel, urdf_path, control_parameters)
        sample: SDKSample = (session.sample().tolist(), time.monotonic_ns(), 1)
        connection.send({"ok": True, "value": None})
        sequence = 1
        next_sample = time.monotonic() + poll_interval_s
        sample_error: str | None = None
        while True:
            if connection.poll(max(0.0, next_sample - time.monotonic())):
                operation, payload = connection.recv()
                try:
                    value = None
                    if operation == "state":
                        if sample_error is not None:
                            raise RuntimeError(sample_error)
                        value = sample
                    elif operation == "send":
                        if sample_error is not None:
                            raise RuntimeError(sample_error)
                        session.send(*payload)
                    elif operation == "stop":
                        session.stop()
                    elif operation == "close":
                        session.stop()
                        connection.send({"ok": True, "value": None})
                        connection.close()
                        os._exit(0)
                    else:
                        raise ValueError(f"unknown X5 session operation: {operation}")
                    connection.send({"ok": True, "value": value})
                except Exception as error:
                    connection.send({"ok": False, "error": f"{type(error).__name__}: {error}"})
            if time.monotonic() >= next_sample:
                if sample_error is None:
                    try:
                        joints = session.sample()
                        sequence += 1
                        sample = (joints.tolist(), time.monotonic_ns(), sequence)
                    except Exception as error:
                        sample_error = f"ARX polling failed: {type(error).__name__}: {error}"
                        try:
                            session.stop()
                        except Exception as stop_error:
                            sample_error += f"; protect failed: {stop_error}"
                next_sample = time.monotonic() + poll_interval_s
    except Exception as error:
        try:
            if session is not None:
                session.stop()
            connection.send({"ok": False, "error": f"{type(error).__name__}: {error}"})
        finally:
            connection.close()
            os._exit(1)
