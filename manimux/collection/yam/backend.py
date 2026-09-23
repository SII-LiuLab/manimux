"""Original collection GUI protocol backed by ManiMux robot control."""

from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from json import dumps, loads
from pathlib import Path

import numpy as np

from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.collection.yam.data.command_rates import CommandRates
from manimux.collection.yam.data.control_timing import arm_scope, stage
from manimux.embodiments.robot import build_robot
from manimux.runtime.executors import DirectExecutor, MPCExecutor, SmoothExecutor
from manimux.runtime.lock import RuntimeInstanceLock
from manimux.runtime.safety import SafetyGuard
from manimux.timing import timed_lock
from manimux.types import ActionHorizon, RobotCommand, copy_group_vector


class CollectionBackend:
    def __init__(
        self,
        config,
        *,
        driver=None,
        lock_dir=None,
        config_path=None,
        execution_mode="synchronous",
        collection_hz,
    ):
        if execution_mode not in {"synchronous", "threaded"}:
            raise ValueError("execution_mode must be synchronous or threaded")
        self.execution_mode = execution_mode
        self.config = config
        self.clock = SystemClock()
        self.dt = 1.0 / config["robot"]["control_hz"]
        if (
            isinstance(collection_hz, bool)
            or not math.isfinite(collection_hz)
            or collection_hz <= 0
        ):
            raise ValueError("collection_hz must be finite and positive")
        self._target_interval_s = 1.0 / float(collection_hz)
        execution = config["executor"]
        if execution["type"] == "direct":
            self.executor = DirectExecutor(execution["motion_limits"], self.dt)
            position_limit = None
        elif execution["type"] == "smooth":
            self.executor = SmoothExecutor(execution["smooth"], self.dt)
            position_limit = execution["smooth"]["position_limit_abs"]
        else:
            if execution["motion_limits"] is not None:
                raise ValueError(
                    "shared motion_limits currently support direct and smooth, not mpc"
                )
            self.executor = MPCExecutor(execution["mpc"], self.dt)
            position_limit = execution["mpc"]["position_limit_abs"]
        limits = execution["command_safety"]
        self.safety = SafetyGuard(
            config["robot"]["group_dims"],
            position_limit,
            position_lower=limits["position_lower"],
            position_upper=limits["position_upper"],
            max_velocity=limits["max_velocity"],
            max_acceleration=limits["max_acceleration"],
            control_dt_s=self.dt,
        )
        robot_config = deepcopy(config["robot"])
        self.driver = driver or build_robot(robot_config, self.clock)
        self.lease = RuntimeInstanceLock(
            "yam",
            mode="collection",
            config_path=Path(config_path or "manimux/configs/collection/yam/control.yaml"),
            lock_dir=lock_dir,
        )
        self._mutex = threading.RLock()
        self._quit = threading.Event()
        self._thread = None
        self._connected = False
        self._fault = None
        self._halted = False
        self._enabled = False
        self._batch = None
        self._target = None
        self._target_ns = 0
        self._sequence = 0
        self._last_command = None
        self._command_unix_ns = 0
        self._trace = None
        self.command_rates = CommandRates()

    def connect(self, *, start_thread=True):
        self.lease.__enter__()
        try:
            self.driver.connect()
            self._state = self.driver.get_state()
            self.safety.reset(self._state)
            self.executor.reset(self._state)
            self._target = copy_group_vector(self._state.groups)
            self._connected = True
            if start_thread and self.execution_mode == "threaded":
                self._thread = threading.Thread(
                    target=self._run, name="manimux-collection-executor", daemon=True
                )
                self._thread.start()
        except BaseException:
            try:
                self.driver.close()
            finally:
                self.lease.__exit__(None, None, None)
            raise

    def _check(self):
        if self._fault:
            raise RuntimeError(f"ManiMux collection executor failed: {self._fault}")
        if not self._connected:
            raise RuntimeError("collection backend is closed")

    @property
    def leader_timeout_s(self) -> float:
        """Allow one scheduled target period plus margin, with the configured timeout as a floor."""

        return max(
            self.config["policy"]["timeout_s"],
            2 * self._target_interval_s,
        )

    def _target_timed_out(self, now_ns: int) -> bool:
        return now_ns - self._target_ns > int(self.leader_timeout_s * 1e9)

    def set_control_hz(self, hz: float) -> None:
        """Update robot execution timing without changing collection timing."""

        if isinstance(hz, bool) or not math.isfinite(hz) or hz <= 0:
            raise ValueError("robot control_hz must be finite and positive")
        with self._mutex:
            self._check()
            if (
                self.execution_mode != "synchronous"
                or self.config["policy"]["worker"] != "local_yam_leader"
                or not isinstance(self.executor, (DirectExecutor, SmoothExecutor))
            ):
                raise RuntimeError(
                    "online frequency changes require synchronous Direct/Smooth teleop"
                )
            if self._batch is not None or self._trace is not None:
                raise RuntimeError("cannot change frequency during a target batch or recording")
            if self._enabled and self._target_timed_out(self.clock.now_ns()):
                raise RuntimeError("leader target timed out; cannot apply frequency change")
            dt = 1.0 / hz
            self.executor.set_control_period(dt)
            self.safety.set_control_period(dt)
            self.dt = dt
            self.config["robot"]["control_hz"] = hz

    def set_collection_hz(self, hz: float) -> None:
        """Update collection target timing without changing robot execution timing."""

        if isinstance(hz, bool) or not math.isfinite(hz) or hz <= 0:
            raise ValueError("collection_hz must be finite and positive")
        with self._mutex:
            self._check()
            if self._batch is not None or self._trace is not None:
                raise RuntimeError("cannot change collection frequency during a batch or recording")
            self._target_interval_s = 1.0 / hz

    @contextmanager
    def target_batch(self):
        with timed_lock(self._mutex, "backend.target_lock_wait"):
            self._check()
            self._batch = copy_group_vector(self._target)
            try:
                yield
                self._publish(self._batch)
            finally:
                self._batch = None

    def _publish(self, groups):
        if self._halted:
            raise RuntimeError("collection is stopped; Reset Session before resuming")
        if self._enabled and self._target_timed_out(self.clock.now_ns()):
            error = RuntimeError("leader target timed out")
            self._stop_on_error(error)
            raise error
        if set(groups) != set(self.config["robot"]["group_dims"]):
            raise ValueError("leader groups do not match the robot")
        command = RobotCommand(groups, self.clock.now_ns(), f"yam-leader-{self._sequence + 1}")
        for name, value in command.groups.items():
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError(f"invalid leader target for {name}")
            if not 0 <= value[-1] <= 1:
                raise ValueError(f"invalid leader gripper for {name}")
        if not self._enabled:
            with arm_scope("initial"), stage("state_read"):
                self._state = self.driver.get_state()
            self.executor.reset(self._state)
            self.safety.reset(self._state)
        self._target = copy_group_vector(command.groups)
        self._target_ns = command.monotonic_ns
        self._sequence += 1
        self._enabled = True
        if self.execution_mode == "synchronous":
            try:
                self.tick()
            except Exception as exc:
                self._stop_on_error(exc)
                raise

    def set_target(self, group, target):
        with self._mutex:
            self._check()
            target = np.asarray(target, dtype=np.float64).copy()
            if self._batch is not None:
                self._batch[group] = target
            else:
                groups = copy_group_vector(self._target)
                groups[group] = target
                self._publish(groups)

    def tick(self):
        with self._mutex:
            self._check()
            with stage("backend.state_read_validate"), arm_scope("precommand"):
                with stage("state_read"):
                    self._state = self.driver.get_state()
                with stage("state_validate"):
                    self.safety.validate_state(self._state)
            if not self._enabled:
                return
            now = self.clock.now_ns()
            if self._target_timed_out(now):
                raise RuntimeError("leader target timed out")
            with stage("backend.executor"):
                reference = ActionHorizon(
                    now,
                    int(self.dt * 1e9),
                    f"yam-leader-{self._sequence}",
                    {
                        name: np.tile(value, (self.executor.horizon_steps, 1))
                        for name, value in self._target.items()
                    },
                    observation_time_ns=self._target_ns,
                )
                command = self.executor.step(now, self._state, reference)
            with stage("backend.command_validate"):
                self.safety.validate_command(command)
            with stage(
                "backend.follower_sdk_submit",
                sequence=self._sequence,
                source_monotonic_ns=self._target_ns,
            ), arm_scope("submit"):
                self.driver.send_command(command)
            self.command_rates.record(self._sequence)
            self._last_command = command
            self._command_unix_ns = time.time_ns()
            if self._trace is not None:
                with stage("backend.trace_buffer"):
                    self._trace.append(
                        {
                            "monotonic_ns": now,
                            "unix_ns": self._command_unix_ns,
                            "source_monotonic_ns": self._target_ns,
                            "sequence": self._sequence,
                            "source": {
                                name: value.tolist() for name, value in self._target.items()
                            },
                            "command": {
                                name: value.tolist() for name, value in command.groups.items()
                            },
                            "feedback": {
                                name: value.tolist() for name, value in self._state.groups.items()
                            },
                        }
                    )

    def _stop_on_error(self, error):
        with self._mutex:
            self._fault = str(error)
            self._halted = True
            self._enabled = False
            try:
                self.driver.stop()
            except Exception as stop_exc:
                self._fault += f"; stop failed: {stop_exc}"

    def _run(self):
        while not self._quit.is_set():
            started = time.monotonic()
            try:
                self.tick()
            except Exception as exc:
                self._stop_on_error(exc)
                return
            self._quit.wait(max(0.0, self.dt - (time.monotonic() - started)))

    def observation(self, group):
        with arm_scope(f"observation_{group}"), timed_lock(self._mutex, "lock_wait"):
            self._check()
            if self.execution_mode == "synchronous":
                with stage("state_read"):
                    self._state = self.driver.get_state()
                with stage("state_validate"):
                    self.safety.validate_state(self._state)
            values = self._state.groups[group].copy()
            timestamp = time.time_ns() - (self.clock.now_ns() - self._state.monotonic_ns)
            return {
                "joint_pos": values[:6],
                "gripper_pos": values[6:],
                "feedback_timestamp_ns": np.int64(timestamp),
            }

    def controller_input(self, group):
        with self._mutex:
            if self._last_command is None:
                return None
            return {
                "joint_pos": self._last_command.groups[group][:6].copy(),
                "timestamp_ns": np.int64(self._command_unix_ns),
            }

    def pause(self, *, halt=False):
        with self._mutex:
            if not self._connected:
                return
            self._enabled = False
            self._halted = self._halted or halt
            if self._fault:
                return
            with arm_scope("pause"), stage("state_read"):
                self._state = self.driver.get_state()
            hold = RobotCommand(
                copy_group_vector(self._state.groups), self.clock.now_ns(), "collection-hold"
            )
            with stage("backend.hold_submit"), arm_scope("hold"):
                self.driver.send_command(hold)
            self._target = copy_group_vector(self._state.groups)
            self.executor.reset(self._state)
            self.safety.reset(self._state)

    def start_trace(self):
        with self._mutex:
            self._trace = []

    def finish_trace(self):
        with self._mutex:
            result, self._trace = self._trace or [], None
            return result

    def metadata(self):
        return {
            "backend": "manimux",
            "policy": "yam_leader",
            "execution_mode": self.execution_mode,
            "control_profile": (
                str(self.config["control_profile"]) if self.config["control_profile"] else None
            ),
            "policy_config": loads(dumps(deepcopy(self.config["policy"]), default=str)),
            "leader_timeout_s": self.leader_timeout_s,
            "robot": loads(dumps(deepcopy(self.config["robot"]), default=str)),
            "executor": loads(dumps(deepcopy(self.config["executor"]), default=str)),
            "controller_tracking_scope": "executor output before driver joint-limit clipping",
            "trace_file": "manimux-control.jsonl",
        }

    def close(self):
        self._quit.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                raise RuntimeError("collection executor did not stop; ownership retained")
        with self._mutex:
            if not self._connected:
                return
            self.driver.close()
            self._connected = False
            self.lease.__exit__(None, None, None)


class FollowerView:
    def __init__(self, backend, group):
        self.backend = backend
        self.group = group

    def num_dofs(self):
        return 7

    def get_joint_pos(self):
        observation = self.get_observations()
        return np.concatenate([observation["joint_pos"], observation["gripper_pos"]])

    def get_observations(self):
        return self.backend.observation(self.group)

    def command_joint_pos(self, pos):
        self.backend.set_target(self.group, pos)

    def get_controller_input(self):
        return self.backend.controller_input(self.group)

    def stop(self):
        self.backend.pause(halt=True)

    def relax(self):
        self.backend.close()


def load_backend_config(station):
    from .config import check_channel_conflicts, robot_channel_for

    config = load_config(station.manimux_config)
    station.__post_init__()
    if config["control_profile"] is not None and config["executor"]["motion_limits"] is not None:
        closing_velocity = config["executor"]["motion_limits"]["gripper"]["max_closing_velocity"]
        duration = 0.0 if closing_velocity is None else 1.0 / closing_velocity
        if abs(station.robot.gripper_close_duration_s - duration) > 1e-9:
            raise ValueError("station gripper_close_duration_s conflicts with control_profile")
    if (
        station.execution_mode == "synchronous"
        and abs(config["robot"]["control_hz"] - station.collection_hz) > 1e-6
    ):
        raise ValueError(
            "synchronous execution requires matching collection_hz and robot.control_hz"
        )
    if config["robot"]["type"] != "yam":
        raise ValueError("collection runtime must select the YAM assembly")
    if station.robot.num_arm_joints != 6:
        raise ValueError("YAM collection requires six arm joints")
    robots = {robot.type: robot for robot in station.robot.robots}
    if len(station.robot.robots) != 2 or set(robots) != {"yam_left", "yam_right"}:
        raise ValueError("YAM collection requires one yam_left and one yam_right")
    bindings = [controller.controls for controller in station.robot.controllers]
    if sorted(bindings) != ["yam_left", "yam_right"]:
        raise ValueError("YAM collection requires exactly one leader per follower")
    check_channel_conflicts(station.robot.robots, station.robot.controllers)
    for side in ("left", "right"):
        robot = robots[f"yam_{side}"]
        options = {"arm_type": station.robot.arm_type, "gripper_type": robot.gripper}
        if station.robot.ee_mass is not None:
            options["ee_mass"] = station.robot.ee_mass
        if robot.gripper_limits is not None:
            options["gripper_limits_override"] = robot.gripper_limits
        # 工位配置拥有设备绑定；SDK 已固定线性夹爪的 50 N 力限。
        # 继续保留原来对非默认力限的拒绝，避免把未实现的设置当作生效。
        if station.robot.gripper_force_limit != 50.0:
            raise ValueError("installed ManiMux i2rt fixes gripper force at 50 N")
        bindings = config["robot"]["options"].setdefault("component_hardware", {})
        bindings[f"{side}_yam"] = {"channel": robot_channel_for(robot), **options}
    config["robot"]["options"]["execute"] = True
    config["robot"]["options"]["move_to_start_on_connect"] = False
    config["robot"]["options"]["home_on_close"] = False
    return config
