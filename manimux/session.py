from __future__ import annotations

import importlib
import math
import threading
import time
import uuid
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot
from manimux.runtime import RunResult, build_runtime
from manimux.types import RobotCommand
from manimux.viewer.communication import ControlClient, RuntimeEvent, ViewerPublisher


class _Runtime(Protocol):
    def run(self) -> RunResult: ...


class _ControlClient(Protocol):
    def poll(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class _Publisher(Protocol):
    def publish(self, message: Any) -> None: ...

    def close(self) -> None: ...


RuntimeFactory = Callable[[dict, Path], _Runtime]
ControlFactory = Callable[[], _ControlClient]
PublisherFactory = Callable[[], _Publisher]


def _build_served_runtime(config: dict, run_dir: Path) -> _Runtime:
    return build_runtime(config, run_dir, launch_mode="serve")


def _new_marvin_session():
    sdk = importlib.import_module("manimux.embodiments.arm.tianji.sdk.marvin.fx_robot")
    return sdk.Marvin_Robot(), sdk.DCSS()


_DRAG_HZ = 250.0
_DRAG_TRACK_RATE_DEG_S = 15.0
# Measured per-arm UMI data from teleop/configs/tool/umi.yaml (2026-08-23).
# Keep the idle Viewer session self-contained instead of importing a sibling checkout.
_DRAG_TOOL = {
    "A": {
        "kine": [-36.745, 0.0, 169.450, 0.0, -90.0, 180.0],
        "dynamic": [
            0.7592901345868115,
            -26.297010972083665,
            -5.114926721017194,
            79.87115777180252,
            0.008816778161434263,
            0.0,
            0.0,
            0.001,
            0.0,
            0.0008785353008534495,
        ],
    },
    "B": {
        "kine": [-36.745, 0.0, 169.450, 0.0, -90.0, 180.0],
        "dynamic": [
            0.7388190421557731,
            -31.69050032486985,
            -7.5863175968931635,
            82.72853176515602,
            0.004192653302493302,
            0.0,
            0.0,
            0.001,
            0.0,
            0.0026402611847252235,
        ],
    },
}


class _TianjiRecovery:
    """Idle Viewer recovery without adding commands to the embodiment classes."""

    def __init__(
        self,
        config: dict,
        *,
        sdk_factory: Callable = _new_marvin_session,
        robot_factory: Callable = build_robot,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        options = config["robot"].get("options", {})
        self._ip = options.get("hardware", {}).get("ip")
        self._sdk_factory = sdk_factory
        self._robot_factory = robot_factory
        self._robot_config = deepcopy(config["robot"])
        self._period_s = 1.0 / float(config["robot"].get("control_hz", 100.0))
        self._monotonic = monotonic
        self._sleep = sleep
        self.available = bool(options.get("execute", False) and self._ip)
        self._busy = False
        self._home_thread: threading.Thread | None = None
        self._home_cancel = threading.Event()
        self._home_error = ""
        self._drag_thread: threading.Thread | None = None
        self._drag_stop = threading.Event()
        self._drag_error = ""
        self._state = "idle"
        self._ack = self._error = self._arm = ""

    @property
    def busy(self) -> bool:
        return self._busy or self._home_thread is not None or self._drag_thread is not None

    def _poll_home(self) -> None:
        if self._home_thread is None or self._home_thread.is_alive():
            return
        self._home_thread.join()
        self._home_thread = None
        self._error = self._home_error
        self._state = "error" if self._error else "idle"

    def _poll_drag(self) -> None:
        if self._drag_thread is None or self._drag_thread.is_alive():
            return
        self._drag_thread.join()
        self._drag_thread = None
        self._error = self._drag_error
        self._state = "error" if self._error else "idle"
        if not self._error:
            self._arm = ""

    def metadata(self) -> dict[str, object]:
        self._poll_home()
        self._poll_drag()
        return {
            "available": self.available,
            "actions": ["clear_error", "home", "drag"] if self.available else [],
            "busy": self.busy,
            "state": self._state,
            "ack": self._ack,
            "error": self._error,
            "arm": self._arm,
        }

    def update(self, control: dict[str, Any]) -> None:
        self._poll_home()
        self._poll_drag()
        if self._drag_thread is not None and not bool(control.get("recovery_lease", False)):
            self._drag_stop.set()
            self._state = "stopping"
        request = str(control.get("recovery_request", ""))
        request_id = str(control.get("recovery_request_id", ""))
        if not self.available or not request or not request_id or request_id == self._ack:
            return
        self._ack = request_id
        if request == "stop":
            if self._drag_thread is not None:
                self._error = ""
                self._drag_stop.set()
                self._state = "stopping"
            elif self._drag_error:
                # A late Stop acknowledgement must not hide a cleanup failure
                # that completed between the Viewer click and this poll.
                self._error = self._drag_error
                self._state = "error"
            else:
                self._error = ""
                self._state = "idle"
                self._arm = ""
            return
        self._error = ""
        if self.busy:
            self._error = "Recovery is already running"
            self._state = "error"
            return
        if request.startswith("drag:"):
            arm = request.partition(":")[2]
            if arm not in {"A", "B", "AB"}:
                self._error = f"Unsupported drag arm: {arm}"
                self._state = "error"
                return
            self._arm = arm
            self._state = "starting"
            self._drag_error = ""
            self._drag_stop.clear()
            self._drag_thread = threading.Thread(
                target=self._drag,
                args=(tuple(arm),),
                name="tianji-viewer-drag",
                daemon=True,
            )
            self._drag_thread.start()
            return
        if request == "home":
            self._state = "homing"
            self._home_error = ""
            self._home_cancel.clear()
            self._home_thread = threading.Thread(
                target=self._home,
                name="tianji-viewer-home",
                daemon=True,
            )
            self._home_thread.start()
            return
        self._busy = True
        self._state = "clearing"
        self._arm = "AB"
        try:
            if request != "clear_error":
                raise ValueError(f"unsupported recovery request: {request}")
            self._clear_errors()
            self._state = "cleared"
        except Exception as exc:  # noqa: BLE001 - report the SDK error in Viewer
            self._state = "error"
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._busy = False

    def _clear_connected(self, controller, buffer, arms: tuple[str, ...]) -> dict:
        for attempt in range(4):
            data = controller.subscribe(buffer)
            if not data:
                raise RuntimeError("Marvin feedback unavailable")
            faults = []
            for arm in arms:
                index = 0 if arm == "A" else 1
                status = data["states"][index]
                if int(status["err_code"]) or int(status["cur_state"]) == 100:
                    faults.append((arm, int(status["err_code"]), int(status["cur_state"])))
            if not faults:
                return data
            if attempt == 3:
                details = ", ".join(
                    f"{arm}: fault {error}, state {state}" for arm, error, state in faults
                )
                raise RuntimeError(
                    "controller did not confirm errors cleared; release the physical "
                    f"E-stop and retry ({details})"
                )
            for arm, _, _ in faults:
                # Marvin can return false while accepting an asynchronous clear;
                # fresh feedback, not this return value, is the confirmation.
                controller.clear_error(arm)
            self._sleep(0.2)
        raise AssertionError("unreachable")

    def _clear_errors(self) -> None:
        controller = None
        connected = False
        try:
            controller, buffer = self._sdk_factory()
            if not controller.connect(self._ip):
                raise RuntimeError("Marvin connect failed")
            connected = True
            self._clear_connected(controller, buffer, ("A", "B"))
        finally:
            if connected and controller is not None and not controller.release_robot():
                raise RuntimeError("Marvin release_robot failed")

    def _drag(self, arms: tuple[str, ...]) -> None:
        controller = None
        buffer = None
        connected = False
        touched: list[str] = []
        errors: list[str] = []
        period_s = 1.0 / _DRAG_HZ
        max_step = _DRAG_TRACK_RATE_DEG_S * period_s
        try:
            controller, buffer = self._sdk_factory()
            if not controller.connect(self._ip):
                raise RuntimeError("Marvin connect failed")
            connected = True
            data = self._clear_connected(controller, buffer, arms)
            last_frames = tuple(
                int(data["outputs"][0 if arm == "A" else 1]["frame_serial"]) for arm in arms
            )
            refreshes = 0
            for _ in range(5):
                self._sleep(0.01)
                data = controller.subscribe(buffer)
                if not data:
                    raise RuntimeError("Marvin feedback unavailable")
                frames = tuple(
                    int(data["outputs"][0 if arm == "A" else 1]["frame_serial"])
                    for arm in arms
                )
                if all(
                    current != previous
                    for current, previous in zip(frames, last_frames, strict=True)
                ):
                    refreshes += 1
                last_frames = frames
            if refreshes < 3:
                raise RuntimeError("Marvin realtime feedback is not refreshing")
            if self._drag_stop.is_set():
                return

            controller.clear_set()
            for arm in arms:
                # Record the cleanup obligation before any mode-setting call:
                # a partially accepted batch must still be disabled.
                touched.append(arm)
                tool = _DRAG_TOOL[arm]
                controller.set_state(arm=arm, state=3)
                controller.set_impedance_type(arm=arm, type=1)
                controller.set_tool(
                    arm=arm,
                    kineParams=list(tool["kine"]),
                    dynamicParams=list(tool["dynamic"]),
                )
                controller.set_joint_kd_params(arm=arm, K=[1.0] * 7, D=[0.3] * 7)
            controller.send_cmd()
            self._sleep(0.5)
            data = controller.subscribe(buffer)
            for arm in arms:
                index = 0 if arm == "A" else 1
                state = data["states"][index]
                if int(state["cur_state"]) != 3 or int(state["err_code"]):
                    raise RuntimeError(
                        f"Arm {arm} failed to enter torque mode "
                        f"(state {state['cur_state']}, fault {state['err_code']})"
                    )
            if self._drag_stop.is_set():
                return

            controller.clear_set()
            for arm in arms:
                controller.set_drag_space(arm=arm, dgType=1)
            controller.send_cmd()
            self._sleep(0.2)
            data = controller.subscribe(buffer)
            for arm in arms:
                index = 0 if arm == "A" else 1
                if int(data["inputs"][index]["drag_sp_type"]) != 1:
                    raise RuntimeError(f"Arm {arm} joint-drag readback failed")

            commands = {
                arm: list(data["outputs"][0 if arm == "A" else 1]["fb_joint_pos"])
                for arm in arms
            }
            if any(
                len(joints) != 7 or not all(math.isfinite(value) for value in joints)
                for joints in commands.values()
            ):
                raise RuntimeError("Marvin returned invalid initial joint feedback")
            frame_watch = {
                arm: (
                    int(data["outputs"][0 if arm == "A" else 1]["frame_serial"]),
                    self._monotonic(),
                )
                for arm in arms
            }
            self._state = "active"
            while not self._drag_stop.is_set():
                data = controller.subscribe(buffer)
                if not data:
                    raise RuntimeError("Marvin feedback unavailable during drag")
                now = self._monotonic()
                for arm in arms:
                    index = 0 if arm == "A" else 1
                    state = data["states"][index]
                    if int(state["cur_state"]) != 3 or int(state["err_code"]):
                        raise RuntimeError(f"Arm {arm} left drag torque mode or faulted")
                    output = data["outputs"][index]
                    frame = int(output["frame_serial"])
                    previous, changed_at = frame_watch[arm]
                    if frame != previous:
                        changed_at = now
                    frame_watch[arm] = (frame, changed_at)
                    if now - changed_at > 0.1:
                        raise RuntimeError(f"Arm {arm} feedback stopped refreshing")
                    measured = list(output["fb_joint_pos"])
                    if len(measured) != 7 or not all(math.isfinite(value) for value in measured):
                        raise RuntimeError(f"Arm {arm} returned invalid joint feedback")
                    commands[arm] = [
                        command + max(-max_step, min(max_step, actual - command))
                        for command, actual in zip(commands[arm], measured, strict=True)
                    ]
                controller.clear_set()
                for arm in arms:
                    controller.set_joint_cmd_pose(arm=arm, joints=commands[arm])
                controller.send_cmd()
                self._sleep(period_s)
        except Exception as exc:  # noqa: BLE001 - surface hardware failures in Viewer
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            if touched:
                self._state = "stopping"
            if connected and controller is not None:
                if touched:
                    try:
                        controller.clear_set()
                        for arm in touched:
                            controller.set_drag_space(arm=arm, dgType=0)
                        controller.send_cmd()
                        self._sleep(0.5)
                    except Exception as exc:  # noqa: BLE001 - continue to servo-off
                        errors.append(f"drag exit failed: {type(exc).__name__}: {exc}")
                    try:
                        remaining = list(touched)
                        for _ in range(3):
                            controller.clear_set()
                            for arm in remaining:
                                controller.set_state(arm=arm, state=0)
                            controller.send_cmd()
                            self._sleep(0.3)
                            data = controller.subscribe(buffer)
                            remaining = [
                                arm
                                for arm in touched
                                if int(data["states"][0 if arm == "A" else 1]["cur_state"])
                                != 0
                            ]
                            if not remaining:
                                break
                        if remaining:
                            raise RuntimeError(
                                f"servo-off was not confirmed for arm(s) {','.join(remaining)}"
                            )
                    except Exception as exc:  # noqa: BLE001 - still release the SDK
                        errors.append(f"servo-off failed: {type(exc).__name__}: {exc}")
                try:
                    if not controller.release_robot():
                        raise RuntimeError("Marvin release_robot failed")
                except Exception as exc:  # noqa: BLE001 - report cleanup failure
                    errors.append(f"{type(exc).__name__}: {exc}")
            self._drag_error = "; ".join(errors)

    def _home(self) -> None:
        robot = None
        errors: list[str] = []
        try:
            # Return Home is also the recovery path after an E-stop. Clear and
            # confirm faults before the normal robot connection can enable arms.
            self._clear_errors()
            robot_config = deepcopy(self._robot_config)
            robot_config.setdefault("options", {})["end_effector_control"] = True
            robot = self._robot_factory(robot_config, SystemClock())
            model = getattr(robot, "model", None)
            targets = {} if model is None else dict(model.home_joints)
            if not targets:
                raise RuntimeError("Tianji-TacCap Home target is not configured")
            robot.connect()
            state = robot.get_state()
            if set(targets) != set(state.groups):
                raise RuntimeError("Home targets do not match the connected robot groups")
            start = {
                name: np.asarray(state.groups[name][: len(target)], dtype=float).copy()
                for name, target in targets.items()
            }
            delta = {name: np.asarray(target) - start[name] for name, target in targets.items()}
            distance = max(float(np.max(np.abs(value))) for value in delta.values())
            if distance > 1e-8:
                # Same 6 deg/s cosine profile used by the previous Tianji Home.
                duration_s = (math.pi / 2.0) * distance / np.radians(6.0)
                tick = 0
                deadline = self._monotonic() + duration_s * 2.0 + 5.0
                while True:
                    if self._home_cancel.is_set():
                        raise RuntimeError("Tianji-TacCap Home cancelled")
                    elapsed = tick * self._period_s
                    fraction = (
                        1.0
                        if elapsed >= duration_s
                        else 0.5 * (1.0 - math.cos(math.pi * elapsed / duration_s))
                    )
                    groups = {
                        name: np.asarray(values).copy() for name, values in state.groups.items()
                    }
                    for name in targets:
                        groups[name][: len(targets[name])] = start[name] + delta[name] * fraction
                    robot.send_command(RobotCommand(groups, time.monotonic_ns(), None))
                    if fraction >= 1.0:
                        break
                    if self._monotonic() > deadline:
                        raise TimeoutError("Tianji-TacCap Home trajectory timed out")
                    tick += 1
                    self._sleep(self._period_s)

            deadline = self._monotonic() + 5.0
            while True:
                if self._home_cancel.is_set():
                    raise RuntimeError("Tianji-TacCap Home cancelled")
                state = robot.get_state()
                if all(
                    np.max(np.abs(state.groups[name][: len(target)] - target)) <= np.radians(0.5)
                    for name, target in targets.items()
                ):
                    break
                if self._monotonic() >= deadline:
                    raise TimeoutError("Tianji-TacCap did not reach Home within 0.5 degrees")
                groups = {name: np.asarray(values).copy() for name, values in state.groups.items()}
                for name, target in targets.items():
                    groups[name][: len(target)] = target
                robot.send_command(RobotCommand(groups, time.monotonic_ns(), None))
                self._sleep(self._period_s)

            state = robot.get_state()
            groups = {name: np.asarray(values).copy() for name, values in state.groups.items()}
            if any(len(groups[name]) != len(target) + 1 for name, target in targets.items()):
                raise RuntimeError("Tianji-TacCap groups must include one gripper coordinate")
            for name in targets:
                groups[name][-1] = 1.0
            robot.send_command(RobotCommand(groups, time.monotonic_ns(), None))
            deadline = self._monotonic() + 5.0
            while True:
                state = robot.get_state()
                if all(state.groups[name][-1] >= 0.98 for name in targets):
                    break
                if self._monotonic() >= deadline:
                    raise TimeoutError("Tianji-TacCap grippers did not fully open")
                self._sleep(self._period_s)
        except Exception as exc:  # noqa: BLE001 - report recovery failure to Viewer
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            if robot is not None:
                try:
                    robot.close()
                except Exception as exc:  # noqa: BLE001 - include cleanup failures
                    errors.append(f"{type(exc).__name__}: {exc}")
            self._home_error = "; ".join(errors)

    def close(self) -> None:
        if self._drag_thread is not None:
            self._drag_stop.set()
            self._state = "stopping"
            self._drag_thread.join()
            self._poll_drag()
        if self._home_thread is not None:
            self._home_cancel.set()
            self._home_thread.join()
            self._poll_home()


class RuntimeSessionService:
    """Keep one runtime service alive while Viser creates isolated rollouts."""

    def __init__(
        self,
        config: dict,
        run_dir: Path,
        *,
        runtime_factory: RuntimeFactory | None = None,
        control_factory: ControlFactory = ControlClient,
        publisher_factory: PublisherFactory = ViewerPublisher,
        poll_interval_s: float = 0.1,
        announcement_interval_s: float = 1.0,
    ) -> None:
        if not config["viewer"]["enabled"]:
            raise ValueError("manimux serve requires viewer.enabled=true")
        self._config = config
        self._run_dir = run_dir
        self._runtime_factory = runtime_factory or _build_served_runtime
        self._control_factory = control_factory
        self._publisher_factory = publisher_factory
        self._poll_interval_s = poll_interval_s
        self._announcement_interval_s = announcement_interval_s
        self._last_episode_dir: Path | None = None
        self._last_error = ""
        self._last_failure_id = ""
        self._recovery = None
        if config["robot"]["type"] == "tianji_taccap" and config["robot"].get("options", {}).get(
            "execute", False
        ):
            self._recovery = _TianjiRecovery(config)

    def _ready_metadata(self) -> dict[str, object]:
        return {
            "run_dir": str(self._run_dir.resolve()),
            "task": self._config["run"]["task"],
            "runtime": self._config["inference"]["algorithm"],
            "executor": self._config["executor"]["type"],
            "policy_label": self._config["viewer"]["policy_label"],
            "camera_map": self._config["policy"]["adapter"].get("camera_map", {}),
            "default_experiment_mode": self._config["run"]["experiment_mode"],
            "default_layout_id": self._config["run"]["layout_id"],
            "last_episode_dir": (
                "" if self._last_episode_dir is None else str(self._last_episode_dir.resolve())
            ),
            "last_error": self._last_error,
            "last_failure_id": self._last_failure_id,
            "recovery": (
                self._recovery.metadata()
                if self._recovery is not None
                else {"available": False}
            ),
        }

    def _publish_once(self, event: str, metadata: dict[str, object]) -> None:
        publisher = self._publisher_factory()
        try:
            publisher.publish(
                RuntimeEvent(
                    event,
                    robot=self._config["viewer"]["robot"],
                    policy=self._config["viewer"]["policy_label"],
                    metadata=metadata,
                )
            )
        finally:
            publisher.close()

    def _wait_for_rollout_request(self) -> dict[str, Any]:
        control = self._control_factory()
        publisher = self._publisher_factory()
        last_announcement = float("-inf")
        try:
            while True:
                state = control.poll()
                recovery_control = (
                    state
                    if state.get("recovery_service_id") == str(self._run_dir.resolve())
                    else {}
                )
                if self._recovery is not None:
                    self._recovery.update(recovery_control)
                now = time.monotonic()
                if now - last_announcement >= self._announcement_interval_s:
                    publisher.publish(
                        RuntimeEvent(
                            "runtime_service_ready",
                            robot=self._config["viewer"]["robot"],
                            policy=self._config["viewer"]["policy_label"],
                            metadata=self._ready_metadata(),
                        )
                    )
                    last_announcement = now
                if (
                    bool(state.get("new_rollout_requested", False))
                    and (self._recovery is None or not self._recovery.busy)
                    and not state.get("recovery_lease", False)
                    and not state.get("recovery_request", "")
                ):
                    return state
                time.sleep(self._poll_interval_s)
        finally:
            try:
                if self._recovery is not None:
                    self._recovery.close()
            finally:
                control.close()
                publisher.close()

    def serve(self, *, max_rollout_attempts: int | None = None) -> None:
        attempts = 0
        print(f"runtime service ready; run_dir={self._run_dir.resolve()}")
        print(
            "Viewer flow: Prepare normal/experiment rollout -> Start rollout -> "
            "Finish & Home / Finish without homing"
        )
        if self._recovery is not None and self._recovery.available:
            print("Manual recovery is always visible: stop a rollout, drag A/B/AB, or Return Home")
        print(
            "Normal rollouts need no scoring; experiment rollouts offer evaluation "
            "that can be saved or skipped"
        )
        while max_rollout_attempts is None or attempts < max_rollout_attempts:
            request = self._wait_for_rollout_request()
            attempts += 1
            self._last_error = ""
            self._last_failure_id = ""
            rollout_config = deepcopy(self._config)
            task_command = str(request.get("task_command", "")).strip()
            if task_command:
                rollout_config["run"]["task"] = task_command
            rollout_config["run"]["experiment_mode"] = bool(
                request.get("experiment_mode", self._config["run"]["experiment_mode"])
            )
            rollout_config["run"]["layout_id"] = str(
                request.get("layout_id", self._config["run"]["layout_id"])
            ).strip()
            try:
                result = self._runtime_factory(rollout_config, self._run_dir).run()
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the session alive after one failed episode
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._last_failure_id = uuid.uuid4().hex
                print(f"rollout attempt failed; service remains available: {self._last_error}")
                self._publish_once(
                    "episode_failed",
                    {
                        **self._ready_metadata(),
                        "error": self._last_error,
                    },
                )
                continue
            self._last_episode_dir = result.episode_dir
            print(
                f"rollout completed; reason={result.terminal_reason}; episode={result.episode_dir}"
            )
