from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot
from manimux.runtime import RunResult, build_runtime
from manimux.robogui.communication import ControlClient, RuntimeEvent, RoboGUIPublisher


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


def _error_text(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(_error_text(inner) for inner in error.exceptions)
    return f"{type(error).__name__}: {error}"


class _IdleRecovery:
    """Idle RoboGUI recovery protocol; the robot assembly implements every action.

    The assembly's recovery_actions and drag_selections decide what RoboGUI may
    request. Each request builds a fresh, disconnected assembly from the config.
    """

    def __init__(self, config: dict, *, robot_factory: Callable = build_robot) -> None:
        self._robot_factory = robot_factory
        self._robot_config = deepcopy(config["robot"])
        # Reading capabilities constructs the assembly without opening any device.
        robot = self._build(end_effector_control=False)
        executing = bool(self._robot_config.get("options", {}).get("execute", False))
        self.actions = tuple(robot.recovery_actions) if executing else ()
        self._drag_selections = dict(robot.drag_selections)
        self.available = bool(self.actions)
        self._busy = False
        self._task = ""  # "home" or "drag" while the worker thread runs.
        self._thread: threading.Thread | None = None
        self._thread_error = ""
        self._drag_stop = threading.Event()
        self._state = "idle"
        self._ack = self._error = self._arm = ""

    @property
    def busy(self) -> bool:
        return self._busy or self._thread is not None

    def _build(self, *, end_effector_control: bool):
        config = deepcopy(self._robot_config)
        config.setdefault("options", {})["end_effector_control"] = end_effector_control
        return self._robot_factory(config, SystemClock())

    def _poll(self) -> None:
        if self._thread is None or self._thread.is_alive():
            return
        self._thread.join()
        self._thread = None
        self._task = ""
        self._error = self._thread_error
        self._state = "error" if self._error else "idle"
        if not self._error:
            self._arm = ""

    def metadata(self) -> dict[str, object]:
        self._poll()
        return {
            "available": self.available,
            "actions": list(self.actions),
            "busy": self.busy,
            "state": self._state,
            "ack": self._ack,
            "error": self._error,
            "arm": self._arm,
        }

    def update(self, control: dict[str, Any]) -> None:
        self._poll()
        if self._task == "drag" and not bool(control.get("recovery_lease", False)):
            self._drag_stop.set()
            self._state = "stopping"
        request = str(control.get("recovery_request", ""))
        request_id = str(control.get("recovery_request_id", ""))
        if not self.available or not request or not request_id or request_id == self._ack:
            return
        self._ack = request_id
        if request == "stop":
            if self._task == "drag":
                self._error = ""
                self._drag_stop.set()
                self._state = "stopping"
            elif self._thread_error:
                # A late Stop acknowledgement must not hide a cleanup failure
                # that completed between the RoboGUI click and this poll.
                self._error = self._thread_error
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
        action, _, selection = request.partition(":")
        if action not in self.actions:
            self._error = f"Unsupported recovery request: {request}"
            self._state = "error"
            return
        if action == "drag":
            if selection not in self._drag_selections:
                self._error = f"Unsupported drag arm: {selection}"
                self._state = "error"
                return
            self._arm = selection
            self._state = "starting"
            self._start("drag", self._drag, self._drag_selections[selection])
            return
        if action == "home":
            self._state = "homing"
            self._start("home", self._home)
            return
        self._busy = True
        self._state = "clearing"
        # Clearing faults affects every arm: report the selection covering the most arms.
        self._arm = max(
            self._drag_selections,
            key=lambda label: len(self._drag_selections[label]),
            default="",
        )
        try:
            self._build(end_effector_control=False).clear_errors()
            self._state = "cleared"
        except Exception as exc:  # noqa: BLE001 - report the robot error in RoboGUI
            self._state = "error"
            self._error = _error_text(exc)
        finally:
            self._busy = False

    def _start(self, task: str, job: Callable, *args) -> None:
        def run() -> None:
            try:
                job(*args)
            except Exception as exc:  # noqa: BLE001 - surface hardware failures in RoboGUI
                self._thread_error = _error_text(exc)

        self._task = task
        self._thread_error = ""
        self._drag_stop.clear()
        self._thread = threading.Thread(target=run, name=f"robogui-recovery-{task}", daemon=True)
        self._thread.start()

    def _drag_active(self) -> None:
        if not self._drag_stop.is_set():
            self._state = "active"

    def _drag(self, groups: tuple[str, ...]) -> None:
        # Drag uses only the arms, so grippers stay closed.
        robot = self._build(end_effector_control=False)
        robot.recover_drag(groups, self._drag_stop, self._drag_active)

    def _home(self) -> None:
        # Home also restores the grippers, so they are opened for this assembly.
        self._build(end_effector_control=True).recover_home()

    def close(self) -> None:
        if self._thread is not None:
            if self._task == "drag":
                self._drag_stop.set()
                self._state = "stopping"
            # Home has no cancellation; it finishes before the service exits.
            self._thread.join()
            self._poll()


class RuntimeSessionService:
    """Keep one runtime service alive while Viser creates isolated rollouts."""

    def __init__(
        self,
        config: dict,
        run_dir: Path,
        *,
        runtime_factory: RuntimeFactory | None = None,
        control_factory: ControlFactory = ControlClient,
        publisher_factory: PublisherFactory = RoboGUIPublisher,
        poll_interval_s: float = 0.1,
        announcement_interval_s: float = 1.0,
    ) -> None:
        if not config["robogui"]["enabled"]:
            raise ValueError("manimux serve requires robogui.enabled=true")
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
        if config["robot"].get("options", {}).get("execute", False):
            recovery = _IdleRecovery(config)
            self._recovery = recovery if recovery.available else None

    def _ready_metadata(self) -> dict[str, object]:
        return {
            "run_dir": str(self._run_dir.resolve()),
            "task": self._config["run"]["task"],
            "evaluation": deepcopy(self._config.get("evaluation", {"kind": "binary"})),
            "runtime": self._config["inference"]["algorithm"],
            "executor": self._config["executor"]["type"],
            "policy_label": self._config["robogui"]["policy_label"],
            "camera_map": self._config["policy"]["adapter"].get("camera_map", {}),
            "experiment_template": deepcopy(self._config["run"].get("experiment_template")),
            "research_defaults": {
                key: self._config["run"].get(key, "")
                for key in ("experiment_name", "condition", "notes")
            },
            "default_experiment_mode": self._config["run"]["experiment_mode"],
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
                    robot=self._config["robogui"]["robot"],
                    policy=self._config["robogui"]["policy_label"],
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
                            robot=self._config["robogui"]["robot"],
                            policy=self._config["robogui"]["policy_label"],
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
            "RoboGUI flow: Prepare free/study rollout -> Start rollout -> "
            "Finish & Home / Finish without homing"
        )
        if self._recovery is not None and self._recovery.available:
            print("Manual recovery is always visible: stop a rollout, drag A/B/AB, or Return Home")
        print(
            "Free rollouts need no scoring; study rollouts offer evaluation "
            "that can be saved or skipped"
        )
        while max_rollout_attempts is None or attempts < max_rollout_attempts:
            request = self._wait_for_rollout_request()
            attempts += 1
            self._last_error = ""
            self._last_failure_id = ""
            try:
                rollout_config = deepcopy(self._config)
                task_command = str(request.get("task_command", "")).strip()
                if task_command:
                    rollout_config["run"]["task"] = task_command
                # Identity comes only from this Prepare, never the preceding attempt.
                rollout_config["run"].update(
                    experiment_name=request.get("experiment_name", ""),
                    condition=request.get("condition", ""),
                    notes=request.get("notes", ""),
                    experiment_mode=request.get("experiment_mode", False),
                    layout_id=request.get("layout_id", ""),
                    repeat_id=request.get("repeat_id"),
                    reference_layout=deepcopy(request.get("reference_layout")),
                )
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
