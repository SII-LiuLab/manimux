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


# Viewer drag choices use the Marvin arm labels.
_DRAG_SIDES = {"A": ("left",), "B": ("right",), "AB": ("left", "right")}


def _error_text(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(_error_text(inner) for inner in error.exceptions)
    return f"{type(error).__name__}: {error}"


class _TianjiRecovery:
    """Idle Viewer recovery protocol; the robot assembly owns every hardware action."""

    def __init__(self, config: dict, *, robot_factory: Callable = build_robot) -> None:
        options = config["robot"].get("options", {})
        self._robot_factory = robot_factory
        self._robot_config = deepcopy(config["robot"])
        ip = options.get("hardware", {}).get("ip")
        self.available = bool(options.get("execute", False) and ip)
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
            "actions": ["clear_error", "home", "drag"] if self.available else [],
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
                # that completed between the Viewer click and this poll.
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
        if request.startswith("drag:"):
            arm = request.partition(":")[2]
            if arm not in _DRAG_SIDES:
                self._error = f"Unsupported drag arm: {arm}"
                self._state = "error"
                return
            self._arm = arm
            self._state = "starting"
            self._start("drag", self._drag, _DRAG_SIDES[arm])
            return
        if request == "home":
            self._state = "homing"
            self._start("home", self._home)
            return
        self._busy = True
        self._state = "clearing"
        self._arm = "AB"
        try:
            if request != "clear_error":
                raise ValueError(f"unsupported recovery request: {request}")
            self._build(end_effector_control=False).clear_errors()
            self._state = "cleared"
        except Exception as exc:  # noqa: BLE001 - report the SDK error in Viewer
            self._state = "error"
            self._error = _error_text(exc)
        finally:
            self._busy = False

    def _start(self, task: str, job: Callable, *args) -> None:
        def run() -> None:
            try:
                job(*args)
            except Exception as exc:  # noqa: BLE001 - surface hardware failures in Viewer
                self._thread_error = _error_text(exc)

        self._task = task
        self._thread_error = ""
        self._drag_stop.clear()
        self._thread = threading.Thread(target=run, name=f"tianji-viewer-{task}", daemon=True)
        self._thread.start()

    def _drag_active(self) -> None:
        if not self._drag_stop.is_set():
            self._state = "active"

    def _drag(self, sides: tuple[str, ...]) -> None:
        # connect() inside drag rejects faulted arms, so latched errors are cleared first.
        robot = self._build(end_effector_control=False)
        robot.clear_errors()
        robot.drag(sides, self._drag_stop, self._drag_active)

    def _home(self) -> None:
        # Return Home is also the recovery path after an E-stop.
        robot = self._build(end_effector_control=True)
        robot.clear_errors()
        robot.connect()
        try:
            robot.home()
        except Exception as error:
            try:
                robot.close()
            except Exception as cleanup_error:
                raise ExceptionGroup("home and cleanup failed", [error, cleanup_error]) from None
            raise
        robot.close()

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
            "Normal rollouts require no reward; experiment rollouts require a human "
            "label before the next rollout"
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
