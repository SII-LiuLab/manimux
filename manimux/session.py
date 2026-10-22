from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from manimux.clock import SystemClock
from manimux.configuration import initial_pose_metadata
from manimux.embodiments.robot import build_robot
from manimux.experiment_list import ExperimentList
from manimux.failures import describe_failure
from manimux.robogui.communication import ControlClient, RoboGUIPublisher, RuntimeEvent
from manimux.runtime import RunResult, build_runtime


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

    The assembly's recovery_actions, hand_guiding_actions and drag_selections
    decide what RoboGUI may request. Each request builds a fresh, disconnected
    assembly from the config; hand-guiding mode and duration belong to the robot.
    """

    def __init__(self, config: dict, *, robot_factory: Callable = build_robot) -> None:
        self._robot_factory = robot_factory
        self._robot_config = deepcopy(config["robot"])
        # Reading capabilities constructs the assembly without opening any device.
        robot = self._build()
        executing = bool(self._robot_config.get("options", {}).get("execute", False))
        self.actions = tuple(robot.recovery_actions) if executing else ()
        self._hand_guiding_actions = tuple(robot.hand_guiding_actions)
        self._drag_selections = dict(robot.drag_selections)
        self.available = bool(self.actions)
        self._busy = False
        self._task = ""  # Robot-owned action running on the recovery worker.
        self._thread: threading.Thread | None = None
        self._thread_error = ""
        self._recovery_stop = threading.Event()
        self._state = "idle"
        self._ack = self._error = self._arm = ""

    @property
    def busy(self) -> bool:
        return self._busy or self._thread is not None

    def _build(self):
        return self._robot_factory(deepcopy(self._robot_config), SystemClock())

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
            "hand_guiding_actions": list(self._hand_guiding_actions),
            "task": self._task,
            "selections": list(self._drag_selections),
            "busy": self.busy,
            "state": self._state,
            "ack": self._ack,
            "error": self._error,
            "arm": self._arm,
        }

    def update(self, control: dict[str, Any]) -> None:
        self._poll()
        if self._task in self._hand_guiding_actions and not control.get("recovery_lease", False):
            self._recovery_stop.set()
            self._state = "stopping"
        request = str(control.get("recovery_request", ""))
        request_id = str(control.get("recovery_request_id", ""))
        if not self.available or not request or not request_id or request_id == self._ack:
            return
        self._ack = request_id
        if request == "stop":
            if self._thread is not None:
                self._error = ""
                self._recovery_stop.set()
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
        if action in self._hand_guiding_actions:
            if selection not in self._drag_selections:
                self._error = f"Unsupported {action} arm: {selection}"
                self._state = "error"
                return
            if not control.get("recovery_lease", False):
                self._error = "Hand-guiding requires an active recovery lease"
                self._state = "error"
                return
            self._arm = selection
            self._state = "starting"
            self._start(action, self._hand_guide, action, self._drag_selections[selection])
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
            self._build().clear_errors()
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
        self._recovery_stop.clear()
        self._thread = threading.Thread(target=run, name=f"robogui-recovery-{task}", daemon=True)
        self._thread.start()

    def _hand_guiding_active(self) -> None:
        if not self._recovery_stop.is_set():
            self._state = "active"

    def _hand_guide(self, action: str, groups: tuple[str, ...]) -> None:
        self._build().hand_guiding_actions[action](
            groups, self._recovery_stop, self._hand_guiding_active
        )

    def _home(self) -> None:
        self._build().recover_home(self._recovery_stop)

    def close(self) -> None:
        if self._thread is not None:
            self._recovery_stop.set()
            self._state = "stopping"
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
        poll_interval_s: float = 0.02,
        announcement_interval_s: float = 1.0,
        experiment_list: ExperimentList | None = None,
        selection_dir_factory: Callable[[str], Path] | None = None,
        pose_dir_factory: Callable[[dict], Path] | None = None,
        allow_experiments: bool = True,
    ) -> None:
        if not config["robogui"]["enabled"]:
            raise ValueError(
                "Interactive manimux run requires robogui.enabled=true; "
                "use --once for a standalone headless rollout"
            )
        self._config = config
        self._experiment_list = experiment_list
        self._allow_experiments = allow_experiments
        self._selection_dir_factory = selection_dir_factory
        self._pose_dir_factory = pose_dir_factory
        self._pose_base_config = deepcopy(config)
        if experiment_list is not None and selection_dir_factory is None:
            raise ValueError("Experiment switching requires a recording directory factory")
        self._selection_id = experiment_list.default_id if experiment_list else ""
        self._selection_error = ""
        self._run_dir = run_dir
        self._runtime_factory = runtime_factory or _build_served_runtime
        self._control_factory = control_factory
        self._publisher_factory = publisher_factory
        self._poll_interval_s = poll_interval_s
        self._announcement_interval_s = announcement_interval_s
        self._last_episode_dir: Path | None = None
        self._last_error = ""
        self._last_failure = {}
        self._last_failure_id = ""
        self._prepare_ack = ""
        self._recovery = None
        if config["robot"].get("options", {}).get("execute", False):
            recovery = _IdleRecovery(config)
            self._recovery = recovery if recovery.available else None

    def _ready_metadata(self) -> dict[str, object]:
        from manimux.agents.execution import execution_choices
        return {
            "execution_choices": execution_choices(self._config),
            "execution_mode": self._config["run"].get("execution_mode", "vla"),
            "run_dir": str(self._run_dir.resolve()),
            "history_sessions": self._config["run"].get("history_sessions", []),
            "experiment_choices": self._experiment_list.metadata() if self._experiment_list else [],
            "experiment_id": self._selection_id,
            "selection_error": self._selection_error,
            "initial_pose": initial_pose_metadata(self._config),
            "initial_pose_options": deepcopy(self._config["run"].get("initial_pose_options", [])),
            "task": self._config["run"]["task_prompt"],
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
            "experiments_available": self._allow_experiments,
            "last_episode_dir": (
                "" if self._last_episode_dir is None else str(self._last_episode_dir.resolve())
            ),
            "last_error": self._last_error,
            "failure": self._last_failure,
            "last_failure_id": self._last_failure_id,
            "prepare_ack": self._prepare_ack,
            "recovery": (
                self._recovery.metadata() if self._recovery is not None else {"available": False}
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
                if self._try_select_experiment(state):
                    last_announcement = float("-inf")
                elif self._try_select_initial_pose(state):
                    last_announcement = float("-inf")
                now = time.monotonic()
                if now - last_announcement >= self._announcement_interval_s:
                    from manimux.deployment.reporter import report
                    report(
                        "recovery" if self._recovery is not None and self._recovery.busy
                        else "waiting_for_prepare",
                        run_dir=str(self._run_dir.resolve()),
                        last_episode_dir=(
                            str(self._last_episode_dir) if self._last_episode_dir else None
                        ),
                        detail="Runtime session alive; robot readiness remains unverified",
                    )
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
                    and (not state.get("prepare_request_id")
                         or state["prepare_request_id"] != self._prepare_ack)
                    and (self._recovery is None or not self._recovery.busy)
                    and not state.get("recovery_lease", False)
                    and not state.get("recovery_request", "")
                    and not state.get("select_experiment_id")
                    and not state.get("select_initial_pose_id")
                    and (self._experiment_list is None or (
                        state.get("experiment_id") == self._selection_id
                        and state.get("control_run_dir") == str(self._run_dir.resolve())
                    ))
                ):
                    self._prepare_ack = str(state.get("prepare_request_id", ""))
                    from manimux.deployment.reporter import report
                    report("operator_control", run_dir=str(self._run_dir.resolve()),
                           detail="Operator requested Prepare; see RoboGUI for rollout/cleanup")
                    publisher.publish(RuntimeEvent(
                        "rollout_prepare_accepted",
                        robot=self._config["robogui"]["robot"],
                        policy=self._config["robogui"]["policy_label"],
                        metadata={"run_dir": str(self._run_dir.resolve()),
                                  "prepare_ack": self._prepare_ack},
                    ))
                    return state
                time.sleep(self._poll_interval_s)
        finally:
            try:
                if self._recovery is not None:
                    self._recovery.close()
            finally:
                control.close()
                publisher.close()

    def _try_select_experiment(self, state: dict[str, Any]) -> bool:
        """Called only by the idle loop; never mutate an active runtime."""
        requested = state.get("select_experiment_id")
        if not requested or self._experiment_list is None:
            return False
        if state.get("control_run_dir") != str(self._run_dir.resolve()):
            return False  # A delayed request from the preceding session.
        if (state.get("evaluation_complete") is not True
                or state.get("new_rollout_requested")
                or state.get("recovery_lease") or state.get("recovery_request")
                or (self._recovery is not None and self._recovery.busy)):
            self._selection_error = "Finish evaluation and recovery before changing experiments."
            return True
        if requested == self._selection_id:
            self._selection_error = ""
            return False
        try:
            choice = self._experiment_list.choices[requested]
            from manimux.collections import service_contract, pose_configuration
            if service_contract(choice.config) != service_contract(self._config):
                raise ValueError("Service configuration differs. Restart affected services and runtime with --experiment-id " + requested)
            selected = deepcopy(choice.config)
            pose_id = initial_pose_metadata(self._config)["id"]
            if (self._config["run"]["task_prompt"] == selected["run"]["task_prompt"]
                    and pose_id != initial_pose_metadata(self._pose_base_config)["id"]
                    and any(p["id"] == pose_id for p in selected["run"].get("initial_pose_options", []))):
                selected = pose_configuration(selected, pose_id)
                run_dir = self._pose_dir_factory(selected)
            else:
                run_dir = self._selection_dir_factory(requested)
        except (KeyError, TypeError, OSError, ValueError) as exc:
            self._selection_error = f"Cannot select experiment: {exc}"
            return True
        self._config = selected
        self._pose_base_config = deepcopy(choice.config)
        self._run_dir = run_dir
        self._selection_id = requested
        self._selection_error = self._last_error = self._last_failure_id = ""
        self._last_failure = {}
        self._last_episode_dir = None
        # Hardware/recovery contract is identical across the validated catalog.
        # Next Prepare builds a fresh worker, strategy, timeline and warmup state.
        return True

    def _try_select_initial_pose(self, state: dict[str, Any]) -> bool:
        """Idle-only configuration selection; actual motion belongs to Prepare."""
        requested = state.get("select_initial_pose_id")
        if not requested or state.get("select_experiment_id"):
            return False
        if state.get("control_run_dir") != str(self._run_dir.resolve()):
            return False
        if (state.get("evaluation_complete") is not True
                or state.get("new_rollout_requested")
                or state.get("recovery_lease") or state.get("recovery_request")
                or (self._recovery is not None and self._recovery.busy)):
            self._selection_error = "Finish evaluation and recovery before changing initial pose."
            return True
        if requested == initial_pose_metadata(self._config)["id"]:
            self._selection_error = ""
            return False
        try:
            if self._pose_dir_factory is None:
                raise ValueError("Initial pose switching is unavailable in this service")
            from manimux.collections import pose_configuration
            config = pose_configuration(self._pose_base_config, requested)
            run_dir = self._pose_dir_factory(config)
        except (KeyError, TypeError, OSError, ValueError) as exc:
            self._selection_error = f"Cannot select initial pose: {exc}"
            return True
        self._config, self._run_dir = config, run_dir
        self._selection_error = self._last_error = self._last_failure_id = ""
        self._last_failure = {}
        self._last_episode_dir = None
        return True

    def serve(self, *, max_rollout_attempts: int | None = None) -> None:
        attempts = 0
        print(f"runtime service ready; run_dir={self._run_dir.resolve()}")
        print("RoboGUI flow: Prepare -> Activate Warmup (when enabled) -> Start rollout -> Finish")
        if self._recovery is not None and self._recovery.available:
            print(
                "Manual recovery follows the assembly declaration: "
                + ", ".join(self._recovery.actions)
            )
        print(
            "Use Prepare Run or Prepare Experiment in RoboGUI; experiments offer evaluation "
            "that can be saved or skipped" if self._allow_experiments else
            "Trial mode: use Prepare Run, then Start. Trials do not count toward study progress."
        )
        while max_rollout_attempts is None or attempts < max_rollout_attempts:
            request = self._wait_for_rollout_request()
            attempts += 1
            self._last_error = ""
            self._last_failure = {}
            self._last_failure_id = ""
            try:
                if not self._allow_experiments and request.get("experiment_mode", False):
                    raise ValueError(
                        "This is a --config trial; use --config-list for scored experiments"
                    )
                rollout_config = deepcopy(self._config)
                from manimux.agents.execution import execution_choices, execution_condition
                execution_mode = request.get(
                    "execution_mode", rollout_config["run"].get("execution_mode", "vla")
                )
                if execution_mode not in execution_choices(rollout_config):
                    raise ValueError("Selected execution mode is not configured for this experiment")
                rollout_config["run"]["execution_mode"] = execution_mode
                task_command = str(request.get("task_command", "")).strip()
                if (self._experiment_list is not None and task_command
                        and task_command != rollout_config["run"]["task_prompt"]):
                    raise ValueError("Task instruction does not match the selected experiment")
                if task_command:
                    rollout_config["run"]["task_prompt"] = task_command
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
                rollout_config["run"]["condition"] = execution_condition(
                    self._config["run"].get("condition", ""), execution_mode
                ) if execution_mode == "rpent" else request.get("condition", "")
                from manimux.runtime.lifecycle import OperationStatus
                progress = OperationStatus(
                    enabled=True, robot=self._config["robogui"]["robot"],
                    policy=self._config["robogui"]["policy_label"], event="runtime_preparing",
                    identity={"run_dir": str(self._run_dir.resolve()),
                              "prepare_request_id": self._prepare_ack},
                )
                progress.update("Building runtime")
                try:
                    runtime = self._runtime_factory(rollout_config, self._run_dir)
                finally:
                    progress.close()
                result = runtime.run()
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the session alive after one failed episode
                self._last_failure = describe_failure(exc)
                self._last_error = self._last_failure["detail"]
                self._last_failure_id = uuid.uuid4().hex
                failure = self._last_failure
                print(f"{failure['title']} [{failure['source']} / {failure['stage']} / {failure['target']}]: {failure['reason']}")
                print(f"Next step: {failure['action']}")
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
            self._refresh_record_index()
            print(
                f"rollout completed; reason={result.terminal_reason}; episode={result.episode_dir}"
            )

    def _refresh_record_index(self) -> None:
        collection = self._config.get("collection")
        if collection is None:
            return
        from manimux.collections import write_inventory
        try:
            write_inventory(Path(collection["root"]) / collection["id"])
        except (OSError, ValueError) as error:
            print(f"Record index refresh failed; raw recordings are intact: {error}")
