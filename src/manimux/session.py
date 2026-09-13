from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from manimux.config import ManiMuxConfig
from manimux.runtime import RunResult, build_runtime
from manimux.viewer.protocol import RuntimeEvent
from manimux.viewer.transport import ControlClient, ViewerPublisher


class _Runtime(Protocol):
    def run(self) -> RunResult: ...


class _ControlClient(Protocol):
    def poll(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class _Publisher(Protocol):
    def publish(self, message: Any) -> None: ...

    def close(self) -> None: ...


RuntimeFactory = Callable[[ManiMuxConfig, Path], _Runtime]
ControlFactory = Callable[[], _ControlClient]
PublisherFactory = Callable[[], _Publisher]


def _build_served_runtime(config: ManiMuxConfig, run_dir: Path) -> _Runtime:
    return build_runtime(config, run_dir, launch_mode="serve")


class RuntimeSessionService:
    """Keep one runtime service alive while Viser creates isolated rollouts."""

    def __init__(
        self,
        config: ManiMuxConfig,
        run_dir: Path,
        *,
        runtime_factory: RuntimeFactory | None = None,
        control_factory: ControlFactory = ControlClient,
        publisher_factory: PublisherFactory = ViewerPublisher,
        poll_interval_s: float = 0.1,
        announcement_interval_s: float = 1.0,
    ) -> None:
        if not config.viewer.enabled:
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

    def _ready_metadata(self) -> dict[str, object]:
        return {
            "run_dir": str(self._run_dir.resolve()),
            "task": self._config.run.task,
            "runtime": self._config.execution.runtime,
            "executor": self._config.execution.executor,
            "policy_label": self._config.viewer.policy_label,
            "camera_map": self._config.policy.options.get("camera_map", {}),
            "default_experiment_mode": self._config.run.experiment_mode,
            "default_layout_id": self._config.run.layout_id,
            "last_episode_dir": (
                "" if self._last_episode_dir is None else str(self._last_episode_dir.resolve())
            ),
            "last_error": self._last_error,
        }

    def _publish_once(self, event: str, metadata: dict[str, object]) -> None:
        publisher = self._publisher_factory()
        try:
            publisher.publish(
                RuntimeEvent(
                    event,
                    robot=self._config.viewer.robot_adapter,
                    policy=self._config.viewer.policy_label,
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
                now = time.monotonic()
                if now - last_announcement >= self._announcement_interval_s:
                    publisher.publish(
                        RuntimeEvent(
                            "runtime_service_ready",
                            robot=self._config.viewer.robot_adapter,
                            policy=self._config.viewer.policy_label,
                            metadata=self._ready_metadata(),
                        )
                    )
                    last_announcement = now
                state = control.poll()
                if bool(state.get("new_rollout_requested", False)):
                    return state
                time.sleep(self._poll_interval_s)
        finally:
            control.close()
            publisher.close()

    def serve(self, *, max_rollout_attempts: int | None = None) -> None:
        attempts = 0
        print(f"runtime service ready; run_dir={self._run_dir.resolve()}")
        print(
            "Viewer flow: Prepare normal/experiment rollout -> Start rollout -> "
            "Finish & Home"
        )
        print(
            "Normal rollouts require no reward; experiment rollouts require a human "
            "label before the next rollout"
        )
        while max_rollout_attempts is None or attempts < max_rollout_attempts:
            request = self._wait_for_rollout_request()
            attempts += 1
            self._last_error = ""
            rollout_config = self._config.model_copy(deep=True)
            task_command = str(request.get("task_command", "")).strip()
            if task_command:
                rollout_config.run.task = task_command
            rollout_config.run.experiment_mode = bool(
                request.get("experiment_mode", self._config.run.experiment_mode)
            )
            rollout_config.run.layout_id = str(
                request.get("layout_id", self._config.run.layout_id)
            ).strip()
            try:
                result = self._runtime_factory(rollout_config, self._run_dir).run()
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the session alive after one failed episode
                self._last_error = f"{type(exc).__name__}: {exc}"
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
                f"rollout completed; reason={result.terminal_reason}; "
                f"episode={result.episode_dir}"
            )
