from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from manimux.types import ActionChunk, ActionHorizon, RobotState, SensorFrame

from .communication import PolicyPlan, RobotSnapshot, RuntimeEvent, ViewerPublisher


@dataclass(frozen=True, slots=True)
class ViewerControl:
    paused: bool
    home_requested: bool = False
    finish_requested: bool = False
    finish_home: bool | None = None


class ViewerBridge:
    """Best-effort bridge to the viewer bundled with ManiMux."""

    def __init__(
        self,
        enabled: bool,
        robot: str,
        *,
        policy: str = "manimux-local",
        instruction: str = "",
        camera_hz: float = 5.0,
    ) -> None:
        if camera_hz < 0:
            raise ValueError("camera_hz must be non-negative")
        self._enabled = enabled
        self._robot = robot
        self._policy = policy
        self._instruction = instruction
        self._camera_period_s = 0.0 if camera_hz == 0 else 1.0 / camera_hz
        self._last_camera_publish = float("-inf")
        self._state_metadata: dict[str, object] = {}
        self._publisher: Any | None = None
        self._controls: Any | None = None
        self._policy_plan_type: Any | None = None
        self._snapshot_type: Any | None = None
        self._runtime_event_type: Any | None = None
        if not enabled:
            return
        from manimux.viewer.communication import (
            ControlClient,
            PolicyPlan,
            RobotSnapshot,
            RuntimeEvent,
            ViewerPublisher,
        )

        self._publisher = ViewerPublisher()
        self._controls = ControlClient()
        self._policy_plan_type = PolicyPlan
        self._snapshot_type = RobotSnapshot
        self._runtime_event_type = RuntimeEvent

    def publish_event(
        self,
        event: str,
        *,
        step: int = 0,
        chunk_id: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if not self._enabled:
            return
        assert self._publisher is not None
        assert self._runtime_event_type is not None
        self._publisher.publish(
            self._runtime_event_type(
                event=event,
                robot=self._robot,
                policy=self._policy,
                step=step,
                chunk_id=chunk_id,
                metadata=dict(metadata or {}),
            )
        )

    def poll_control(self) -> ViewerControl:
        if not self._enabled:
            return ViewerControl(paused=False)
        assert self._controls is not None
        state = self._controls.poll()
        return ViewerControl(
            paused=bool(state.get("paused", True)),
            home_requested=bool(state.get("home_requested", False)),
            finish_requested=bool(state.get("finish_requested", False)),
            finish_home=(
                state.get("finish_home") if isinstance(state.get("finish_home"), bool) else None
            ),
        )

    def set_state_metadata(self, metadata: dict[str, object]) -> None:
        """Attach recoverable rollout context to every state heartbeat."""

        self._state_metadata = dict(metadata)

    def publish_plan(
        self,
        chunk: ActionChunk,
        inference_ms: float,
        *,
        committed: ActionHorizon | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if not self._enabled:
            return
        assert self._publisher is not None
        assert self._policy_plan_type is not None
        groups = chunk.groups if committed is None else committed.groups
        action_dt_ns = chunk.dt_ns if committed is None else committed.dt_ns
        plan_metadata: dict[str, object] = {"plan_id": chunk.plan_id}
        if committed is not None:
            plan_metadata["committed_start_time_ns"] = committed.start_time_ns
        plan_metadata.update(metadata or {})
        message = self._policy_plan_type(
            robot=self._robot,
            policy=self._policy,
            instruction=self._instruction,
            groups=dict(groups),
            action_space=chunk.action_space,
            action_dt=action_dt_ns / 1_000_000_000,
            inference_ms=inference_ms,
            chunk_id=chunk.request_seq,
            metadata=plan_metadata,
        )
        self._publisher.publish(message)

    def publish_state(
        self,
        state: RobotState,
        frames: dict[str, SensorFrame],
        *,
        step: int,
        max_steps: int,
        chunk_index: int = 0,
        active_chunk_id: int | None = None,
    ) -> None:
        if not self._enabled:
            return
        assert self._publisher is not None
        assert self._snapshot_type is not None
        now = time.monotonic()
        publish_frames: dict[str, SensorFrame] = {}
        if self._camera_period_s > 0 and now - self._last_camera_publish >= self._camera_period_s:
            publish_frames = frames
            if publish_frames:
                self._last_camera_publish = now
        message = self._snapshot_type(
            robot=self._robot,
            groups=dict(state.groups),
            timestamp_ns=state.monotonic_ns,
            sequence=state.sequence,
            camera_metadata={
                name: {"timestamp_ns": frame.capture_monotonic_ns, "sequence": frame.sequence}
                for name, frame in publish_frames.items()
            },
            cameras={name: frame.data for name, frame in publish_frames.items()},
            step=step,
            max_steps=max_steps,
            chunk_index=chunk_index,
            active_chunk_id=active_chunk_id,
            connected=True,
            metadata=dict(self._state_metadata),
        )
        self._publisher.publish(message)

    def close(self) -> None:
        for component in (self._publisher, self._controls):
            if component is not None:
                component.close()


class ViewerClient:
    """Non-owning observer used by a policy executor.

    The client never runs inference or commands a robot. The executor remains the
    source of truth and calls these methods at its existing lifecycle boundaries.
    """

    def __init__(
        self,
        *,
        robot: str,
        policy: str,
        endpoint: str = "tcp://127.0.0.1:5568",
        camera_hz: float = 5.0,
        publisher: ViewerPublisher | None = None,
        control_mode: str = "observe",
    ) -> None:
        if camera_hz < 0:
            raise ValueError("camera_hz must be non-negative")
        if control_mode not in {"observe", "managed"}:
            raise ValueError("control_mode must be 'observe' or 'managed'")
        self.robot = robot
        self.policy = policy
        self.control_mode = control_mode
        self._publisher = publisher or ViewerPublisher(endpoint)
        self._camera_period_s = 0.0 if camera_hz == 0 else 1.0 / camera_hz
        self._last_camera_publish = float("-inf")

    def _event(
        self,
        event: str,
        *,
        step: int = 0,
        chunk_id: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._publisher.publish(
            RuntimeEvent(
                event=event,
                robot=self.robot,
                policy=self.policy,
                step=step,
                chunk_id=chunk_id,
                metadata=dict(metadata or {}),
            )
        )

    def episode_started(
        self,
        *,
        instruction: str,
        max_steps: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        details = dict(metadata or {})
        details.update(
            instruction=instruction,
            max_steps=int(max_steps),
            control_mode=self.control_mode,
        )
        self._event("episode_started", metadata=details)

    def inference_submitted(
        self,
        *,
        step: int,
        chunk_id: int,
        planned_switch_step: int | None = None,
    ) -> None:
        self._event(
            "inference_submitted",
            step=step,
            chunk_id=chunk_id,
            metadata={"planned_switch_step": planned_switch_step},
        )

    def plan_activated(
        self,
        *,
        groups: Mapping[str, np.ndarray],
        action_index: int,
        chunk_id: int,
        step: int,
        action_dt: float,
        inference_ms: float,
        instruction: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._publisher.publish(
            PolicyPlan(
                policy=self.policy,
                instruction=instruction,
                groups=dict(groups),
                action_dt=action_dt,
                inference_ms=max(0.0, inference_ms),
                chunk_id=chunk_id,
                robot=self.robot,
                metadata=dict(metadata or {}),
                start_index=action_index,
            )
        )

    def step_executed(
        self,
        *,
        groups: Mapping[str, np.ndarray],
        cameras: Mapping[str, np.ndarray],
        step: int,
        max_steps: int,
        action_index: int,
        chunk_id: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        now = time.monotonic()
        frames: dict[str, np.ndarray] = {}
        if self._camera_period_s > 0 and (now - self._last_camera_publish >= self._camera_period_s):
            frames = {
                name: np.asarray(frame, dtype=np.uint8).copy() for name, frame in cameras.items()
            }
            if frames:
                self._last_camera_publish = now
        self._publisher.publish(
            RobotSnapshot(
                groups=dict(groups),
                cameras=frames,
                step=step,
                max_steps=max_steps,
                chunk_index=action_index,
                robot=self.robot,
                metadata=dict(metadata or {}),
                active_chunk_id=chunk_id,
            )
        )

    def episode_finished(
        self,
        *,
        reason: str,
        step: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        details = dict(metadata or {})
        details["reason"] = reason
        self._event("episode_finished", step=step, metadata=details)

    def close(self) -> None:
        self._publisher.close()
