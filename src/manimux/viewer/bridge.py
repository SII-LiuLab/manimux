from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from manimux.types import ActionChunk, ActionHorizon, RobotState, SensorFrame


@dataclass(frozen=True, slots=True)
class ViewerControl:
    paused: bool
    home_requested: bool = False
    finish_requested: bool = False


class ViewerBridge:
    """Best-effort bridge to the viewer bundled with ManiMux."""

    def __init__(
        self,
        enabled: bool,
        robot_adapter: str,
        group_order: list[str],
        *,
        policy: str = "manimux-local",
        instruction: str = "",
        camera_hz: float = 5.0,
    ) -> None:
        if camera_hz < 0:
            raise ValueError("camera_hz must be non-negative")
        self._enabled = enabled
        self._robot_adapter = robot_adapter
        self._group_order = group_order
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
        from manimux.viewer.protocol import PolicyPlan, RobotSnapshot, RuntimeEvent
        from manimux.viewer.transport import ControlClient, ViewerPublisher

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
                robot=self._robot_adapter,
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
        actions = np.concatenate([groups[name] for name in self._group_order], axis=1)
        action_dt_ns = chunk.dt_ns if committed is None else committed.dt_ns
        plan_metadata: dict[str, object] = {"plan_id": chunk.plan_id}
        if committed is not None:
            plan_metadata["committed_start_time_ns"] = committed.start_time_ns
        plan_metadata.update(metadata or {})
        message = self._policy_plan_type(
            robot=self._robot_adapter,
            policy=self._policy,
            instruction=self._instruction,
            actions=actions,
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
        joint_positions = np.concatenate([state.groups[name] for name in self._group_order])
        now = time.monotonic()
        publish_frames: dict[str, SensorFrame] = {}
        if self._camera_period_s > 0 and now - self._last_camera_publish >= self._camera_period_s:
            publish_frames = frames
            if publish_frames:
                self._last_camera_publish = now
        message = self._snapshot_type(
            robot=self._robot_adapter,
            joint_positions=joint_positions,
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
