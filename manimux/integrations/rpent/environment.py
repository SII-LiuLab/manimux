"""RPent's native YAM environment, with one ManiMux hardware writer."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from robots.yam.contracts import YAM_CAMERA_NAMES, env_runtime_contract, validate_actions
from robots.yam.geometry import enforce_hard_limits
from robots.yam.operator_control import read_receipt
from robots.yam.tasks import resolve_task_language

from manimux.integrations.rpent.geometry import build_geometry
from manimux.kinematics.robot import RobotKinematics
from manimux.runtime.external import ExternalActionSource
from manimux.types import ObservationSnapshot


class YamEnvironment:
    def __init__(
        self,
        source: ExternalActionSource,
        kinematics: RobotKinematics,
        options: dict[str, Any],
        *,
        geometry=None,
    ) -> None:
        self.source = source
        self.geometry = (
            geometry
            if geometry is not None
            else build_geometry(
                kinematics,
                options.get("geometry", {}),
            )
        )
        self.task_name = options["task_name"]
        self.language = str(resolve_task_language(self.task_name, options.get("task_language")))
        self.seed = int(options.get("seed", 0))
        self.step_limit = int(options.get("max_episode_steps", 1000))
        self.max_batch_actions = int(options.get("max_batch_actions", 2000))
        if self.step_limit < 1 or not 1 <= self.max_batch_actions <= 10000:
            raise ValueError("invalid episode or batch action budget")
        self.receipt_path = Path(options["operator_receipt_path"]).expanduser()
        self.camera_map = options["camera_map"]
        if set(self.camera_map) != set(YAM_CAMERA_NAMES) or len(set(self.camera_map.values())) != 3:
            raise ValueError("camera_map must map top/left/right to three distinct streams")
        self.max_frame_age_s = float(options.get("max_frame_age_s", 0.5))
        self.max_camera_state_skew_s = float(options.get("max_camera_state_skew_s", 0.1))
        if not all(
            np.isfinite(v) and v > 0
            for v in (
                self.max_frame_age_s,
                self.max_camera_state_skew_s,
            )
        ):
            raise ValueError("observation age and camera/state skew limits must be positive")
        self.episode_id = uuid.uuid4().hex
        self.ready_receipt: dict[str, Any] | None = None
        self.success_receipt: dict[str, Any] | None = None
        self.terminal_event: str | None = None
        self._consumed: set[str] = set()
        self._count_start = source.executed
        self._lock = threading.RLock()
        self._geometry_lock = threading.RLock()
        self._active_request = False
        self._results: dict[tuple[str, str], tuple[str, tuple]] = {}
        source.before_tick = self._poll_receipt

    def _poll_receipt(self) -> None:
        with self._lock:
            if self.source.executed - self._count_start >= self.step_limit:
                self.source.request_stop()
            receipt = read_receipt(self.receipt_path)
            if receipt is None or receipt["request_id"] in self._consumed:
                return
            if receipt["episode_id"] != self.episode_id:
                return
            event = receipt["event"]
            if event in {"success", "failure", "abort"} and self.terminal_event is None:
                self._consumed.add(receipt["request_id"])
                self.terminal_event = event
                if event == "success":
                    self.success_receipt = receipt
                self.source.request_stop()

    def _pending_ready_receipt(self) -> dict[str, Any] | None:
        receipt = read_receipt(self.receipt_path)
        if (
            receipt is not None
            and receipt["event"] == "ready"
            and receipt["episode_id"] == self.episode_id
            and receipt["request_id"] not in self._consumed
        ):
            return dict(receipt)
        return None

    def _status(self) -> dict[str, Any]:
        return {
            "ready_to_reset": not self._active_request
            and self._pending_ready_receipt() is not None,
            "eval_success": self.terminal_event == "success",
            "eval_success_source": "operator_local" if self.success_receipt else "operator_unset",
            "terminal_event": self.terminal_event,
            "stop_requested": self.source.stopped.is_set(),
            "take_action_cnt": self.source.executed - self._count_start,
            "step_lim": self.step_limit,
            "actual_seed": self.seed,
            "episode_id": self.episode_id,
            "ready_for_motion": self.ready_receipt is not None and not self.source.stopped.is_set(),
            "operator_ready_receipt": deepcopy(self.ready_receipt),
            "operator_success_receipt": deepcopy(self.success_receipt),
        }

    def get_env_meta(self) -> dict[str, Any]:
        meta = env_runtime_contract(
            task_name=self.task_name,
            seed=self.seed,
            max_episode_steps=self.step_limit,
        )
        meta["execution"].update(
            compact_control=True,
            joint_limit_min=self.geometry.lower.tolist(),
            joint_limit_max=self.geometry.upper.tolist(),
        )
        return meta

    def is_started(self) -> bool:
        return (
            self.source.snapshot is not None
            and self.source.failure is None
            and not self.source.shutdown.is_set()
        )

    def get_task_language(self) -> str:
        return self.language

    def _snapshot(self) -> ObservationSnapshot:
        with self.source.condition:
            snapshot = self.source.snapshot
            if self.source.failure:
                raise RuntimeError("ManiMux execution failed") from self.source.failure
            if snapshot is None or self.source.shutdown.is_set():
                raise RuntimeError("ManiMux runtime is not started")
            if not 0 <= time.monotonic_ns() - snapshot.state.monotonic_ns <= 500_000_000:
                raise RuntimeError("ManiMux robot snapshot is stale")
            return snapshot

    @staticmethod
    def _qpos(snapshot):
        return np.r_[snapshot.state.groups["left_arm"], snapshot.state.groups["right_arm"]]

    def _control_info(self, snapshot: ObservationSnapshot) -> dict[str, Any]:
        with self._geometry_lock:
            robot_state = self.geometry.robot_state(self._qpos(snapshot))
        with self.source.condition:
            command = self.source.last_command
            commanded = (
                self._qpos(snapshot)
                if command is None
                else np.r_[
                    command["left_arm"],
                    command["right_arm"],
                ]
            )
        with self._lock:
            return {
                "episode_status": self._status(),
                "robot_state": robot_state,
                "commanded_qpos": commanded.copy(),
            }

    def _observation(self, snapshot: ObservationSnapshot, *, historical=False) -> tuple[dict, dict]:
        views = {}
        for name, stream in self.camera_map.items():
            frame = snapshot.frames[stream]
            if frame.depth is None or frame.camera_meta is None:
                raise ValueError(f"{stream}: RPent requires RGB-D observations")
            age = (time.monotonic_ns() - frame.capture_monotonic_ns) / 1e9
            if not historical and not 0 <= age <= self.max_frame_age_s:
                raise ValueError(f"{stream}: stale RGB-D frame")
            skew = abs(frame.capture_monotonic_ns - snapshot.state.monotonic_ns) / 1e9
            with self._geometry_lock:
                meta = self.geometry.camera_meta_for_qpos(
                    name,
                    deepcopy(frame.camera_meta),
                    self._qpos(snapshot),
                )
            meta.update(
                capture_monotonic_ns=frame.capture_monotonic_ns,
                state_monotonic_ns=snapshot.state.monotonic_ns,
                camera_state_skew_s=skew,
                projection_valid=name == "top" or skew <= self.max_camera_state_skew_s,
            )
            if not meta["projection_valid"]:
                meta["projection_limitation"] = "Wrist image and measured state exceed allowed skew"
            views[name] = {
                "rgb": frame.data.copy(),
                "depth": frame.depth.copy(),
                "camera_meta": meta,
            }
        return {
            "frames": {name: view["rgb"] for name, view in views.items()},
            "views": views,
            "state": {"joint_position": self._qpos(snapshot)},
            "snapshot_id": str(snapshot.state.sequence),
        }, self._control_info(snapshot)

    def observe(self) -> tuple[dict, dict]:
        self._poll_receipt()
        return self._observation(self._snapshot())

    def read_control_state(self) -> tuple[dict, dict]:
        self._poll_receipt()
        snapshot = self._snapshot()
        return {"state": {"joint_position": self._qpos(snapshot)}}, self._control_info(snapshot)

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        if options:
            raise ValueError("reset options are not supported")
        with self._lock:
            if self._active_request:
                raise RuntimeError("reset rejected while an action response is pending")
            self._active_request = True
        try:
            observation, info = self.observe()
            with self._lock:
                receipt = self._pending_ready_receipt()
                if receipt is None:
                    raise RuntimeError(
                        "reset requires a new local ready receipt for the current episode"
                    )
                self.source.arm()
                self._consumed.add(receipt["request_id"])
                self.episode_id = uuid.uuid4().hex
                self.ready_receipt = {**receipt, "ready_for_episode_id": self.episode_id}
                self.terminal_event = None
                self.success_receipt = None
                self._count_start = self.source.executed
                if seed is not None:
                    self.seed = int(seed)
                info["episode_status"] = self._status()
                self.source.event("external_episode_started", **info["episode_status"])
                return observation, info
        finally:
            with self._lock:
                self._active_request = False

    def plan_arm_path(self, arm, target_pose):
        snapshot = self._snapshot()
        with self._geometry_lock:
            return self.geometry.plan_arm_path(
                arm, target_pose, current_qpos14=self._qpos(snapshot)
            )

    def step(self, action, *, action_type="qpos", expected_episode_id=None):
        actions = validate_actions(action, action_type=action_type)
        if len(actions) != 1:
            raise ValueError("step requires exactly one action")
        return self.chunk_step(
            actions, action_type=action_type, expected_episode_id=expected_episode_id
        )

    def control_step(self, action, *, expected_episode_id=None):
        return self.step(
            action,
            expected_episode_id=expected_episode_id,
        )

    def chunk_step(
        self,
        actions,
        *,
        action_type="qpos",
        expected_episode_id=None,
        return_all_frames=False,
        request_id=None,
    ):
        array = validate_actions(actions, action_type=action_type).copy()
        if len(array) > self.max_batch_actions:
            raise ValueError("action path exceeds max_batch_actions; no waypoints were dropped")
        if request_id is not None and (
            not isinstance(request_id, str) or not 1 <= len(request_id) <= 128
        ):
            raise ValueError("request_id must contain 1..128 characters")
        fingerprint = hashlib.sha256(array.tobytes() + str(return_all_frames).encode()).hexdigest()
        key = (expected_episode_id, request_id)
        with self._lock:
            if request_id is not None and key in self._results:
                digest, result = self._results[key]
                if digest != fingerprint:
                    raise ValueError("request_id already used with different actions")
                return deepcopy(result)
            self._poll_receipt()
            if expected_episode_id != self.episode_id:
                raise ValueError("expected_episode_id does not match the active episode")
            if self._active_request:
                raise RuntimeError("another environment request is active")
            if self.ready_receipt is None or self.terminal_event is not None:
                raise RuntimeError("episode is not ready for motion")
            if request_id is not None and len(self._results) >= 128:
                raise RuntimeError(
                    "idempotent request retention limit reached; restart the runtime"
                )
            self._active_request = True
        batch = None
        try:
            previous = self._qpos(self._snapshot())
            remaining = max(0, self.step_limit - (self.source.executed - self._count_start))
            selected = array[:remaining]
            with self._geometry_lock:
                for action in selected:
                    enforce_hard_limits(action, self.geometry.lower, self.geometry.upper)
                    checked = self.geometry.check_qpos_transition(previous, action)
                    if not checked["ok"]:
                        raise ValueError(f"action path rejected: {checked}")
                    previous = action
            batch = (
                None
                if not len(selected)
                else self.source.submit(
                    {"left_arm": selected[:, :7], "right_arm": selected[:, 7:]},
                )
            )
            if batch is not None:
                batch.future.result(timeout=len(selected) / 30 + 5)
            # Keep the admitted episode reserved through the final response snapshot.
            obs, info = self.observe()
            executed = 0 if batch is None else batch.executed
            info.update(requested_actions=len(array), executed_actions=executed)
            status = info["episode_status"]
            terminated = status["eval_success"]
            truncated = bool(
                status["terminal_event"] in {"failure", "abort"}
                or status["stop_requested"]
                or status["take_action_cnt"] >= self.step_limit
            )
            payload = obs
            if return_all_frames and batch is not None and batch.feedback:
                # Old captures keep their own timestamps and geometry, not a later observation.
                payload = [self._observation(frame, historical=True)[0] for frame in batch.feedback]
            result = (payload, float(terminated), terminated, truncated, info)
            if request_id is not None:
                with self._lock:
                    self._results[key] = (fingerprint, deepcopy(result))
            self.source.event(
                "external_call_finished",
                episode_id=expected_episode_id,
                request_id=request_id,
                requested_actions=len(array),
                executed_actions=executed,
            )
            return result
        except BaseException:
            self.source.request_stop()
            raise
        finally:
            with self._lock:
                self._active_request = False

    def render_camera(self, camera_name, *, depth=False):
        obs, _ = self.observe()
        return obs["views"][camera_name]["depth" if depth else "rgb"]

    def get_camera_meta(self, camera_name):
        obs, _ = self.observe()
        return obs["views"][camera_name]["camera_meta"]

    def request_stop(self) -> None:
        self.source.request_stop()

    def close(self) -> None:
        self.source.request_stop()
        self.source.shutdown.set()
