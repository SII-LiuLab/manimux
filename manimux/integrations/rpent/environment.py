"""RPent's YAM environment contract, with hardware owned by ManiMux."""

from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from robots.yam.contracts import YAM_CAMERA_NAMES, env_runtime_contract, validate_actions
from robots.yam.operator_control import read_receipt
from robots.yam.tasks import resolve_task_language
from scipy.spatial.transform import Rotation

from manimux.kinematics.robot import RobotKinematics
from manimux.runtime.external import ExternalActionSource
from manimux.types import ObservationSnapshot


class YamEnvironment:
    def __init__(
        self, source: ExternalActionSource, kinematics: RobotKinematics, options: dict[str, Any]
    ) -> None:
        self.source = source
        self.kinematics = kinematics
        self.task_name = options["task_name"]
        self.language = str(resolve_task_language(self.task_name, options.get("task_language")))
        self.seed = int(options.get("seed", 0))
        self.step_limit = int(options.get("max_episode_steps", 1000))
        if self.step_limit < 1:
            raise ValueError("max_episode_steps must be positive")
        self.receipt_path = Path(options["operator_receipt_path"]).expanduser()
        self.camera_map = options["camera_map"]
        if set(self.camera_map) != set(YAM_CAMERA_NAMES) or len(set(self.camera_map.values())) != 3:
            raise ValueError("camera_map must map top/left/right to three distinct streams")
        self.episode_id = uuid.uuid4().hex
        self.ready_receipt: dict[str, Any] | None = None
        self.success_receipt: dict[str, Any] | None = None
        self.terminal_event: str | None = None
        self._consumed: set[str] = set()
        self._count_start = source.executed
        self._lock = threading.RLock()
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
            "ready_to_reset": self._pending_ready_receipt() is not None,
            "eval_success": self.terminal_event == "success",
            "eval_success_source": "operator_local" if self.success_receipt else "operator_unset",
            "terminal_event": self.terminal_event,
            "stop_requested": self.source.stopped.is_set(),
            "take_action_cnt": self.source.executed - self._count_start,
            "step_lim": self.step_limit,
            "actual_seed": self.seed,
            "episode_id": self.episode_id,
            "ready_for_motion": self.ready_receipt is not None,
            "operator_ready_receipt": deepcopy(self.ready_receipt),
            "operator_success_receipt": deepcopy(self.success_receipt),
        }

    def get_env_meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = env_runtime_contract(
            task_name=self.task_name,
            seed=self.seed,
            max_episode_steps=self.step_limit,
        )
        meta["execution"]["compact_control"] = True
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
            if time.monotonic_ns() - snapshot.state.monotonic_ns > 500_000_000:
                raise RuntimeError("ManiMux robot snapshot is stale")
            return snapshot

    def _observation(self, snapshot: ObservationSnapshot) -> tuple[dict[str, Any], dict[str, Any]]:
        groups = snapshot.state.groups
        qpos = np.concatenate([groups["left_arm"], groups["right_arm"]])
        views = {}
        for name, stream in self.camera_map.items():
            frame = snapshot.frames[stream]
            if frame.depth is None or frame.camera_meta is None:
                raise ValueError(f"{stream}: RPent requires RGB-D observations")
            meta = deepcopy(frame.camera_meta)
            # VLA-only integration deliberately does not advertise a calibrated
            # world map. Camera snapshots remain useful without inventing an
            # inter-arm transform or wrist hand-eye calibration.
            meta.update(
                cam2world_cv=np.full((4, 4), np.nan),
                projection_valid=False,
                projection_limitation=(
                    "Calibrated world projection is unavailable in the VLA integration"
                ),
                capture_monotonic_ns=frame.capture_monotonic_ns,
            )
            views[name] = {
                "rgb": frame.data.copy(),
                "depth": frame.depth.copy(),
                "camera_meta": meta,
            }
        obs = {
            "frames": {name: view["rgb"] for name, view in views.items()},
            "views": views,
            "state": {"joint_position": qpos.copy()},
            "snapshot_id": str(snapshot.state.sequence),
        }
        return obs, self._control_info(snapshot)

    def _control_info(self, snapshot: ObservationSnapshot) -> dict[str, Any]:
        groups = snapshot.state.groups
        qpos = np.r_[groups["left_arm"], groups["right_arm"]]
        poses = self.kinematics.fk(groups)
        robot_state = {"qpos": qpos, "world_frame": "left_base"}
        for side in ("left", "right"):
            pose = poses[f"{side}_arm"]
            if side == "left":
                quat = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
                robot_state[f"{side}_eef_pose"] = np.r_[pose[:3, 3], quat]
            else:
                robot_state[f"{side}_eef_pose"] = None
        robot_state["right_eef_pose_limitation"] = "right-to-left base calibration is not supplied"
        command = self.source.last_command
        info = {
            "episode_status": self._status(),
            "robot_state": robot_state,
            "commanded_qpos": None
            if command is None
            else np.r_[command["left_arm"], command["right_arm"]],
        }
        return info

    def observe(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self._poll_receipt()
        return self._observation(self._snapshot())

    def read_control_state(self) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._lock:
            self._poll_receipt()
            snapshot = self._snapshot()
            qpos = np.r_[snapshot.state.groups["left_arm"], snapshot.state.groups["right_arm"]]
            return {"state": {"joint_position": qpos}}, self._control_info(snapshot)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if options:
            raise ValueError("reset options are not supported")
        with self._lock:
            self._snapshot()
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
            return self.observe()

    def step(
        self, action: Any, *, action_type: str = "qpos", expected_episode_id: str | None = None
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        actions = validate_actions(action, action_type=action_type)
        if len(actions) != 1:
            raise ValueError("step requires exactly one action")
        return self.chunk_step(
            actions, action_type=action_type, expected_episode_id=expected_episode_id
        )

    def control_step(
        self, action: Any, *, expected_episode_id: str | None = None
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self.step(action, expected_episode_id=expected_episode_id)

    def chunk_step(
        self,
        actions: Any,
        *,
        action_type: str = "qpos",
        expected_episode_id: str | None = None,
        return_all_frames: bool = False,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        array = validate_actions(actions, action_type=action_type)
        if len(array) > 30:
            raise ValueError("YAM VLA batches must contain at most 30 actions")
        with self._lock:
            self._poll_receipt()
            if expected_episode_id != self.episode_id:
                raise ValueError("expected_episode_id does not match the active episode")
            if self.ready_receipt is None or self.terminal_event is not None:
                raise RuntimeError("episode is not ready for motion")
            remaining = max(0, self.step_limit - (self.source.executed - self._count_start))
            selected = array[:remaining]
            batch = (
                None
                if not len(selected)
                else self.source.submit(
                    {
                        "left_arm": selected[:, :7],
                        "right_arm": selected[:, 7:],
                    }
                )
            )
        if batch is not None:
            try:
                batch.future.result(timeout=len(selected) / 30 + 5)
            except BaseException:
                self.source.request_stop()
                raise
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
        payload: dict[str, Any] | list[dict[str, Any]] = obs
        if return_all_frames and batch is not None and batch.feedback:
            payload = [self._observation(snapshot)[0] for snapshot in batch.feedback]
        return payload, float(terminated), terminated, truncated, info

    def render_camera(self, camera_name: str, *, depth: bool = False) -> Any:
        obs, _ = self.observe()
        return obs["views"][camera_name]["depth" if depth else "rgb"]

    def get_camera_meta(self, camera_name: str) -> Any:
        obs, _ = self.observe()
        return obs["views"][camera_name]["camera_meta"]

    def request_stop(self) -> None:
        self.source.request_stop()

    def close(self) -> None:
        self.source.request_stop()
        self.source.shutdown.set()
