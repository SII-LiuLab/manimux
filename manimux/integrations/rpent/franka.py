"""Native RPent Franka tools over the standard ManiMux external-action queue.

This layer owns RPC/action conversion, never an SDK session. State is the real
ManiMux joint vector; it is not RLinf's 19-coordinate wrapped observation.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.kinematics.motion import JointPathPlanner, PoseMotionLimits, pose_error
from manimux.runtime.external import ExternalActionSource


class FrankaEnvironment:
    rpc_methods = (
        "get_robot_state",
        "get_observation",
        "move_delta",
        "rotate_delta",
        "set_gripper",
        "is_started",
        "observe",
    )

    def __init__(self, source: ExternalActionSource, kinematics, options: dict) -> None:
        self.source = source
        if set(kinematics.models) != {"arm"} or kinematics.models["arm"].num_coordinates != 8:
            raise ValueError("RPent Franka requires one arm group: seven joints + scalar gripper")
        self.kinematics = kinematics.models["arm"]
        self.options = options
        self.language = str(options.get("task_language", "Franka primitive execution"))
        self.camera_map = dict(options.get("camera_map", {}))
        if "main" not in self.camera_map or len(set(self.camera_map.values())) != len(
            self.camera_map
        ):
            raise ValueError("Franka camera_map requires main and distinct optional extra streams")
        self.extra_cameras = sorted(set(self.camera_map) - {"main"})
        self.max_age = self._positive("max_frame_age_s", 0.5)
        self.control_dt_s = 1 / self._positive("control_hz", 30.0)
        self.settle_timeout = self._positive("settle_timeout_s", 5.0)
        self.position_tolerance = self._positive("position_tolerance_m", 0.002)
        self.rotation_tolerance = self._positive("rotation_tolerance_rad", 0.02)
        self.gripper_tolerance = self._positive("gripper_tolerance", 0.03)
        self.step_limit = int(options.get("max_episode_steps", 3000))
        self.max_batch = int(options.get("max_batch_actions", 2000))
        if not self.step_limit >= 1 or not 1 <= self.max_batch <= 10000:
            raise ValueError("invalid Franka action budget")
        geometry = options.get("geometry", {})
        defaults = PoseMotionLimits()
        self.planner = JointPathPlanner(
            self.kinematics,
            control_dt_s=self.control_dt_s,
            workspace_min_m=geometry.get("workspace_min_m"),
            workspace_max_m=geometry.get("workspace_max_m"),
            max_waypoints=self.max_batch,
            limits=PoseMotionLimits(**{
                name: self._positive(name, default) for name, default in vars(defaults).items()
            }),
        )
        self.vla_contract = options.get("vla_state_contract")
        if self.vla_contract not in (None, "manimux_joint8_base_delta7_v1"):
            raise ValueError("unsupported Franka VLA state/action contract")
        self.action_scale = np.array(
            [
                self._positive("translation_action_scale_m", 0.005),
                self._positive("rotation_action_scale_rad", 0.025),
                1.0,
            ]
        )
        self.episode_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._busy = self._ready = False
        self._count_start = source.executed
        source.before_tick = self._check_budget

    def _positive(self, name, default):
        value = float(self.options.get(name, default))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    @staticmethod
    def _vec3(value, name):
        result = np.asarray(value, dtype=float)
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must contain three finite values")
        return result

    def _check_budget(self):
        with self._lock:
            if self.source.executed - self._count_start >= self.step_limit:
                self.source.request_stop()

    def is_started(self):
        return (
            self.source.snapshot is not None
            and self.source.failure is None
            and not self.source.shutdown.is_set()
        )

    def _snapshot(self):
        with self.source.condition:
            if self.source.failure:
                raise RuntimeError("ManiMux execution failed") from self.source.failure
            snapshot = self.source.snapshot
            if snapshot is None or self.source.shutdown.is_set():
                raise RuntimeError("ManiMux runtime is unavailable")
            if not 0 <= time.monotonic_ns() - snapshot.state.monotonic_ns <= 500_000_000:
                raise RuntimeError("ManiMux robot snapshot is stale")
            return snapshot

    def _q(self, snapshot=None):
        return (snapshot or self._snapshot()).state.groups["arm"].copy()

    def _reference(self):
        # A new path continues the last submitted reference; its requested TCP
        # goal is still defined from measured state. Never replace observations
        # with command history when tracking error is nonzero.
        with self.source.condition:
            self._snapshot()
            if self.source.last_command is None:
                raise RuntimeError("ManiMux has no submitted command reference")
            return self.source.last_command["arm"].copy()

    def _pose(self, q):
        pose = self.kinematics.fk(q)
        return pose, np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]

    def _status(self):
        return dict(
            episode_id=self.episode_id,
            ready_for_motion=self._ready and not self.source.stopped.is_set(),
            stop_requested=self.source.stopped.is_set(),
            executed_actions=self.source.executed - self._count_start,
            max_episode_steps=self.step_limit,
        )

    def get_env_meta(self):
        return dict(
            ok=True,
            backend="manimux",
            action_dim=7,
            action_scale=self.action_scale.tolist(),
            action_dt_s=self.control_dt_s,
            use_relative_frame=False,
            state_contract="manimux_joint8_base_delta7_v1",
            state_layout=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
                "gripper_opening",
            ],
            native_rlinf_state_compatible=False,
            vla_enabled=self.vla_contract is not None,
            reset_semantics="bookkeeping/rearm only; no physical reset",
            primitives=["move_delta", "rotate_delta", "set_gripper"],
            **self._status(),
        )

    def get_task_language(self):
        return self.language

    def get_robot_state(self):
        snapshot = self._snapshot()
        q = self._q(snapshot)
        _, pose = self._pose(q)
        return dict(
            raw_base_state=dict(tcp_pose=pose, joint_positions=q[:7], gripper_opening=float(q[7])),
            joint_position=q,
            timestamp_ns=snapshot.state.monotonic_ns,
            sequence=snapshot.state.sequence,
            episode_status=self._status(),
            action_dim=7,
            action_scale=self.action_scale.tolist(),
            use_relative_frame=False,
        )

    def _observation(self, snapshot, *, historical=False):
        frames = {}
        for alias, stream in self.camera_map.items():
            frame = snapshot.frames[stream]
            if (
                not historical
                and not 0
                <= (time.monotonic_ns() - frame.capture_monotonic_ns) / 1e9
                <= self.max_age
            ):
                raise RuntimeError(f"{stream}: stale camera capture")
            frames[alias] = frame
        output = dict(
            main_images=frames["main"].data.copy(),
            states=self._q(snapshot),
            state_contract="manimux_joint8_base_delta7_v1",
            state_timestamp_ns=snapshot.state.monotonic_ns,
            camera_timestamps_ns={k: v.capture_monotonic_ns for k, v in frames.items()},
        )
        if frames["main"].depth is not None:
            output["main_depths"] = frames["main"].depth.copy()
        if self.extra_cameras:
            output["extra_view_images"] = np.stack([frames[k].data for k in self.extra_cameras])
            if all(frames[k].depth is not None for k in self.extra_cameras):
                output["extra_view_depths"] = np.stack(
                    [frames[k].depth for k in self.extra_cameras]
                )
        return output

    def get_observation(self):
        return self._observation(self._snapshot())

    observe = get_observation

    def get_camera_meta(self):
        snapshot = self._snapshot()
        self._observation(snapshot)  # Apply the same freshness check.
        return dict(
            cameras={
                stream: snapshot.frames[stream].camera_meta for stream in self.camera_map.values()
            },
            observation_camera_map=self.camera_map.copy(),
            state_timestamp_ns=snapshot.state.monotonic_ns,
            camera_timestamps_ns={
                name: snapshot.frames[stream].capture_monotonic_ns
                for name, stream in self.camera_map.items()
            },
        )

    def render_camera(self, camera_name, *, depth=False):
        snapshot = self._snapshot()
        self._observation(snapshot)
        frame = snapshot.frames[self.camera_map[camera_name]]
        return frame.depth if depth else frame.data

    @contextmanager
    def _request(self, *, reset=False):
        with self._lock:
            if self._busy:
                raise RuntimeError("another Franka request is active")
            if not reset and (not self._ready or self.source.stopped.is_set()):
                raise RuntimeError("Franka episode is stopped; explicit reset/rearm required")
            self._busy = True
        try:
            yield
        except BaseException:
            self.source.request_stop()
            raise
        finally:
            with self._lock:
                self._busy = False

    def reset(self):
        with self._request(reset=True):
            observation = self.get_observation()
            with self._lock:
                self.source.arm()
                self.episode_id = uuid.uuid4().hex
                self._count_start = self.source.executed
                self._ready = True
            self.source.event("external_episode_started", **self._status())
            return dict(ok=True, states=observation["states"], robot_state=self.get_robot_state())

    def _submit(self, path):
        if len(path) > self.max_batch:
            raise ValueError("Franka path exceeds max_batch_actions; no waypoints were dropped")
        if self.source.executed - self._count_start + len(path) > self.step_limit:
            raise ValueError("Franka path exceeds remaining episode budget")
        self.planner.validate(path, self._reference())
        batch = self.source.submit({"arm": path})
        batch.future.result(timeout=len(path) * self.control_dt_s + self.settle_timeout)
        self.source.event(
            "external_call_finished",
            episode_id=self.episode_id,
            requested_actions=len(path),
            executed_actions=batch.executed,
        )
        return batch

    def _plan_pose(self, q, target):
        reference = self._reference()
        return self.planner.plan(
            target, reference, fixed_coordinates={"gripper": float(reference[7])}
        )

    def _wait_arrival(self, target, gripper=None):
        deadline = time.monotonic() + self.settle_timeout
        contact_since = None
        while True:
            snapshot = self._snapshot()
            q = self._q(snapshot)
            pose, pose_vector = self._pose(q)
            position_error, rotation_error = pose_error(target, pose)
            gripper_error = None if gripper is None else abs(float(q[7]) - gripper)
            feedback = snapshot.state.grippers.get("arm")
            closing_contact = (
                gripper == 0.0 and feedback is not None
                and feedback.contact == "closing" and feedback.target_opening == 0.0
            )
            if not closing_contact:
                contact_since = None
            elif contact_since is None:
                contact_since = feedback.timestamp
            contact_confirmed = (
                closing_contact and feedback.timestamp - contact_since >= 0.1
            )
            reached = (
                position_error <= self.position_tolerance
                and rotation_error <= self.rotation_tolerance
                and (
                    gripper_error is None or gripper_error <= self.gripper_tolerance
                    or contact_confirmed
                )
            )
            result = (
                q, pose_vector, position_error, rotation_error, gripper_error, contact_confirmed,
            )
            if self.source.stopped.is_set():
                return (False, *result)
            if reached:
                return (True, *result)
            if time.monotonic() >= deadline:
                self.source.request_stop()
                return (False, *result)
            with self.source.condition:
                self.source.condition.wait(timeout=0.02)

    def _move(self, delta, *, rotate):
        requested = self._vec3(delta, "delta_rpy" if rotate else "delta_xyz")
        with self._request():
            start = self._q()
            target, start_pose = self._pose(start)
            if rotate:
                target[:3, :3] = (
                    Rotation.from_euler("xyz", requested) * Rotation.from_matrix(target[:3, :3])
                ).as_matrix()
            else:
                target[:3, 3] += requested
            batch = self._submit(self._plan_pose(start, target))
            ok, q, final, pe, re, _, _ = self._wait_arrival(target)
            return dict(
                ok=ok and batch.executed == len(batch.groups["arm"]),
                start_tcp_pose=start_pose,
                final_tcp_pose=final,
                final_error_m=pe,
                final_error_rad=re,
                states=q,
                steps_used=batch.executed,
                episode_id=self.episode_id,
                **{"requested_delta_rpy_base" if rotate else "requested_delta_xyz_base": requested},
            )

    def move_delta(self, delta_xyz):
        return self._move(delta_xyz, rotate=False)

    def rotate_delta(self, delta_rpy):
        return self._move(delta_rpy, rotate=True)

    def set_gripper(self, *, open: bool):
        if not isinstance(open, bool):
            raise ValueError("open must be boolean")
        with self._request():
            q = self._reference()
            target, _ = self._pose(q)
            goal = float(open)
            q[7] = goal
            batch = self._submit(q[None])
            ok, q, final, pe, re, ge, contact = self._wait_arrival(target, gripper=goal)
            return dict(
                ok=ok and batch.executed == 1,
                gripper_opening=float(q[7]),
                requested_open=open,
                final_gripper_error=ge,
                closing_contact_confirmed=contact,
                steps_used=batch.executed,
                states=q,
                final_tcp_pose=final,
                final_error_m=pe,
                final_error_rad=re,
            )

    def chunk_step(self, actions, *, return_all_frames=False):
        if self.vla_contract is None:
            raise NotImplementedError(
                "Native RLinf 19-state VLA is not compatible with ManiMux joint8 observations; "
                "select vla_state_contract only for a model trained for "
                "manimux_joint8_base_delta7_v1"
            )
        actions = np.array(actions, dtype=float, copy=True)
        if (
            actions.ndim != 2
            or actions.shape[1] != 7
            or not 1 <= len(actions) <= self.max_batch
            or not np.isfinite(actions).all()
            or np.max(np.abs(actions)) > 1
        ):
            raise ValueError("Franka VLA requires a bounded finite Nx7 array in [-1,1]")
        with self._request():
            q = self._q()
            path = []
            for action in actions:
                target, _ = self._pose(q)
                target[:3, 3] += action[:3] * self.action_scale[0]
                target[:3, :3] = (
                    Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
                    * Rotation.from_matrix(target[:3, :3])
                ).as_matrix()
                answer = self.kinematics.ik(
                    target, q, fixed_coordinates={"gripper": float(action[6] > 0)}
                )
                if not answer.converged:
                    raise ValueError("Franka VLA chunk has no bounded IK solution")
                q = answer.joints
                path.append(q.copy())
            batch = self._submit(np.array(path))
            observation = self.get_observation()
            if return_all_frames:
                observation = [self._observation(s, historical=True) for s in batch.feedback]
            return dict(
                observation=observation,
                terminated=False,
                truncated=self.source.stopped.is_set(),
                info=dict(
                    requested_actions=len(actions),
                    executed_actions=batch.executed,
                    episode_id=self.episode_id,
                ),
            )

    def step(self, *args, **kwargs):
        raise NotImplementedError("Use native Franka primitives or explicit chunk_step")

    def request_stop(self):
        self.source.request_stop()

    def close(self):
        self.source.request_stop()
        self.source.shutdown.set()
