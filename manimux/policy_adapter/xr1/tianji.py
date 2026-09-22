"""Decode Xiaomi Robotics 1 Cartesian deltas for Tianji–TacCap.

The checkpoint emits a ``(30, 60)`` packed action.  Both arm poses and
grippers are anchor-relative: every row is reconstructed from the observation
that initiated the request, then solved to the 7-joint Tianji groups.  Waist,
base, and reserved columns have no Tianji counterpart and are deliberately
ignored.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping

import numpy as np

from manimux.kinematics.robot import RobotKinematics
from manimux.kinematics.tianji_diff import rotation_matrix, rotation_vector
from manimux.policies.base import action_interval
from manimux.policy_adapter.base import PolicyAdapter
from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot

ACTION_DIM = 60
ACTION_SEMANTICS = "anchor_relative_ee_delta"
OBSERVATION_PROFILE = "tianji_taccap_two_wrist_black_ego"
GROUP_DIMS = {"left_arm": 8, "right_arm": 8}
# A bent, in-limit pose used only to initialize decoder-process IK state.
WARMUP_JOINTS = np.radians([50.0, -40.0, -30.0, -100.0, -65.0, 0.0, 40.0])
ARM_SLICES = {
    "left_arm": {"position": slice(0, 3), "axis_angle": slice(3, 6), "gripper": 6},
    "right_arm": {"position": slice(8, 11), "axis_angle": slice(11, 14), "gripper": 14},
}


def _state_vector(value: object, *, label: str) -> np.ndarray:
    state = np.asarray(value, dtype=np.float64)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"{label} must contain seven finite joints and one gripper value")
    if not 0.0 <= state[-1] <= 1.0:
        raise ValueError(f"{label} gripper value must be in [0, 1]")
    return np.ascontiguousarray(state)


def _axis_angle_to_rotation(axis_angle: np.ndarray) -> np.ndarray:
    value = np.asarray(axis_angle, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("XR-1 axis-angle delta must be a finite 3-vector")
    return rotation_matrix(value)


class XR1TianjiTacCapAdapter(PolicyAdapter):
    """Convert native XR-1 actions to safe, continuous Tianji joint chunks."""

    supports_context_only_decode = True
    # XR-1 deltas and the first IK seed share the request observation anchor.
    # Timeline owns stale-row trimming after the full source trajectory is decoded.
    decode_seed_source = "observation_state"
    # Each arm can be solved in its own process, but the decoder client commits
    # only after both partitions have completed successfully.
    decode_partitions = tuple(GROUP_DIMS)

    def __init__(self, robot: dict, policy: dict, *, kinematics=None) -> None:
        options = policy["adapter"]
        if kinematics is None and robot.get("config") is not None:
            from manimux.embodiments.robot import RobotModel

            kinematics = RobotModel.from_config(robot["config"]).kinematics
        if not isinstance(kinematics, RobotKinematics):
            raise ValueError("XR-1 Tianji requires the assembled robot kinematics")
        self.kinematics = kinematics
        self.policy = policy
        self._group_order = tuple(options.get("group_order", GROUP_DIMS))
        self._camera_map = dict(options.get("camera_map", {}))
        self._required_cameras = tuple(self._camera_map.values())
        self._horizon_steps = int(policy["horizon_steps"])
        self._action_dt_s = action_interval(policy)
        self._action_dt_ns = round(self._action_dt_s * 1e9)
        validation_dt = float(options.get("ik_validation_dt_s", 0.004))
        if not math.isfinite(validation_dt) or validation_dt <= 0:
            raise ValueError("policy.adapter.ik_validation_dt_s must be finite and positive")
        self._validation_dt_s = validation_dt
        tolerance = float(options.get("gripper_clip_tolerance", 0.0))
        if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
            raise ValueError("policy.adapter.gripper_clip_tolerance must be in [0, 1]")
        self._gripper_clip_tolerance = tolerance

        self.ik_backend = options.get("ik_backend", "analytic")
        if self.ik_backend not in {"analytic", "diff"}:
            raise ValueError("policy.adapter.ik_backend must be analytic or diff")
        if self.ik_backend == "analytic" and options.get("diff_ik"):
            raise ValueError("policy.adapter.diff_ik applies only to ik_backend: diff")
        self.diff_solvers = {}
        if self.ik_backend == "diff":
            from manimux.embodiments.arm.tianji.kinematics import TianjiArmKinematics
            from manimux.kinematics.tianji_diff import (
                DifferentialIKConfig,
                TianjiDifferentialIK,
            )

            config = DifferentialIKConfig.model_validate(options.get("diff_ik", {}))
            if not config.check_j67:
                raise ValueError("XR-1 Tianji differential IK requires the J6/J7 constraint")
            for name, model in self.kinematics.models.items():
                arm = getattr(model, "arm", None)
                if not isinstance(arm, TianjiArmKinematics):
                    raise ValueError("XR-1 differential IK requires Tianji kinematics")
                self.diff_solvers[name] = TianjiDifferentialIK(arm, config)

    def validate(self, robot: dict, policy: dict) -> None:
        if policy["worker"] != "xpolicylab_ws":
            raise ValueError("XR-1 Tianji must use the shared xpolicylab_ws worker")
        if policy.get("action_decoding") not in {"inline", "process"}:
            raise ValueError("XR-1 Tianji action decoding must be inline or process")
        if self.ik_backend == "diff" and policy["action_decoding"] != "process":
            raise ValueError("XR-1 Tianji differential IK requires process action decoding")
        if robot["group_dims"] != GROUP_DIMS or self._group_order != tuple(GROUP_DIMS):
            raise ValueError("XR-1 Tianji requires left_arm/right_arm with 7+1 values")
        if set(self.kinematics.models) != set(GROUP_DIMS):
            raise ValueError("assembled kinematics must provide left_arm and right_arm")
        if self._horizon_steps != 30:
            raise ValueError("the pass-ball XR-1 checkpoint requires a 30-step horizon")
        expected = policy.get("expected_backend") or {}
        identity = expected.get("model", {})
        required = {
            "policy_name": "Xiaomi_Robotics_1",
            "observation_profile": OBSERVATION_PROFILE,
            "output_format": "packed_ee_delta",
            "ego_view_mode": "black",
            "action_semantics": ACTION_SEMANTICS,
        }
        mismatches = {
            name: (identity.get(name), value)
            for name, value in required.items()
            if identity.get(name) != value
        }
        if mismatches:
            raise ValueError(f"XR-1 Tianji backend identity mismatch: {mismatches}")

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._required_cameras if name not in snapshot.frames]
        if missing:
            raise ValueError(f"XR-1 Tianji adapter is missing cameras: {missing}")
        return snapshot

    def prepare_request(self, request: InferenceRequest) -> InferenceRequest:
        if getattr(request, "action_condition", None) is not None:
            raise ValueError("XR-1 Tianji RTC conditioning is not implemented")
        return request

    def _solve_knot(self, model, current, target, gripper, *, diff_solver=None, lag=None):
        current = _state_vector(current, label="IK seed").copy()
        current[-1] = gripper
        start = model.fk(current)
        rotation = rotation_vector(start[:3, :3].T @ target[:3, :3])

        step_dt_s = self._validation_dt_s
        if diff_solver is not None:
            step_dt_s = min(step_dt_s, diff_solver.config.dt_max_s)
        substeps = max(1, math.ceil(self._action_dt_s / step_dt_s))

        for index in range(1, substeps + 1):
            alpha = index / substeps
            waypoint = np.eye(4, dtype=np.float64)
            waypoint[:3, 3] = (1.0 - alpha) * start[:3, 3] + alpha * target[:3, 3]
            waypoint[:3, :3] = start[:3, :3] @ rotation_matrix(alpha * rotation)
            if diff_solver is None:
                result = model.ik(
                    waypoint,
                    current,
                    fixed_coordinates={"gripper": gripper},
                )
                if not result.converged:
                    raise ValueError(f"Tianji IK {result.reason}; rejecting the entire chunk")
                current = _state_vector(result.joints, label="IK result").copy()
            else:
                result = diff_solver.solve(
                    model.flange_target(waypoint, np.array([gripper])),
                    current[:7],
                    self._action_dt_s / substeps,
                )
                if not result.ok:
                    raise ValueError(
                        f"Tianji differential IK {result.reason}; rejecting the entire chunk "
                        f"(lag_mm={result.pos_err_mm}, lag_deg={result.rot_err_deg}, "
                        f"detail={result.detail})"
                    )
                current = np.r_[result.joints, gripper]
                if lag is not None:
                    lag["worst_lag_mm"] = max(lag["worst_lag_mm"], float(result.pos_err_mm))
                    lag["worst_lag_deg"] = max(lag["worst_lag_deg"], float(result.rot_err_deg))
                    lag["lag_exceedances"] += int(result.lag_exceeded)
        return current

    def warmup_decode(self, partition: str | None) -> None:
        """Initialize the selected child process's IK backend without hardware."""

        if partition is None:
            groups = self._group_order
        elif partition in self.decode_partitions:
            groups = (partition,)
        else:
            raise ValueError(f"unknown XR-1 Tianji decode partition {partition!r}")
        for name in groups:
            model = self.kinematics.models[name]
            configuration = np.r_[WARMUP_JOINTS, 0.5]
            target = model.fk(configuration)
            target[:3, 3] += (0.001, 0.0, 0.0)
            solver = self.diff_solvers.get(name)
            if solver is None:
                model.ik(target, configuration, fixed_coordinates={"gripper": 0.5})
            else:
                solver.reset()
                solver.solve(
                    model.flange_target(target, np.array([0.5])),
                    WARMUP_JOINTS.copy(),
                    self._validation_dt_s,
                )
                solver.reset()

    def _actions_and_anchor(
        self, raw: object, context: ActionContext
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        payload = raw.get("actions") if isinstance(raw, Mapping) else raw
        actions = np.asarray(payload, dtype=np.float64)
        if actions.shape != (self._horizon_steps, ACTION_DIM):
            raise ValueError(
                f"XR-1 actions must have shape ({self._horizon_steps}, {ACTION_DIM}), "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("XR-1 actions must be finite")
        observation = context.measured_state
        if observation is None:
            raise ValueError(
                f"XR-1 Tianji has no observation state for request {context.request_seq}"
            )
        if observation.monotonic_ns != context.observation_time_ns:
            raise ValueError(
                "XR-1 Tianji observation state time must equal the request observation time "
                f"({observation.monotonic_ns} != {context.observation_time_ns})"
            )
        anchor = {
            name: _state_vector(observation.groups[name], label=f"observation {name}").copy()
            for name in self._group_order
        }
        return actions, anchor

    def _gripper_targets(
        self, actions: np.ndarray, anchor: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float]:
        targets = {}
        clip_max_abs = 0.0
        tolerance = self._gripper_clip_tolerance
        for name in self._group_order:
            raw = anchor[name][-1] + actions[:, ARM_SLICES[name]["gripper"]]
            invalid = np.flatnonzero((raw < -tolerance) | (raw > 1.0 + tolerance))
            if invalid.size:
                value = float(raw[int(invalid[0])])
                raise ValueError(
                    f"XR-1 {name} gripper target {value} exceeds "
                    f"[0, 1] by more than tolerance {tolerance}"
                )
            clipped = np.clip(raw, 0.0, 1.0)
            targets[name] = clipped
            clip_max_abs = max(clip_max_abs, float(np.max(np.abs(clipped - raw))))
        return targets, clip_max_abs

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        return self._decode(raw, context, self._group_order)

    def decode_action_partition(
        self, raw: object, context: ActionContext, partition: str
    ) -> ActionChunk:
        if partition not in self.decode_partitions:
            raise ValueError(f"unknown XR-1 Tianji decode partition {partition!r}")
        return self._decode(raw, context, (partition,))

    def _decode(
        self, raw: object, context: ActionContext, groups_to_decode: tuple[str, ...]
    ) -> ActionChunk:
        actions, anchor = self._actions_and_anchor(raw, context)
        gripper_targets, gripper_clip_max_abs = self._gripper_targets(actions, anchor)

        groups: dict[str, np.ndarray] = {}
        lag_stats: dict[str, dict[str, float | int]] = {}
        for name in groups_to_decode:
            model = self.kinematics.models[name]
            anchor_state = _state_vector(anchor[name], label=f"anchor {name}")
            anchor_pose = model.fk(anchor_state)
            anchor_rotation = anchor_pose[:3, :3]
            anchor_position = anchor_pose[:3, 3]
            current = anchor_state.copy()
            solver = self.diff_solvers.get(name)
            lag = None
            if solver is not None:
                solver.reset()
                lag = {
                    "worst_lag_mm": 0.0,
                    "worst_lag_deg": 0.0,
                    "lag_exceedances": 0,
                }
            columns = ARM_SLICES[name]
            rows: list[np.ndarray] = []
            for index, row in enumerate(actions):
                gripper = float(gripper_targets[name][index])
                target = np.eye(4, dtype=np.float64)
                target[:3, 3] = anchor_position + anchor_rotation @ row[columns["position"]]
                target[:3, :3] = anchor_rotation @ _axis_angle_to_rotation(
                    row[columns["axis_angle"]]
                )
                current = self._solve_knot(
                    model,
                    current,
                    target,
                    gripper,
                    diff_solver=solver,
                    lag=lag,
                )
                rows.append(current.copy())
            groups[name] = np.ascontiguousarray(np.stack(rows))
            if lag is not None:
                lag_stats[name] = lag

        ignored = actions[:, 16:20]
        return ActionChunk(
            plan_id=f"xr1-tianji-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
            source_offset_steps=0,
            metadata={
                "source_action_semantics": ACTION_SEMANTICS,
                "ignored_waist_base_max_abs": float(np.max(np.abs(ignored))),
                "gripper_clip_tolerance": self._gripper_clip_tolerance,
                "gripper_clip_max_abs": gripper_clip_max_abs,
                "ik_backend": self.ik_backend,
                "ik_seed_source": self.decode_seed_source,
                "ik_seed_time_ns": context.measured_state.monotonic_ns,
                **({"diff_ik_lag": lag_stats} if lag_stats else {}),
            },
        )


def build_adapter(robot: dict, policy: dict, *, kinematics=None) -> XR1TianjiTacCapAdapter:
    return XR1TianjiTacCapAdapter(robot, policy, kinematics=kinematics)
