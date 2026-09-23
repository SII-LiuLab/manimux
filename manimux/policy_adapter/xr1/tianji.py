"""Tianji FK/IK boundary for Xiaomi Robotics 1's standard absolute EE actions.

The checkpoint natively emits ``(30, 60)`` anchor-relative deltas.  XPolicyLab
owns that transform: this adapter sends the observed per-arm base TCP poses,
receives absolute ``[xyz, quat_wxyz]`` targets and gripper apertures, and
solves them to the 7-joint Tianji groups.  RTC conditions travel in the same
absolute layout and are made anchor-relative by the model server.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from manimux.kinematics.robot import RobotKinematics
from manimux.kinematics.tianji_diff import rotation_matrix, rotation_vector
from manimux.policies.base import action_interval
from manimux.policies.xpolicylab.codec import matrix_pose, pose_matrix
from manimux.policy_adapter.base import PolicyAdapter
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot

ACTION_SEMANTICS = "absolute_per_arm_base_xyz_wxyz"
NATIVE_ACTION_SEMANTICS = "anchor_relative_ee_delta"
OBSERVATION_PROFILE = "tianji_taccap_two_wrist_black_ego"
GROUP_DIMS = {"left_arm": 8, "right_arm": 8}
GROUP_SIDES = {"left_arm": "left", "right_arm": "right"}
# A bent, in-limit pose used only to initialize decoder-process IK state.
WARMUP_JOINTS = np.radians([50.0, -40.0, -30.0, -100.0, -65.0, 0.0, 40.0])


@dataclass(slots=True)
class XR1TianjiRequest(RtcInferenceRequest):
    xpolicylab_state: dict | None = None


def _state_vector(value: object, *, label: str) -> np.ndarray:
    state = np.asarray(value, dtype=np.float64)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"{label} must contain seven finite joints and one gripper value")
    if not 0.0 <= state[-1] <= 1.0:
        raise ValueError(f"{label} gripper value must be in [0, 1]")
    return np.ascontiguousarray(state)


def _pose_row(model, state: np.ndarray) -> np.ndarray:
    """Return one absolute ``[xyz, quat_wxyz, gripper]`` row from Tianji FK."""

    return np.r_[matrix_pose(model.fk(state)), state[-1]]


class XR1TianjiTacCapAdapter(PolicyAdapter):
    """Convert native XR-1 actions to safe, continuous Tianji joint chunks."""

    supports_context_only_decode = True
    # XR-1 targets are restored against the request observation, which also
    # seeds IK.
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
            "output_format": "xpolicylab",
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

    def prepare_request(self, request: InferenceRequest) -> XR1TianjiRequest:
        groups = request.observation.state.groups
        extra = {}
        for name in self._group_order:
            state = _state_vector(groups[name], label=f"observation {name}")
            extra[f"{GROUP_SIDES[name]}_ee_pose"] = matrix_pose(
                self.kinematics.models[name].fk(state)
            )
        condition = getattr(request, "action_condition", None)
        weights = getattr(request, "condition_weights", None)
        if condition is not None:
            condition, weights = self._absolute_condition(condition, weights)
        return XR1TianjiRequest(
            session_id=request.session_id,
            request_seq=request.request_seq,
            observation_time_ns=request.observation_time_ns,
            deadline_ns=request.deadline_ns,
            observation=request.observation,
            instruction=request.instruction,
            action_condition=condition,
            condition_weights=weights,
            rtc_beta=getattr(request, "rtc_beta", 5.0),
            xpolicylab_state=extra,
        )

    def _absolute_condition(
        self, condition: object, weights: object
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode Tianji joint waypoints as absolute dual TCP/gripper rows."""

        condition = np.asarray(condition, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        width = sum(GROUP_DIMS.values())
        if condition.shape != (self._horizon_steps, width) or weights.shape != (
            self._horizon_steps,
        ):
            raise ValueError(
                f"XR-1 Tianji RTC condition must have shape ({self._horizon_steps}, {width}) "
                f"with ({self._horizon_steps},) weights, got {condition.shape} and "
                f"{weights.shape}"
            )
        if not np.isfinite(condition).all() or not np.isfinite(weights).all():
            raise ValueError("XR-1 Tianji RTC condition must be finite")
        poses = np.zeros_like(condition)
        for row, weight in enumerate(weights):
            # Zero-weight rows carry no guidance; the model leaves them at zero delta.
            if weight == 0:
                continue
            offset = 0
            for name in self._group_order:
                state = _state_vector(
                    condition[row, offset : offset + GROUP_DIMS[name]],
                    label=f"condition target {name}",
                )
                poses[row, offset : offset + GROUP_DIMS[name]] = _pose_row(
                    self.kinematics.models[name], state
                )
                offset += GROUP_DIMS[name]
        return poses, weights

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

    def _steps(self, raw: object, context: ActionContext) -> Sequence[Mapping]:
        # The WebSocket client unwraps a plain response to its action list.
        steps = raw.get("actions") if isinstance(raw, Mapping) else raw
        if (
            not isinstance(steps, Sequence)
            or isinstance(steps, str | bytes)
            or len(steps) != self._horizon_steps
            or not all(isinstance(step, Mapping) for step in steps)
        ):
            raise ValueError(
                f"XR-1 Tianji expects {self._horizon_steps} standard EE action dictionaries"
            )
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
        return steps

    def _gripper_targets(self, steps: Sequence[Mapping]) -> tuple[dict[str, np.ndarray], float]:
        targets = {}
        clip_max_abs = 0.0
        tolerance = self._gripper_clip_tolerance
        for name in self._group_order:
            key = f"{GROUP_SIDES[name]}_ee_joint_state"
            values = [np.asarray(step[key], dtype=np.float64) for step in steps]
            if any(value.shape != (1,) for value in values):
                raise ValueError(f"XR-1 {key} must contain one gripper value")
            raw = np.concatenate(values)
            if not np.isfinite(raw).all():
                raise ValueError(f"XR-1 {key} must be finite")
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
        steps = self._steps(raw, context)
        # Clip metadata covers both arms so partitioned decodes agree.
        gripper_targets, gripper_clip_max_abs = self._gripper_targets(steps)

        groups: dict[str, np.ndarray] = {}
        lag_stats: dict[str, dict[str, float | int]] = {}
        for name in groups_to_decode:
            model = self.kinematics.models[name]
            current = _state_vector(
                context.measured_state.groups[name], label=f"observation {name}"
            ).copy()
            solver = self.diff_solvers.get(name)
            lag = None
            if solver is not None:
                solver.reset()
                lag = {
                    "worst_lag_mm": 0.0,
                    "worst_lag_deg": 0.0,
                    "lag_exceedances": 0,
                }
            key = f"{GROUP_SIDES[name]}_ee_pose"
            rows: list[np.ndarray] = []
            for index, step in enumerate(steps):
                current = self._solve_knot(
                    model,
                    current,
                    pose_matrix(step[key]),
                    float(gripper_targets[name][index]),
                    diff_solver=solver,
                    lag=lag,
                )
                rows.append(current.copy())
            groups[name] = np.ascontiguousarray(np.stack(rows))
            if lag is not None:
                lag_stats[name] = lag

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
                "native_action_semantics": NATIVE_ACTION_SEMANTICS,
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
