"""Shared Tianji FK/IK for standard absolute per-arm-base EE actions.

Model transforms stay in XPolicyLab. Decoder scheduling and motion limits
are unchanged from the Xiaomi Tianji deployment.
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
from manimux.policy_adapter.gripper_mapping import GripperMapping
from manimux.policy_adapter.handoff import WaypointHandoff
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.types import (
    ActionChunk,
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RuntimeTrajectory,
)

ACTION_SEMANTICS = "absolute_per_arm_base_xyz_wxyz"
NATIVE_ACTION_SEMANTICS = "anchor_relative_ee_delta"
OBSERVATION_PROFILE = "tianji_taccap_two_wrist_black_ego"
GROUP_DIMS = {"left_arm": 8, "right_arm": 8}
GROUP_SIDES = {"left_arm": "left", "right_arm": "right"}
# A bent, in-limit pose used only to initialize decoder-process IK state.
WARMUP_JOINTS = np.radians([50.0, -40.0, -30.0, -100.0, -65.0, 0.0, 40.0])


@dataclass(slots=True)
class TianjiEERequest(RtcInferenceRequest):
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


class TianjiAbsoluteEEAdapter(PolicyAdapter):
    """Decode absolute TCP targets with continuous grippers and atomic arm chunks."""

    uses_motion_limits = True
    supports_context_only_decode = True
    supports_gripper_mapping = True
    # Preserve the real request observation (including the model's pose anchor).
    # Waypoint handoff selects its IK seed separately from the outgoing runtime
    # reference and drops the expired source prefix before IK.
    decode_seed_source = "observation_state"
    # Each arm can be solved in its own process, but the decoder client commits
    # only after both partitions have completed successfully.
    decode_partitions = tuple(GROUP_DIMS)

    def __init__(
        self, robot: dict, policy: dict, *, kinematics=None, motion_limits=None
    ) -> None:
        options = policy["adapter"]
        if kinematics is None and robot.get("config") is not None:
            from manimux.embodiments.robot import RobotModel

            kinematics = RobotModel.from_config(robot["config"]).kinematics
        if not isinstance(kinematics, RobotKinematics):
            raise ValueError("XR-1 Tianji requires the assembled robot kinematics")
        self.kinematics = kinematics
        self.policy = policy
        self.motion_limits = motion_limits
        self._group_order = tuple(options.get("group_order", GROUP_DIMS))
        self._camera_map = dict(options.get("camera_map", {}))
        self._required_cameras = tuple(self._camera_map.values())
        self._horizon_steps = int(policy["horizon_policy_steps"])
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
        legacy_mapping = {"gripper_output_deadzone", "gripper_output_exponent"}.intersection(
            options
        )
        if legacy_mapping:
            raise ValueError(
                f"move {sorted(legacy_mapping)} to executor.smooth.gripper mode: curve"
            )
        self._gripper_mapping = GripperMapping.from_options(options.get("gripper_mapping"))

        self.ik_backend = options.get("ik_backend", "analytic")
        if self.ik_backend not in {"analytic", "diff"}:
            raise ValueError("policy.adapter.ik_backend must be analytic or diff")
        self.execute_diff_ik_substeps = options.get("execute_diff_ik_substeps", False)
        if not isinstance(self.execute_diff_ik_substeps, bool):
            raise ValueError("policy.adapter.execute_diff_ik_substeps must be boolean")
        if self.execute_diff_ik_substeps and self.ik_backend != "diff":
            raise ValueError("execute_diff_ik_substeps requires ik_backend: diff")
        self.waypoint_handoff = WaypointHandoff.from_options(
            options,
            source_dt_ns=self._action_dt_ns,
            runtime_dt_ns=(
                self._action_dt_ns // max(1, math.ceil(self._action_dt_s / self._validation_dt_s))
            ),
        )
        if self.waypoint_handoff is not None and not self.execute_diff_ik_substeps:
            raise ValueError("handoff_waypoint requires execute_diff_ik_substeps: true")
        self.supports_waypoint_handoff = self.waypoint_handoff is not None
        if self.ik_backend == "analytic" and options.get("diff_ik"):
            raise ValueError("policy.adapter.diff_ik applies only to ik_backend: diff")
        self.diff_solvers = {}
        if self.ik_backend == "diff":
            from manimux.embodiments.arm.tianji.kinematics import TianjiArmKinematics
            from manimux.kinematics.tianji_diff import (
                DifferentialIKConfig,
                TianjiDifferentialIK,
            )

            arm_motion = None if motion_limits is None else motion_limits.get("arm")
            if arm_motion is None or arm_motion.get("max_velocity") is None:
                raise ValueError("Tianji differential IK requires resolved arm motion limits")
            tuning = dict(options.get("diff_ik", {}))
            # The QP rate bound is the shared executor profile, never a second constant.
            if "max_velocity_rad_s" in tuning:
                raise ValueError(
                    "policy.adapter.diff_ik.max_velocity_rad_s comes from the motion limits"
                )
            tuning["max_velocity_rad_s"] = arm_motion["max_velocity"]
            config = DifferentialIKConfig.model_validate(tuning)
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

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._required_cameras if name not in snapshot.frames]
        if missing:
            raise ValueError(f"XR-1 Tianji adapter is missing cameras: {missing}")
        return snapshot

    def prepare_request(self, request: InferenceRequest) -> TianjiEERequest:
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
        return TianjiEERequest(
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
                poses[row, offset + GROUP_DIMS[name] - 1] = self._gripper_mapping.inverse(
                    state[-1]
                )
                offset += GROUP_DIMS[name]
        return poses, weights

    def _solve_knot(
        self,
        model,
        current,
        target,
        gripper,
        *,
        diff_solver=None,
        lag=None,
        substep_joints=None,
        duration_s=None,
    ):
        current = _state_vector(current, label="IK seed").copy()
        current[-1] = gripper
        start = model.fk(current)
        rotation = rotation_vector(start[:3, :3].T @ target[:3, :3])

        duration_s = self._action_dt_s if duration_s is None else duration_s
        substeps = max(1, math.ceil(duration_s / self._validation_dt_s))

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
                    duration_s / substeps,
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
            if substep_joints is not None:
                substep_joints.append(current[:7].copy())
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
            targets[name] = self._gripper_mapping.map(clipped)
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
        targets = {
            name: [
                (pose_matrix(step[f"{GROUP_SIDES[name]}_ee_pose"]), float(gripper))
                for step, gripper in zip(steps, gripper_targets[name], strict=True)
            ]
            for name in self._group_order
        }
        handoff = None
        if context.handoff_reference is not None and self.waypoint_handoff is not None:
            # Plan over both groups even in a partition so atomic decodes agree.
            handoff = self.waypoint_handoff.plan(
                context,
                origin_ns=context.observation_time_ns,
                targets=targets,
                fk={
                    name: lambda state, model=self.kinematics.models[name]: (
                        model.fk(state),
                        float(state[-1]),
                    )
                    for name in self._group_order
                },
            )
            targets = handoff.targets
        seed_groups = context.measured_state.groups if handoff is None else handoff.start_state

        groups: dict[str, np.ndarray] = {}
        runtime_groups: dict[str, np.ndarray] = {}
        samples_per_action = None
        lag_stats: dict[str, dict[str, float | int]] = {}
        for name in groups_to_decode:
            model = self.kinematics.models[name]
            current = _state_vector(seed_groups[name], label=f"IK seed {name}").copy()
            previous_gripper = float(current[-1])
            solver = self.diff_solvers.get(name)
            lag = None
            if solver is not None:
                solver.reset()
                lag = {
                    "worst_lag_mm": 0.0,
                    "worst_lag_deg": 0.0,
                    "lag_exceedances": 0,
                }
            rows: list[np.ndarray] = []
            runtime_rows: list[np.ndarray] | None = (
                [] if self.execute_diff_ik_substeps else None
            )
            if handoff is not None:
                for pose, gripper in handoff.lead_in[name]:
                    current = self._solve_knot(
                        model,
                        current,
                        pose,
                        gripper,
                        diff_solver=solver,
                        lag=lag,
                        duration_s=self.waypoint_handoff.runtime_dt_ns / 1e9,
                    )
                    runtime_rows.append(current.copy())
                    previous_gripper = gripper
                # The last lead-in target is the first retained source row.
                rows.append(current.copy())
            for pose, gripper in targets[name]:
                substep_joints: list[np.ndarray] | None = (
                    [] if runtime_rows is not None else None
                )
                current = self._solve_knot(
                    model,
                    current,
                    pose,
                    gripper,
                    diff_solver=solver,
                    lag=lag,
                    substep_joints=substep_joints,
                )
                rows.append(current.copy())
                if runtime_rows is not None:
                    count = len(substep_joints)
                    if not count or (
                        samples_per_action is not None and count != samples_per_action
                    ):
                        raise ValueError("diffIK substep count changed within one action chunk")
                    samples_per_action = count
                    for substep_index, joints in enumerate(substep_joints, start=1):
                        alpha = substep_index / count
                        aperture = (1.0 - alpha) * previous_gripper + alpha * gripper
                        runtime_rows.append(np.r_[joints, aperture])
                previous_gripper = gripper
            groups[name] = np.ascontiguousarray(np.stack(rows))
            if runtime_rows is not None:
                runtime_groups[name] = np.ascontiguousarray(np.stack(runtime_rows))
            if lag is not None:
                lag_stats[name] = lag

        runtime_trajectory = None
        runtime_dt_ns = None
        if runtime_groups:
            if samples_per_action is None and handoff is None:
                raise ValueError("diffIK runtime trajectory has no samples")
            runtime_dt_ns = (
                self.waypoint_handoff.runtime_dt_ns
                if samples_per_action is None
                else self._action_dt_ns // samples_per_action
            )
            if runtime_dt_ns <= 0:
                raise ValueError("diffIK runtime trajectory interval rounded to zero")
            runtime_trajectory = RuntimeTrajectory(
                start_time_ns=(
                    context.observation_time_ns - self._action_dt_ns + runtime_dt_ns
                    if handoff is None
                    else handoff.time_ns + runtime_dt_ns
                ),
                dt_ns=runtime_dt_ns,
                groups=runtime_groups,
            )

        chunk = ActionChunk(
            plan_id=f"xr1-tianji-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
            source_offset_steps=0,
            runtime_trajectory=runtime_trajectory,
            metadata={
                "source_action_semantics": ACTION_SEMANTICS,
                "native_action_semantics": NATIVE_ACTION_SEMANTICS,
                "gripper_clip_tolerance": self._gripper_clip_tolerance,
                "gripper_clip_max_abs": gripper_clip_max_abs,
                "gripper_mapping": self._gripper_mapping.metadata(),
                "ik_backend": self.ik_backend,
                "ik_seed_source": (
                    self.decode_seed_source if handoff is None else "active_reference"
                ),
                "ik_seed_time_ns": (
                    context.measured_state.monotonic_ns if handoff is None else handoff.time_ns
                ),
                **({"ik_seed_plan_id": handoff.plan_id} if handoff is not None else {}),
                **(
                    {
                        "execute_diff_ik_substeps": True,
                        "diff_ik_substeps_per_action": samples_per_action,
                        "runtime_trajectory_dt_ns": runtime_dt_ns,
                    }
                    if runtime_trajectory is not None
                    else {}
                ),
                **({"diff_ik_lag": lag_stats} if lag_stats else {}),
            },
        )
        return chunk if handoff is None else self.waypoint_handoff.finish(chunk, handoff)
