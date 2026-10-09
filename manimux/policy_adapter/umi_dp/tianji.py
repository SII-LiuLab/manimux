"""Tianji FK/IK boundary for UMI_DP's standard absolute EE action dictionaries."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np

from manimux.kinematics import build_kinematics
from manimux.kinematics.tianji_diff import rotation_matrix, rotation_vector
from manimux.policies.base import action_interval
from manimux.policies.xpolicylab.codec import matrix_pose, pose_matrix
from manimux.policy_adapter.base import PolicyAdapter
from manimux.policy_adapter.gripper_mapping import GripperMapping
from manimux.policy_adapter.handoff import WaypointHandoff
from manimux.policy_adapter.umi_dp.history import WindowSnapshot, execution_offset_s
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.types import ActionChunk, RuntimeTrajectory

SEMANTICS = "absolute_per_arm_base_xyz_wxyz"
CAMERA_MAP = {
    "cam_left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
    "cam_left_wrist_prev": "left_wrist_prev",
    "cam_right_wrist_prev": "right_wrist_prev",
}
# A bent, in-limit pose for both arms; decoder warmup ignores the solve result.
WARMUP_JOINTS = np.radians([50.0, -40.0, -30.0, -100.0, -65.0, 0.0, 40.0])


@dataclass(slots=True)
class UmiRequest(RtcInferenceRequest):
    xpolicylab_state: dict | None = None
    xpolicylab_additional_info: dict | None = None


def state_vector(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (8,) or not np.isfinite(value).all() or not 0 <= value[-1] <= 1:
        raise ValueError("Tianji state must be seven finite radian joints and aperture in [0, 1]")
    return value


class UmiDpTianjiAdapter(PolicyAdapter):
    supports_gripper_mapping = True
    uses_motion_limits = True
    supports_context_only_decode = True
    # This checkpoint's full source trajectory starts at the request
    # observation. Timeline owns all stale-row trimming at commit time.
    decode_seed_source = "observation_state"
    # Arm IK is independent, so process decoding can solve both sides in
    # parallel and merge them atomically.
    decode_partitions = ("left_arm", "right_arm")

    def __init__(self, robot, policy, *, kinematics=None, motion_limits=None):
        self.validate(robot, policy)
        self.policy = policy
        self.motion_limits = motion_limits
        self.horizon = policy["horizon_policy_steps"]
        self.dt_ns = round(action_interval(policy) * 1e9)
        self.offset_ns = round(execution_offset_s(policy) * 1e9)
        self.gripper_mapping = GripperMapping.from_options(
            policy["adapter"].get("gripper_mapping")
        )
        self.execute_diff_ik_substeps = policy["adapter"].get(
            "execute_diff_ik_substeps", False
        )
        validation_dt = float(policy["adapter"].get("ik_validation_dt_s", 0.004))
        if not math.isfinite(validation_dt) or validation_dt <= 0:
            raise ValueError("ik_validation_dt_s must be positive")
        self.validation_dt = validation_dt
        self.cameras = policy["adapter"].get("camera_map", CAMERA_MAP)
        if set(self.cameras) != set(CAMERA_MAP):
            raise ValueError("UMI requires both wrist cameras at both observation times")
        # In-process decoding receives the robot's exact models. A spawned
        # action decoder loads geometry only, from the same assembly config.
        self.robot_kinematics = kinematics
        if self.robot_kinematics is None and robot["type"] == "tianji_taccap":
            from manimux.embodiments.robot import RobotModel

            if robot.get("config") is None:
                raise ValueError("UMI Tianji requires robot.config or assembled kinematics")
            self.robot_kinematics = RobotModel.from_config(robot["config"]).kinematics
        self.kin = {}
        self.ik_backend = policy["adapter"].get("ik_backend", "analytic")
        if self.ik_backend not in {"analytic", "diff"}:
            raise ValueError("ik_backend must be analytic or diff")
        if self.ik_backend == "analytic" and policy["adapter"].get("diff_ik"):
            raise ValueError("diff_ik settings apply only to ik_backend: diff")
        self.diff_solvers = {}
        for side in ("left", "right"):
            if self.robot_kinematics is None:
                # Non-hardware runtime fixtures use the fixed offline Tianji model.
                # Adapter configuration cannot replace or modify its geometry.
                self.kin[side] = build_kinematics("tianji", arm=side)
                arm_solver = self.kin[side]
            else:
                self.kin[side] = self.robot_kinematics.models[f"{side}_arm"]
                arm_solver = self.kin[side].arm
            if self.ik_backend == "diff":
                from manimux.embodiments.arm.tianji.kinematics import TianjiArmKinematics
                from manimux.kinematics.tianji_diff import (
                    DifferentialIKConfig,
                    TianjiDifferentialIK,
                )

                if not isinstance(arm_solver, TianjiArmKinematics):
                    raise ValueError("UMI differential IK requires Tianji kinematics")
                arm_motion = None if motion_limits is None else motion_limits.get("arm")
                if arm_motion is None or arm_motion.get("max_velocity") is None:
                    raise ValueError("UMI differential IK requires resolved arm motion limits")
                tuning = dict(policy["adapter"].get("diff_ik", {}))
                tuning["max_velocity_rad_s"] = arm_motion["max_velocity"]
                config = DifferentialIKConfig.model_validate(tuning)
                self.diff_solvers[side] = TianjiDifferentialIK(arm_solver, config)
        self.waypoint_handoff = WaypointHandoff.from_options(
            policy["adapter"], source_dt_ns=self.dt_ns, runtime_dt_ns=self._runtime_dt_ns()
        )
        self.supports_waypoint_handoff = self.waypoint_handoff is not None

    def validate(self, robot, policy):
        if policy["worker"] != "xpolicylab_ws":
            raise ValueError("UMI_DP must use xpolicylab_ws")
        if list(robot["group_dims"].items()) != [("left_arm", 8), ("right_arm", 8)]:
            raise ValueError("UMI Tianji requires left_arm/right_arm with 7+1 values")
        options = policy["adapter"]
        for key in ("first_action_offset_s", "observation_period_s"):
            if key in options:
                raise ValueError(
                    f"adapter.{key} moved to the checkpoint identity; set "
                    "adapter.execution_offset_s to execute at a different phase"
                )
        execute_substeps = options.get("execute_diff_ik_substeps", False)
        if not isinstance(execute_substeps, bool):
            raise ValueError("execute_diff_ik_substeps must be boolean")
        if execute_substeps and options.get("ik_backend", "analytic") != "diff":
            raise ValueError("execute_diff_ik_substeps requires ik_backend: diff")
        if options.get("handoff_waypoint") is not None and not execute_substeps:
            raise ValueError("handoff_waypoint requires execute_diff_ik_substeps")
        if "execution_offset_s" in options and not (
            np.isfinite(options["execution_offset_s"]) and options["execution_offset_s"] >= 0
        ):
            raise ValueError("adapter.execution_offset_s must be finite and non-negative")
        legacy_mapping = {"gripper_output_deadzone", "gripper_output_exponent"}.intersection(
            options
        )
        if legacy_mapping:
            raise ValueError(
                f"move {sorted(legacy_mapping)} to executor.smooth.gripper mode: curve"
            )
        GripperMapping.from_options(options.get("gripper_mapping"))
        identity = {} if policy["expected_backend"] is None else policy["expected_backend"]["model"]
        if identity.get("action_semantics") != SEMANTICS:
            raise ValueError(
                "UMI server identity must declare absolute per-arm base pose semantics"
            )
        required = (
            "checkpoint_sha256",
            "training_config_sha256",
            "checkpoint_path",
            "weight_key",
            "rgb_normalize",
            "action_horizon",
            "action_dt_s",
            "first_action_offset_s",
            "observation_period_s",
        )
        if any(key not in identity for key in required):
            raise ValueError("Bind UMI checkpoint identity before constructing the adapter")
        for key in ("first_action_offset_s", "observation_period_s"):
            if not np.isfinite(identity[key]) or identity[key] <= 0:
                raise ValueError(f"Bound checkpoint {key} must be positive")
        for key, value in (
            ("action_horizon", policy["horizon_policy_steps"]),
            ("action_dt_s", action_interval(policy)),
        ):
            if identity[key] != value:
                raise ValueError(f"Runtime {key} differs from the bound checkpoint")

    def build_observation(self, snapshot):
        if not isinstance(snapshot, WindowSnapshot) or snapshot.previous is None:
            raise ValueError(
                "UMI_DP requires the measured history strategy; request history is insufficient"
            )
        if any(name not in snapshot.frames for name in self.cameras.values()):
            raise ValueError("Missing UMI wrist history camera")
        return snapshot

    def prepare_request(self, request):
        snapshot = self.build_observation(request.observation)
        extra = {}
        for side in ("left", "right"):
            group = f"{side}_arm"
            for suffix, source in (("_prev", snapshot.previous), ("", snapshot)):
                values = state_vector(source.state.groups[group])
                extra[f"{side}_ee_pose{suffix}"] = matrix_pose(
                    self._fk(self.kin[side], values[:7], float(values[-1]))
                )
                if suffix:
                    extra[f"{side}_ee_joint_state_prev"] = values[-1:].copy()
        condition, weights = (
            getattr(request, "action_condition", None),
            getattr(request, "condition_weights", None),
        )
        if condition is not None:
            condition = np.asarray(condition, dtype=float)
            weights = np.asarray(weights, dtype=float)
            if condition.shape != (self.horizon, 16) or weights.shape != (self.horizon,):
                raise ValueError("RTC joint condition dimensions differ from the policy horizon")
            poses = np.zeros_like(condition)
            for row, weight in enumerate(weights):
                if weight == 0:
                    continue
                for index, side in enumerate(("left", "right")):
                    values = state_vector(condition[row, index * 8 : (index + 1) * 8])
                    poses[row, index * 8 : index * 8 + 7] = matrix_pose(
                        self._fk(self.kin[side], values[:7], float(values[-1]))
                    )
                    poses[row, index * 8 + 7] = self.gripper_mapping.inverse(values[-1])
            condition = poses
        return UmiRequest(
            session_id=request.session_id,
            request_seq=request.request_seq,
            observation_time_ns=request.observation_time_ns,
            deadline_ns=request.deadline_ns,
            observation=snapshot,
            instruction=request.instruction,
            action_condition=condition,
            condition_weights=weights,
            rtc_beta=getattr(request, "rtc_beta", 5.0),
            xpolicylab_state=extra,
            xpolicylab_additional_info={
                "umi_dp": {
                    "frame_times_ns": [
                        snapshot.previous.state.monotonic_ns,
                        snapshot.state.monotonic_ns,
                    ],
                    "camera_capture_times_ns": {
                        name: frame.capture_monotonic_ns for name, frame in snapshot.frames.items()
                    },
                }
            },
        )

    def _fk(self, kin, joints, aperture):
        # Both paths return TCP in this arm's own base, never the RoboGUI frame.
        if self.robot_kinematics is not None:
            return kin.fk(np.r_[joints, aperture])
        return kin.fk(joints, aperture)

    def _ik(self, kin, target, seed, aperture):
        if self.robot_kinematics is not None:
            result = kin.ik(target, np.r_[seed, aperture], fixed_coordinates={"gripper": aperture})
            return result.converged, None if result.joints is None else result.joints[:7]
        return kin.ik(target, seed, aperture)

    def _runtime_dt_ns(self):
        """Dense runtime interval, using the same substep count as _solve_knot."""
        return self.dt_ns // max(1, math.ceil(self.dt_ns / 1e9 / self.validation_dt))

    def _handoff_fk(self, kin, row):
        """kin: one arm's model; row: seven joints + aperture; returns (TCP pose, aperture)."""
        values = state_vector(row)
        return self._fk(kin, values[:7], float(values[-1])), float(values[-1])

    def _handoff_targets(self, steps):
        """steps: model action rows; returns per group (TCP pose, aperture) targets."""
        targets = {}
        for side in ("left", "right"):
            rows = []
            for step in steps:
                grip = np.asarray(step[f"{side}_ee_joint_state"], dtype=float)
                if grip.shape != (1,) or not np.isfinite(grip).all() or not 0 <= grip[0] <= 1:
                    raise ValueError("UMI gripper action must lie in [0, 1]")
                rows.append((pose_matrix(step[f"{side}_ee_pose"]), float(grip[0])))
            targets[f"{side}_arm"] = rows
        return targets

    def _map_gripper_steps(self, steps):
        """Map model apertures to robot apertures before IK and timeline handoff."""
        if self.gripper_mapping.mode == "continuous":
            return steps
        mapped_steps = []
        for step in steps:
            mapped = dict(step)
            for side in ("left", "right"):
                key = f"{side}_ee_joint_state"
                grip = np.asarray(step[key], dtype=float)
                if grip.shape != (1,) or not np.isfinite(grip).all() or not 0 <= grip[0] <= 1:
                    raise ValueError("UMI gripper action must lie in [0, 1]")
                mapped[key] = self.gripper_mapping.map(grip)
            mapped_steps.append(mapped)
        return mapped_steps

    def _handoff_steps(self, targets):
        """targets: per group speed-limited (TCP pose, aperture) rows; returns action rows."""
        rows = []
        for left, right in zip(targets["left_arm"], targets["right_arm"], strict=True):
            row = {}
            for side, (pose, grip) in (("left", left), ("right", right)):
                row[f"{side}_ee_pose"] = matrix_pose(pose)
                row[f"{side}_ee_joint_state"] = np.array([grip])
            rows.append(row)
        return rows

    def _diff_target(self, kin, target, aperture):
        # The shared assembly removes the tool once; the differential solver
        # operates on the same bare arm model and keeps its existing QP math.
        if self.robot_kinematics is not None:
            return kin.flange_target(target, np.array([aperture]))
        return target

    def _solve_knot(
        self,
        kin,
        current,
        target,
        aperture,
        duration_s,
        *,
        diff_solver=None,
        lag=None,
        substep_joints=None,
    ):
        start = self._fk(kin, current, aperture)
        rotation = rotation_vector(start[:3, :3].T @ target[:3, :3])
        # The existing 1.8-degree branch check came from 250 Hz IK. Validate a
        # sampled SE(3) path at that cadence rather than relaxing that detector
        # or applying its per-servo-step threshold to an entire 30/10 Hz knot.
        step_dt = self.validation_dt
        substeps = max(1, math.ceil(duration_s / step_dt))
        for index in range(1, substeps + 1):
            alpha = index / substeps
            waypoint = np.eye(4)
            waypoint[:3, 3] = (1 - alpha) * start[:3, 3] + alpha * target[:3, 3]
            waypoint[:3, :3] = start[:3, :3] @ rotation_matrix(alpha * rotation)
            if diff_solver is None:
                ok, solved = self._ik(kin, waypoint, current, aperture)
            else:
                result = diff_solver.solve(
                    self._diff_target(kin, waypoint, aperture), current, duration_s / substeps
                )
                if not result.ok:
                    raise ValueError(
                        f"Tianji differential IK {result.reason}; rejecting the entire chunk "
                        f"(lag_mm={result.pos_err_mm}, lag_deg={result.rot_err_deg}, "
                        f"detail={result.detail})"
                    )
                if lag is not None:
                    lag["worst_lag_mm"] = max(lag["worst_lag_mm"], float(result.pos_err_mm))
                    lag["worst_lag_deg"] = max(lag["worst_lag_deg"], float(result.rot_err_deg))
                    lag["lag_exceedances"] += int(result.lag_exceeded)
                ok, solved = result.ok, result.joints
            solved = np.asarray(solved, dtype=float)
            if not ok or solved.shape != (7,) or not np.isfinite(solved).all():
                raise ValueError("Tianji IK failed; rejecting the entire chunk")
            current = solved.copy()
            if substep_joints is not None:
                substep_joints.append(current.copy())
        return current

    def warmup_decode(self, partition):
        # Load libKine and build the OSQP problem before the first real chunk.
        if partition is None:
            sides = ("left", "right")
        elif partition in self.decode_partitions:
            sides = (partition.removesuffix("_arm"),)
        else:
            raise ValueError(f"unknown UMI decode partition {partition!r}")
        for side in sides:
            target = self._fk(self.kin[side], WARMUP_JOINTS, 0.5)
            target[:3, 3] += (0.001, 0.0, 0.0)
            solver = self.diff_solvers.get(side)
            if solver is None:
                self._ik(self.kin[side], target, WARMUP_JOINTS.copy(), 0.5)
            else:
                solver.reset()
                solver.solve(
                    self._diff_target(self.kin[side], target, 0.5),
                    WARMUP_JOINTS.copy(),
                    self.validation_dt,
                )
                solver.reset()

    def decode_action(self, raw, context):
        return self._decode(raw, context, ("left", "right"))

    def decode_action_partition(self, raw, context, partition):
        if partition not in self.decode_partitions:
            raise ValueError(f"unknown UMI decode partition {partition!r}")
        return self._decode(raw, context, (partition.removesuffix("_arm"),))

    def _decode_side(self, side, steps, seed_state, lead_in=None):
        """steps: model rows after the lead-in; seed_state: IK seed; lead_in: handoff targets."""
        seed = state_vector(seed_state)
        current = seed[:7].copy()
        previous_grip = float(seed[-1])
        diff_solver = self.diff_solvers.get(side)
        lag = None
        if diff_solver is not None:
            diff_solver.reset()
            lag = {"worst_lag_mm": 0.0, "worst_lag_deg": 0.0, "lag_exceedances": 0}
        rows = []
        runtime_rows = [] if self.execute_diff_ik_substeps else None
        samples_per_action = None
        for pose, grip in lead_in or ():
            # One diffIK step per runtime row; the last target is the join row.
            current = self._solve_knot(
                self.kin[side],
                current,
                pose,
                grip,
                self.waypoint_handoff.runtime_dt_ns / 1e9,
                diff_solver=diff_solver,
                lag=lag,
            )
            runtime_rows.append(np.r_[current, grip])
            previous_grip = grip
        if lead_in:
            rows.append(np.r_[current, previous_grip])
        for step in steps:
            target = pose_matrix(step[f"{side}_ee_pose"])
            grip = np.asarray(step[f"{side}_ee_joint_state"], dtype=float)
            if grip.shape != (1,) or not np.isfinite(grip).all() or not 0 <= grip[0] <= 1:
                raise ValueError("UMI gripper action must lie in [0, 1]")
            substep_joints = [] if runtime_rows is not None else None
            current = self._solve_knot(
                self.kin[side],
                current,
                target,
                float(grip[0]),
                self.dt_ns / 1e9,
                diff_solver=diff_solver,
                lag=lag,
                substep_joints=substep_joints,
            )
            rows.append(np.r_[current, grip])
            if runtime_rows is not None:
                count = len(substep_joints)
                if not count or (samples_per_action is not None and count != samples_per_action):
                    raise ValueError("diffIK substep count changed within one action chunk")
                samples_per_action = count
                for index, joints in enumerate(substep_joints, start=1):
                    alpha = index / count
                    dense_grip = (1.0 - alpha) * previous_grip + alpha * float(grip[0])
                    runtime_rows.append(np.r_[joints, dense_grip])
            previous_grip = float(grip[0])
        return (
            np.asarray(rows),
            None if runtime_rows is None else np.asarray(runtime_rows),
            samples_per_action,
            lag,
        )

    def _decode(self, raw, context, sides):
        # The WebSocket client unwraps a plain response to its action list.
        steps = raw.get("actions") if isinstance(raw, Mapping) else raw
        if not isinstance(steps, Sequence) or len(steps) != self.horizon:
            raise ValueError("UMI action horizon differs from the checkpoint")
        if context.measured_state is None:
            raise ValueError("UMI-DP decoding requires the request observation state")
        if context.measured_state.monotonic_ns != context.observation_time_ns:
            raise ValueError(
                "UMI-DP IK seed time must equal the request observation time "
                f"({context.measured_state.monotonic_ns} != {context.observation_time_ns})"
            )
        steps = self._map_gripper_steps(steps)
        action_origin = context.observation_time_ns + self.offset_ns
        handoff = None
        if context.handoff_reference is not None and self.waypoint_handoff is not None:
            handoff = self.waypoint_handoff.plan(
                context,
                origin_ns=action_origin,
                targets=self._handoff_targets(steps),
                fk={
                    f"{side}_arm": partial(self._handoff_fk, self.kin[side])
                    for side in ("left", "right")
                },
            )
            steps = self._handoff_steps(handoff.targets)
        groups = {}
        runtime_groups = {}
        samples_per_action = None
        lag_stats = {}
        for side in sides:
            group = f"{side}_arm"
            groups[group], runtime_rows, side_samples, lag = self._decode_side(
                side,
                steps,
                (
                    context.measured_state.groups[group]
                    if handoff is None
                    else handoff.start_state[group]
                ),
                lead_in=None if handoff is None else handoff.lead_in[group],
            )
            if runtime_rows is not None:
                if samples_per_action is not None and side_samples != samples_per_action:
                    raise ValueError("diffIK substep count differs between arms")
                samples_per_action = side_samples
                runtime_groups[group] = runtime_rows
            if lag is not None:
                lag_stats[side] = lag
        runtime_trajectory = None
        runtime_dt_ns = None
        if runtime_groups:
            if samples_per_action is None and handoff is None:
                raise ValueError("diffIK runtime trajectory has no samples")
            runtime_dt_ns = (
                self.waypoint_handoff.runtime_dt_ns
                if samples_per_action is None
                else self.dt_ns // samples_per_action
            )
            if runtime_dt_ns <= 0:
                raise ValueError("diffIK runtime trajectory interval rounded to zero")
            runtime_trajectory = RuntimeTrajectory(
                # The first dense row is the first substep leading to source row 0.
                start_time_ns=action_origin - self.dt_ns + runtime_dt_ns,
                dt_ns=runtime_dt_ns,
                groups=runtime_groups,
            )
        chunk = ActionChunk(
            plan_id=f"umi-dp-{uuid.uuid4().hex}",
            request_seq=context.request_seq,
            observation_time_ns=action_origin,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self.dt_ns,
            groups=groups,
            source_offset_steps=0,
            runtime_trajectory=runtime_trajectory,
            metadata={
                "native_action_semantics": SEMANTICS,
                "measured_observation_time_ns": context.observation_time_ns,
                "first_action_offset_ns": self.offset_ns,
                "ik_backend": self.ik_backend,
                "ik_seed_source": self.decode_seed_source,
                "ik_seed_time_ns": context.measured_state.monotonic_ns,
                "gripper_mapping": self.gripper_mapping.metadata(),
                **(
                    {
                        "execute_diff_ik_substeps": True,
                        "diff_ik_substeps_per_action": samples_per_action,
                        "runtime_trajectory_dt_ns": runtime_dt_ns,
                    }
                    if runtime_trajectory is not None
                    else {}
                ),
                # Per-arm target residual of the diff QP; lag over max_lag_* is only
                # recorded under lag_policy report.
                **({"diff_ik_lag": lag_stats} if lag_stats else {}),
            },
        )
        return chunk if handoff is None else self.waypoint_handoff.finish(chunk, handoff)
