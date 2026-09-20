"""Tianji FK/IK boundary for UMI_DP's standard absolute EE action dictionaries."""

from __future__ import annotations

import math
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from manimux.kinematics import build_kinematics
from manimux.kinematics.tianji_diff import rotation_matrix, rotation_vector
from manimux.policies.base import action_interval
from manimux.policies.xpolicylab.codec import matrix_pose, pose_matrix
from manimux.policy_adapter.base import PolicyAdapter
from manimux.policy_adapter.umi_dp.history import WindowSnapshot
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.types import ActionChunk

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
    supports_context_only_decode = True
    decode_partitions = ("left_arm", "right_arm")

    def __init__(self, robot, policy, *, kinematics=None):
        self.validate(robot, policy)
        self.policy = policy
        self.horizon = policy["horizon_steps"]
        self.dt_ns = round(action_interval(policy) * 1e9)
        self.offset_ns = round(float(policy["adapter"]["first_action_offset_s"]) * 1e9)
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
        if robot["type"] == "tianji_taccap" and self.robot_kinematics is None:
            from manimux.embodiments.robot import RobotModel

            self.robot_kinematics = RobotModel.from_config(robot["config"]).kinematics
        self.kin = {}
        self.ik_backend = policy["adapter"].get("ik_backend", "analytic")
        if self.ik_backend not in {"analytic", "diff"}:
            raise ValueError("ik_backend must be analytic or diff")
        if self.ik_backend == "analytic" and policy["adapter"].get("diff_ik"):
            raise ValueError("diff_ik settings apply only to ik_backend: diff")
        self.diff_solvers = {}
        for side in ("left", "right"):
            if self.robot_kinematics is not None:
                self.kin[side] = self.robot_kinematics.models[f"{side}_arm"]
                arm_solver = self.kin[side].arm
            else:
                # Compatibility for experiments not yet migrated to robot.type/config.
                options = dict(policy["adapter"].get("kinematics_options", {}))
                options.update(policy["adapter"].get(f"{side}_kinematics_options", {}))
                self.kin[side] = build_kinematics(
                    policy["adapter"].get("kinematics", "tianji"), arm=side, **options
                )
                arm_solver = self.kin[side]
            if self.ik_backend == "diff":
                from manimux.embodiments.arm.tianji.kinematics import TianjiArmKinematics
                from manimux.kinematics.tianji_diff import (
                    DifferentialIKConfig,
                    TianjiDifferentialIK,
                )

                if not isinstance(arm_solver, TianjiArmKinematics):
                    raise ValueError("UMI differential IK requires Tianji kinematics")
                config = DifferentialIKConfig.model_validate(policy["adapter"].get("diff_ik", {}))
                if not config.check_j67:
                    raise ValueError("UMI differential IK requires the J6/J7 constraint")
                self.diff_solvers[side] = TianjiDifferentialIK(arm_solver, config)
        self.anchors = OrderedDict()

    def validate(self, robot, policy):
        if policy["worker"] != "xpolicylab_ws":
            raise ValueError("UMI_DP must use xpolicylab_ws")
        if list(robot["group_dims"].items()) != [("left_arm", 8), ("right_arm", 8)]:
            raise ValueError("UMI Tianji requires left_arm/right_arm with 7+1 values")
        options = policy["adapter"]
        for key in ("first_action_offset_s", "observation_period_s"):
            if not np.isfinite(options.get(key, np.nan)) or options[key] <= 0:
                raise ValueError(f"Bind the checkpoint {key} before constructing the adapter")
        identity = {} if policy["expected_backend"] is None else policy["expected_backend"]["model"]
        if identity.get("action_semantics") != SEMANTICS:
            raise ValueError(
                "UMI server identity must declare absolute per-arm base pose semantics"
            )
        if robot["type"] == "tianji_taccap":
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
            if not options.get("deployment_bound") or any(key not in identity for key in required):
                raise ValueError("Bind UMI checkpoint identity before using the Tianji driver")
            for key, value in (
                ("action_horizon", policy["horizon_steps"]),
                ("action_dt_s", action_interval(policy)),
                ("first_action_offset_s", options["first_action_offset_s"]),
                ("observation_period_s", options["observation_period_s"]),
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
        extra, anchors = {}, {}
        for side in ("left", "right"):
            group = f"{side}_arm"
            for suffix, source in (("_prev", snapshot.previous), ("", snapshot)):
                values = state_vector(source.state.groups[group])
                extra[f"{side}_ee_pose{suffix}"] = matrix_pose(
                    self._fk(self.kin[side], values[:7], float(values[-1]))
                )
                if suffix:
                    extra[f"{side}_ee_joint_state_prev"] = values[-1:].copy()
                else:
                    anchors[group] = values.copy()
        self.anchors[request.request_seq] = anchors
        while len(self.anchors) > 8:
            self.anchors.popitem(last=False)
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
                    poses[row, index * 8 + 7] = values[-1]
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
        # Both paths return TCP in this arm's own base, never the Viewer frame.
        if self.robot_kinematics is not None:
            return kin.fk(np.r_[joints, aperture])
        return kin.fk(joints, aperture)

    def _ik(self, kin, target, seed, aperture):
        if self.robot_kinematics is not None:
            result = kin.ik(target, np.r_[seed, aperture], fixed_coordinates={"gripper": aperture})
            return result.converged, None if result.joints is None else result.joints[:7]
        return kin.ik(target, seed, aperture)

    def _diff_target(self, kin, target, aperture):
        # The shared assembly removes the tool once; the differential solver
        # operates on the same bare arm model and keeps its existing QP math.
        if self.robot_kinematics is not None:
            return kin.flange_target(target, np.array([aperture]))
        return target

    def _solve_knot(
        self, kin, current, target, aperture, duration_s, *, diff_solver=None, lag=None
    ):
        start = self._fk(kin, current, aperture)
        rotation = rotation_vector(start[:3, :3].T @ target[:3, :3])
        # The existing 1.8-degree branch check came from 250 Hz IK. Validate a
        # sampled SE(3) path at that cadence rather than relaxing that detector
        # or applying its per-servo-step threshold to an entire 30/10 Hz knot.
        step_dt = self.validation_dt
        if diff_solver is not None:
            step_dt = min(step_dt, diff_solver.config.dt_max_s)
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
        return current

    def warmup_decode(self, partition):
        # Load libKine and build the OSQP problem before the first real chunk.
        sides = ("left", "right") if partition is None else (partition.removesuffix("_arm"),)
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

    def _decode_side(self, side, steps, measured, first_duration_s):
        current = state_vector(measured)[:7].copy()
        diff_solver = self.diff_solvers.get(side)
        lag = None
        if diff_solver is not None:
            diff_solver.reset()
            lag = {"worst_lag_mm": 0.0, "worst_lag_deg": 0.0, "lag_exceedances": 0}
        rows = []
        for knot_index, step in enumerate(steps):
            target = pose_matrix(step[f"{side}_ee_pose"])
            grip = np.asarray(step[f"{side}_ee_joint_state"], dtype=float)
            if grip.shape != (1,) or not np.isfinite(grip).all() or not 0 <= grip[0] <= 1:
                raise ValueError("UMI gripper action must lie in [0, 1]")
            duration_s = first_duration_s if knot_index == 0 else self.dt_ns / 1e9
            current = self._solve_knot(
                self.kin[side],
                current,
                target,
                float(grip[0]),
                duration_s,
                diff_solver=diff_solver,
                lag=lag,
            )
            rows.append(np.r_[current, grip])
        return np.asarray(rows), lag

    def _decode(self, raw, context, sides):
        # The WebSocket client unwraps a plain response to its action list.
        steps = raw.get("actions") if isinstance(raw, Mapping) else raw
        if not isinstance(steps, Sequence) or len(steps) != self.horizon:
            raise ValueError("UMI action horizon differs from the checkpoint")
        anchors = self.anchors.pop(context.request_seq, None)
        if anchors is None and context.measured_state is None:
            raise ValueError("UMI action has no matching observation")
        action_origin = context.observation_time_ns + self.offset_ns
        execution_ns = context.execution_time_ns or context.created_time_ns
        skip = max(0, (execution_ns - action_origin) // self.dt_ns)
        if skip >= self.horizon:
            raise ValueError("UMI response has no future actions")
        first_duration_s = max(
            self.validation_dt,
            min(self.dt_ns / 1e9, (action_origin + skip * self.dt_ns - execution_ns) / 1e9),
        )
        groups = {}
        lag_stats = {}
        for side in sides:
            group = f"{side}_arm"
            measured = (
                anchors[group]
                if context.measured_state is None
                else context.measured_state.groups[group]
            )
            groups[group], lag = self._decode_side(side, steps[skip:], measured, first_duration_s)
            if lag is not None:
                lag_stats[side] = lag
        return ActionChunk(
            plan_id=f"umi-dp-{uuid.uuid4().hex}",
            request_seq=context.request_seq,
            observation_time_ns=action_origin,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self.dt_ns,
            groups=groups,
            source_offset_steps=skip,
            metadata={
                "native_action_semantics": SEMANTICS,
                "measured_observation_time_ns": context.observation_time_ns,
                "first_action_offset_ns": self.offset_ns,
                "ik_backend": self.ik_backend,
                # Per-arm target residual of the diff QP; lag over max_lag_* is only
                # recorded under lag_policy report.
                **({"diff_ik_lag": lag_stats} if lag_stats else {}),
            },
        )
