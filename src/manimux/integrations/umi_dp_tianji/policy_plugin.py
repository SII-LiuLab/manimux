"""Tianji FK/IK boundary for UMI_DP's standard absolute EE action dictionaries."""

from __future__ import annotations

import math
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.integrations.openwam_yam.policy_plugin import pose_matrix
from manimux.integrations.umi_dp_tianji.history import WindowSnapshot
from manimux.kinematics import build_kinematics
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.types import ActionChunk

SEMANTICS = "absolute_per_arm_base_xyz_wxyz"
CAMERA_MAP = {
    "cam_left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
    "cam_left_wrist_prev": "left_wrist_prev",
    "cam_right_wrist_prev": "right_wrist_prev",
}


@dataclass(slots=True)
class UmiRequest(RtcInferenceRequest):
    xpolicylab_state: dict | None = None
    xpolicylab_additional_info: dict | None = None


def matrix_pose(matrix):
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.r_[matrix[:3, 3], quat[[3, 0, 1, 2]]]


def state_vector(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (8,) or not np.isfinite(value).all() or not 0 <= value[-1] <= 1:
        raise ValueError("Tianji state must be seven finite radian joints and aperture in [0, 1]")
    return value


class UmiDpTianjiAdapter:
    def __init__(self, robot, policy):
        self.validate(robot, policy)
        self.policy = policy
        self.horizon = policy.horizon_steps
        self.dt_ns = round(policy.effective_action_dt_s * 1e9)
        self.offset_ns = round(float(policy.options["first_action_offset_s"]) * 1e9)
        validation_dt = float(policy.options.get("ik_validation_dt_s", 0.004))
        if not math.isfinite(validation_dt) or validation_dt <= 0:
            raise ValueError("ik_validation_dt_s must be positive")
        self.validation_dt = validation_dt
        self.cameras = policy.options.get("camera_map", CAMERA_MAP)
        if set(self.cameras) != set(CAMERA_MAP):
            raise ValueError("UMI requires both wrist cameras at both observation times")
        self.kin = {}
        self.ik_backend = policy.options.get("ik_backend", "analytic")
        if self.ik_backend not in {"analytic", "diff"}:
            raise ValueError("ik_backend must be analytic or diff")
        if self.ik_backend == "analytic" and policy.options.get("diff_ik"):
            raise ValueError("diff_ik settings apply only to ik_backend: diff")
        self.diff_solvers = {}
        for side in ("left", "right"):
            options = dict(policy.options.get("kinematics_options", {}))
            options.update(policy.options.get(f"{side}_kinematics_options", {}))
            self.kin[side] = build_kinematics(
                policy.options.get("kinematics", "tianji"), arm=side, **options
            )
            if self.kin[side].num_arm_joints != 7:
                raise ValueError("UMI Tianji requires seven arm joints")
            if self.ik_backend == "diff":
                from manimux.kinematics.tianji import TianjiKinematics
                from manimux.kinematics.tianji_diff import (
                    DifferentialIKConfig,
                    TianjiDifferentialIK,
                )

                if not isinstance(self.kin[side], TianjiKinematics):
                    raise ValueError("UMI differential IK requires Tianji kinematics")
                config = DifferentialIKConfig.model_validate(policy.options.get("diff_ik", {}))
                if not config.check_j67:
                    raise ValueError("UMI differential IK requires the J6/J7 constraint")
                self.diff_solvers[side] = TianjiDifferentialIK(self.kin[side], config)
        self.anchors = OrderedDict()

    def validate(self, robot, policy):
        if policy.worker != "xpolicylab_ws":
            raise ValueError("UMI_DP must use xpolicylab_ws")
        if list(robot.group_dims.items()) != [("left_arm", 8), ("right_arm", 8)]:
            raise ValueError("UMI Tianji requires left_arm/right_arm with 7+1 values")
        options = policy.options
        for key in ("first_action_offset_s", "observation_period_s"):
            if not np.isfinite(options.get(key, np.nan)) or options[key] <= 0:
                raise ValueError(f"Bind the checkpoint {key} before constructing the adapter")
        identity = {} if policy.expected_backend is None else policy.expected_backend.model
        if identity.get("action_semantics") != SEMANTICS:
            raise ValueError(
                "UMI server identity must declare absolute per-arm base pose semantics"
            )
        if robot.driver == "tianji_dual":
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
                ("action_horizon", policy.horizon_steps),
                ("action_dt_s", policy.effective_action_dt_s),
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
                    self.kin[side].fk(values[:7], float(values[-1]))
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
                        self.kin[side].fk(values[:7], float(values[-1]))
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

    def _solve_knot(self, kin, current, target, aperture, duration_s, *, diff_solver=None):
        start = kin.fk(current, aperture)
        rotation = Rotation.from_matrix(start[:3, :3].T @ target[:3, :3]).as_rotvec()
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
            waypoint[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(alpha * rotation).as_matrix()
            if diff_solver is None:
                ok, solved = kin.ik(waypoint, current, aperture)
            else:
                result = diff_solver.solve(waypoint, current, duration_s / substeps)
                if not result.ok:
                    raise ValueError(
                        f"Tianji differential IK {result.reason}; rejecting the entire chunk "
                        f"(lag_mm={result.pos_err_mm}, lag_deg={result.rot_err_deg}, "
                        f"detail={result.detail})"
                    )
                ok, solved = result.ok, result.joints
            solved = np.asarray(solved, dtype=float)
            if not ok or solved.shape != (7,) or not np.isfinite(solved).all():
                raise ValueError("Tianji IK failed; rejecting the entire chunk")
            current = solved.copy()
        return current

    def decode_action(self, raw, context):
        # The WebSocket client unwraps a plain response to its action list.
        steps = raw.get("actions") if isinstance(raw, Mapping) else raw
        if not isinstance(steps, Sequence) or len(steps) != self.horizon:
            raise ValueError("UMI action horizon differs from the checkpoint")
        anchors = self.anchors.pop(context.request_seq, None)
        if anchors is None:
            raise ValueError("UMI action has no matching observation")
        action_origin = context.observation_time_ns + self.offset_ns
        execution_ns = context.execution_time_ns or context.created_time_ns
        skip = max(0, (execution_ns - action_origin) // self.dt_ns)
        if skip >= self.horizon:
            raise ValueError("UMI response has no future actions")
        groups = {}
        for side in ("left", "right"):
            group = f"{side}_arm"
            measured = (
                anchors[group]
                if context.measured_state is None
                else context.measured_state.groups[group]
            )
            current = state_vector(measured)[:7].copy()
            diff_solver = self.diff_solvers.get(side)
            if diff_solver is not None:
                diff_solver.reset()
            rows = []
            for knot_index, step in enumerate(steps[skip:]):
                target = pose_matrix(step[f"{side}_ee_pose"])
                grip = np.asarray(step[f"{side}_ee_joint_state"], dtype=float)
                if grip.shape != (1,) or not np.isfinite(grip).all() or not 0 <= grip[0] <= 1:
                    raise ValueError("UMI gripper action must lie in [0, 1]")
                duration_s = self.dt_ns / 1e9
                if knot_index == 0:
                    duration_s = max(
                        self.validation_dt,
                        min(duration_s, (action_origin + skip * self.dt_ns - execution_ns) / 1e9),
                    )
                current = self._solve_knot(
                    self.kin[side], current, target, float(grip[0]), duration_s,
                    diff_solver=diff_solver,
                )
                rows.append(np.r_[current, grip])
            groups[group] = np.asarray(rows)
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
            },
        )


def build_adapter(robot, policy):
    return UmiDpTianjiAdapter(robot, policy)
