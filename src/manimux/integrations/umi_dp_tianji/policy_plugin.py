"""UMI-DP policy adapter for the Tianji–TacCap embodiment."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from manimux.embodiments.arm.tianji.kinematics import (
    DifferentialIKConfig,
    TianjiDifferentialKinematics,
    rotation_matrix,
    rotation_vector,
)
from manimux.integrations.xpolicylab.obs_codec import matrix_pose, pose_matrix
from manimux.kinematics.composed import RobotKinematics
from manimux.policies.base import PolicyAdapterBase, action_interval
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot

SEMANTICS = "absolute_per_arm_base_xyz_wxyz"
CAMERA_MAP = {
    "cam_left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
    "cam_left_wrist_prev": "left_wrist_prev",
    "cam_right_wrist_prev": "right_wrist_prev",
}


def _analytic_kinematics(kinematics, _options):
    return kinematics


def _differential_kinematics(kinematics, options):
    config = DifferentialIKConfig.model_validate(options.get("diff_ik", {}))
    if not config.check_j67:
        raise ValueError("UMI differential IK requires the J6/J7 constraint")
    return RobotKinematics(
        {
            group: model.with_arm(TianjiDifferentialKinematics(model.arm, config))
            for group, model in kinematics.models.items()
        }
    )


kinematics_tianji_algos_dict = {
    "analytic": _analytic_kinematics,
    "diff": _differential_kinematics,
}


@dataclass(slots=True)
class UmiRequest(RtcInferenceRequest):
    xpolicylab_state: dict | None = None
    xpolicylab_additional_info: dict | None = None


def state_vector(value: object) -> np.ndarray:
    """Return one Tianji arm state: seven joints and one aperture."""

    state = np.asarray(value, dtype=float)
    if state.shape != (8,) or not np.isfinite(state).all() or not 0 <= state[-1] <= 1:
        raise ValueError("Tianji state must be seven finite radian joints and aperture in [0, 1]")
    return state


class UmiDpTianjiAdapter(PolicyAdapterBase):
    """Translate between Tianji runtime data and the UMI-DP policy contract."""

    supports_context_only_decode = True
    # The full source trajectory starts at the request observation.  Seed IK
    # from that same state; Timeline alone removes source rows that are stale
    # by commit time.
    decode_seed_source = "observation_state"
    # Each arm solves from its own seed with its own solver state and OSQP warm
    # start, so the two never exchange anything during a decode: one decoder
    # process per arm turns the chunk's serial IK into wall-clock parallel work.
    decode_partitions = ("left_arm", "right_arm")

    def __init__(self, robot: dict, policy: dict, *, kinematics=None) -> None:
        options = policy["options"]
        self.horizon = policy["horizon_steps"]
        self.dt_ns = round(action_interval(policy) * 1e9)
        self.offset_ns = round(float(options["first_action_offset_s"]) * 1e9)
        self.validation_dt = float(options.get("ik_validation_dt_s", 0.004))
        if kinematics is None:
            from manimux.embodiments.robot import RobotModel

            kinematics = RobotModel.from_config(robot["config"]).kinematics
        self.ik_backend = options.get("ik_backend", "analytic")
        self.kinematics = kinematics_tianji_algos_dict[self.ik_backend](
            kinematics, options
        )
        self.kin = {
            side: self.kinematics.models[f"{side}_arm"] for side in ("left", "right")
        }

    def validate(self, robot: dict, policy: dict) -> None:
        if policy["worker"] != "xpolicylab_ws":
            raise ValueError("UMI-DP must use xpolicylab_ws")
        if robot["group_dims"] != {"left_arm": 8, "right_arm": 8}:
            raise ValueError("UMI-DP Tianji requires left_arm/right_arm with 7+1 values")
        identity = policy["expected_backend"]["model"]
        if identity.get("action_semantics") != SEMANTICS:
            raise ValueError(
                "UMI-DP server identity must declare absolute per-arm base pose semantics"
            )

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        return snapshot

    def _umi_dp_additional_info(self, snapshot: ObservationSnapshot) -> dict:
        """Attach the two measured observation times required by UMI-DP."""
        return {
            "umi_dp": {
                "frame_times_ns": [
                    snapshot.previous.state.monotonic_ns,
                    snapshot.state.monotonic_ns,
                ]
            }
        }

    def _rtc_ee_condition(self, condition, weights):
        if condition is None:
            return None
        converted = np.zeros_like(condition, dtype=float)
        for row in np.flatnonzero(weights):
            for index, side in enumerate(("left", "right")):
                start = index * 8
                state = state_vector(condition[row, start : start + 8])
                converted[row, start : start + 7] = matrix_pose(self.kin[side].fk(state))
                converted[row, start + 7] = state[-1]
        return converted

    def prepare_request(self, request: InferenceRequest) -> UmiRequest:
        snapshot = self.build_observation(request.observation)
        state = {}
        for side in ("left", "right"):
            group = f"{side}_arm"
            model = self.kinematics.models[group]
            current = state_vector(snapshot.state.groups[group])
            previous = state_vector(snapshot.previous.state.groups[group])
            state[f"{side}_ee_pose"] = matrix_pose(model.fk(current))
            state[f"{side}_ee_pose_prev"] = matrix_pose(model.fk(previous))
            state[f"{side}_ee_joint_state_prev"] = previous[-1:].copy()

        condition = getattr(request, "action_condition", None)
        weights = getattr(request, "condition_weights", None)

        return UmiRequest(
            session_id=request.session_id,
            request_seq=request.request_seq,
            observation_time_ns=request.observation_time_ns,
            deadline_ns=request.deadline_ns,
            observation=snapshot,
            instruction=request.instruction,
            action_condition=self._rtc_ee_condition(condition, weights),
            condition_weights=weights,
            rtc_beta=getattr(request, "rtc_beta", 5.0),
            xpolicylab_state=state,
            xpolicylab_additional_info=self._umi_dp_additional_info(snapshot),
        )

    @staticmethod
    def _record_lag(lag, diagnostics):
        if "pos_err_mm" not in diagnostics:
            return
        lag["samples"] += 1
        lag["worst_lag_mm"] = max(lag["worst_lag_mm"], float(diagnostics["pos_err_mm"]))
        lag["worst_lag_deg"] = max(
            lag["worst_lag_deg"], float(diagnostics["rot_err_deg"])
        )
        lag["lag_exceedances"] += int(diagnostics["lag_exceeded"])

    def _solve_knot(self, model, current, target, gripper, duration_s, *, lag):
        current = state_vector(current).copy()
        current[-1] = gripper
        start = model.fk(current)
        rotation = rotation_vector(start[:3, :3].T @ target[:3, :3])

        step_dt_s = self.validation_dt
        if model.arm.max_step_duration_s is not None:
            step_dt_s = min(step_dt_s, model.arm.max_step_duration_s)
        substeps = max(1, math.ceil(duration_s / step_dt_s))

        for index in range(1, substeps + 1):
            alpha = index / substeps
            waypoint = np.eye(4)
            waypoint[:3, 3] = (1 - alpha) * start[:3, 3] + alpha * target[:3, 3]
            waypoint[:3, :3] = start[:3, :3] @ rotation_matrix(alpha * rotation)
            result = model.ik(
                waypoint,
                current,
                fixed_coordinates={"gripper": gripper},
                duration_s=duration_s / substeps,
            )
            if not result.ok:
                raise ValueError(
                    f"Tianji IK {result.reason}; rejecting the entire chunk "
                    f"(diagnostics={result.diagnostics})"
                )
            self._record_lag(lag, result.diagnostics)
            current = state_vector(result.joints).copy()

        return current

    def _decode_side(self, side, steps, seed_state):
        model = self.kin[side]
        model.reset()
        current = state_vector(seed_state).copy()
        lag = {
            "samples": 0,
            "worst_lag_mm": 0.0,
            "worst_lag_deg": 0.0,
            "lag_exceedances": 0,
        }
        rows = []
        for step in steps:
            target = pose_matrix(step[f"{side}_ee_pose"])
            gripper = np.asarray(step[f"{side}_ee_joint_state"], dtype=float)
            if (
                gripper.shape != (1,)
                or not np.isfinite(gripper).all()
                or not 0 <= gripper[0] <= 1
            ):
                raise ValueError("UMI-DP gripper action must be one value in [0, 1]")
            current = self._solve_knot(
                model,
                current,
                target,
                float(gripper[0]),
                self.dt_ns / 1e9,
                lag=lag,
            )
            rows.append(current)
        samples = lag.pop("samples")
        return np.asarray(rows), lag if samples else None

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        return self._decode(raw, context, ("left", "right"))

    def decode_action_partition(
        self, raw: object, context: ActionContext, partition: str
    ) -> ActionChunk:
        """Decode one arm; the decoder client merges the partitions it asked for."""
        if partition not in self.decode_partitions:
            raise ValueError(f"unknown UMI-DP decode partition {partition!r}")
        return self._decode(raw, context, (partition.removesuffix("_arm"),))

    def _decode(self, raw: object, context: ActionContext, sides: tuple[str, ...]) -> ActionChunk:
        if context.measured_state is None:
            raise ValueError("UMI-DP decoding requires the request observation state")
        if context.measured_state.monotonic_ns != context.observation_time_ns:
            raise ValueError(
                "UMI-DP IK seed time must equal the request observation time "
                f"({context.measured_state.monotonic_ns} != {context.observation_time_ns})"
            )

        steps = raw.get("actions") if isinstance(raw, Mapping) else raw
        assert isinstance(steps, Sequence) and len(steps) == self.horizon
        origin_ns = context.observation_time_ns + self.offset_ns

        groups = {}
        lag_stats = {}
        for side in sides:
            group = f"{side}_arm"
            groups[group], lag = self._decode_side(
                side,
                steps,
                context.measured_state.groups[group],
            )
            if lag is not None:
                lag_stats[side] = lag

        return ActionChunk(
            plan_id=f"umi-dp-{uuid.uuid4().hex}",
            request_seq=context.request_seq,
            observation_time_ns=origin_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self.dt_ns,
            groups=groups,
            source_offset_steps=0,
            metadata={
                "native_action_semantics": SEMANTICS,
                "measured_observation_time_ns": context.observation_time_ns,
                "first_action_offset_ns": self.offset_ns,
                "ik_backend": self.ik_backend,
                "ik_seed_source": self.decode_seed_source,
                "ik_seed_time_ns": context.measured_state.monotonic_ns,
                **({"diff_ik_lag": lag_stats} if lag_stats else {}),
            },
        )


def build_adapter(robot: dict, policy: dict, *, kinematics=None) -> UmiDpTianjiAdapter:
    return UmiDpTianjiAdapter(robot, policy, kinematics=kinematics)
