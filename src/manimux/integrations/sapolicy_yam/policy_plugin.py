"""SAPolicy YAM embodiment adapter for the XPolicyLab WebSocket path.

ManiMux uses ``worker: xpolicylab_ws`` plus this adapter. The adapter owns
embodiment knowledge: camera aliases, calibrated intrinsics, YAM FK/IK,
and conversion of absolute EE wire actions into canonical joint-position
``ActionChunk`` values. Wire poses are the YAM grasp-site / ABC TCP frame.

Gripper is the same normalized ``[0, 1]`` stroke as YAM. Model loading lives
in ``XPolicyLab.policy.SAPolicy``.
"""

from __future__ import annotations

import logging
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.config import PolicyConfig, RobotConfig
from manimux.types import (
    ActionChunk,
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    SensorFrame,
)

log = logging.getLogger("manimux.policies.sapolicy")

DEFAULT_GROUP_ORDER = ("left_arm", "right_arm")
ARM_JOINTS = 6
GROUP_DIM = 7
WIRE_ACTION_DIM = 16
# Training Resize is stretch-to-224×168 (4:3), not Pi letterbox-to-224².
DEFAULT_WIRE_IMAGE_HW = (168, 224)


def _string_option(options: Mapping[str, object], name: str, default: str) -> str:
    value = options.get(name, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"policy.options.{name} must be a non-empty string")
    return value


def _string_sequence(
    options: Mapping[str, object], name: str, default: Sequence[str]
) -> tuple[str, ...]:
    value = options.get(name, list(default))
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"policy.options.{name} must be a non-empty list of strings")
    return tuple(value)


def _mapping_option(options: Mapping[str, object], name: str) -> dict[str, object]:
    value = options.get(name)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"policy.options.{name} must be a non-empty mapping")
    if not all(isinstance(key, str) and key for key in value):
        raise ValueError(f"policy.options.{name} keys must be non-empty strings")
    return dict(value)


@dataclass(slots=True)
class SAPolicyXPolicyRequest(InferenceRequest):
    """InferenceRequest plus EE/intrinsics for ``XPolicyLabWsPolicyModel``."""

    xpolicylab_additional_info: dict[str, object] = field(default_factory=dict)


def _parse_camera_map(options: Mapping[str, object]) -> dict[str, str]:
    raw = _mapping_option(options, "camera_map")
    if not all(isinstance(value, str) and value for value in raw.values()):
        raise ValueError("policy.options.camera_map must map model names to sensor names")
    result = {key: str(value) for key, value in raw.items()}
    if len(set(result.values())) != len(result):
        raise ValueError("policy.options.camera_map sensor names must be unique")
    return result


def _parse_intrinsics(
    options: Mapping[str, object], camera_names: Sequence[str]
) -> dict[str, np.ndarray]:
    raw = _mapping_option(options, "camera_intrinsics")
    if set(raw) != set(camera_names):
        raise ValueError(
            "policy.options.camera_intrinsics must contain exactly the model-facing "
            f"camera names {list(camera_names)}"
        )
    result: dict[str, np.ndarray] = {}
    for name in camera_names:
        matrix = np.asarray(raw[name], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"camera intrinsics for {name!r} must be a finite 3x3 matrix")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.isclose(matrix[2, 2], 1.0):
            raise ValueError(f"camera intrinsics for {name!r} are not a valid pinhole matrix")
        result[name] = matrix
    return result


def _parse_model_frame_transforms(
    options: Mapping[str, object], group_order: Sequence[str]
) -> dict[str, np.ndarray]:
    """Parse transforms from each IK base frame into the checkpoint state frame.

    The ABC bottles checkpoint was trained from the two-arm station MJCF, whose
    state positions are expressed in the station/world frame.  ManiMux solves IK
    with one standalone arm model per side.  Relative body-frame actions are
    invariant to this fixed transform, but the checkpoint's *state conditioning*
    is not, so the transform must be applied before observations reach SAPolicy
    and inverted again before local-arm IK.
    """
    raw = options.get("model_from_kinematics")
    if raw is None:
        return {group: np.eye(4, dtype=np.float64) for group in group_order}
    return {
        group: np.asarray(raw[group], dtype=np.float64)
        for group in group_order
    }


def _parse_wire_image_hw(options: Mapping[str, object]) -> tuple[int, int]:
    raw = options.get("wire_image_hw", list(DEFAULT_WIRE_IMAGE_HW))
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("policy.options.wire_image_hw must be [height, width]")
    height, width = int(raw[0]), int(raw[1])
    if height <= 0 or width <= 0:
        raise ValueError("policy.options.wire_image_hw must be positive")
    return height, width


def resize_rgb_uint8_area(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Stretch RGB uint8 with cv2.INTER_AREA. Same geometry as training Resize."""
    import cv2

    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"RGB must have shape [H,W,3], got {image.shape}")
    if image.shape[0] == height and image.shape[1] == width:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _pose_to_wire_endpose(pose: np.ndarray) -> np.ndarray:
    """FK grasp-site pose → ``pos3 + quat_xyzw`` (scipy convention)."""
    pose = np.asarray(pose, dtype=np.float64)
    quaternion_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return np.concatenate([pose[:3, 3], quaternion_xyzw]).astype(np.float64)


def _wire_endpose_to_pose(endpose: np.ndarray) -> np.ndarray:
    """``pos3 + quat_xyzw`` → 4x4 grasp-site pose for local-arm IK."""
    values = np.asarray(endpose, dtype=np.float64).reshape(-1)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(values[3:7]).as_matrix()
    pose[:3, 3] = values[:3]
    return pose


class SAPolicyYamAdapter:
    """Encode YAM observations and solve SAPolicy Cartesian chunks back to joints."""

    def __init__(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        from manimux.kinematics import build_kinematics

        self._group_order = _string_sequence(
            policy.options, "group_order", DEFAULT_GROUP_ORDER
        )
        self._horizon_steps = policy.horizon_steps
        self._action_dt_ns = int(policy.effective_action_dt_s * 1_000_000_000)
        self._camera_map = _parse_camera_map(policy.options)
        self._intrinsics = _parse_intrinsics(
            policy.options, tuple(self._camera_map)
        )
        self._wire_image_hw = _parse_wire_image_hw(policy.options)
        self._model_from_kinematics = _parse_model_frame_transforms(
            policy.options, self._group_order
        )
        self._kinematics_from_model = {
            group: np.linalg.inv(transform)
            for group, transform in self._model_from_kinematics.items()
        }

        kinematics_name = _string_option(policy.options, "kinematics", "yam")
        kinematics_options = policy.options.get("kinematics_options", {})
        if not isinstance(kinematics_options, dict):
            raise ValueError("policy.options.kinematics_options must be a mapping")
        self._kinematics = build_kinematics(kinematics_name, **kinematics_options)
        self._anchors: OrderedDict[int, np.ndarray] = OrderedDict()
        if self._kinematics.num_arm_joints != ARM_JOINTS:
            raise ValueError(
                f"SAPolicy YAM requires {ARM_JOINTS} arm joints, kinematics reports "
                f"{self._kinematics.num_arm_joints}"
            )

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._camera_map.values() if name not in snapshot.frames]
        if missing:
            raise ValueError(f"SAPolicy YAM adapter is missing cameras: {missing}")
        return snapshot

    def prepare_request(self, request: InferenceRequest) -> InferenceRequest:
        snapshot = request.observation
        anchor = np.concatenate(
            [
                np.asarray(snapshot.state.groups[name], dtype=np.float64)
                for name in self._group_order
            ]
        )
        self._anchors[request.request_seq] = anchor
        while len(self._anchors) > 8:
            self._anchors.popitem(last=False)

        payload: dict[str, object] = {}
        for side, group in zip(("left", "right"), self._group_order, strict=True):
            state = np.asarray(snapshot.state.groups[group], dtype=np.float64)
            local_pose = self._kinematics.fk(state[:ARM_JOINTS], float(state[-1]))
            pose = self._model_from_kinematics[group] @ local_pose
            payload[f"{side}_endpose"] = _pose_to_wire_endpose(pose)
            payload[f"{side}_gripper"] = float(state[-1])

        model_cameras = tuple(self._camera_map)
        wire_height, wire_width = self._wire_image_hw
        native_hw: dict[str, list[int]] = {}
        resized_frames: dict[str, SensorFrame] = {}
        for model_name, sensor_name in self._camera_map.items():
            frame = snapshot.frames[sensor_name]
            native_hw[model_name] = [int(frame.data.shape[0]), int(frame.data.shape[1])]
            resized_frames[sensor_name] = SensorFrame(
                name=frame.name,
                data=resize_rgb_uint8_area(frame.data, wire_height, wire_width),
                capture_monotonic_ns=frame.capture_monotonic_ns,
                sequence=frame.sequence,
            )
        sap_info = {
            "left_endpose": payload["left_endpose"],
            "right_endpose": payload["right_endpose"],
            "left_gripper": payload["left_gripper"],
            "right_gripper": payload["right_gripper"],
            "intrinsics": dict(self._intrinsics),
            "camera_names": list(model_cameras),
            "image_native_hw": native_hw,
        }
        return SAPolicyXPolicyRequest(
            session_id=request.session_id,
            request_seq=request.request_seq,
            observation_time_ns=request.observation_time_ns,
            deadline_ns=request.deadline_ns,
            observation=ObservationSnapshot(
                state=snapshot.state, frames=resized_frames
            ),
            instruction=request.instruction,
            xpolicylab_additional_info={"sapolicy": sap_info},
        )

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        raw_actions = raw.get("actions") if isinstance(raw, Mapping) else raw
        actions = np.asarray(raw_actions, dtype=np.float64)
        if actions.shape != (self._horizon_steps, WIRE_ACTION_DIM):
            raise ValueError(
                "SAPolicy actions must have shape "
                f"({self._horizon_steps}, {WIRE_ACTION_DIM}), got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("SAPolicy actions contain non-finite values")
        anchor = self._anchors.pop(context.request_seq, None)
        if anchor is None:
            raise ValueError(
                f"SAPolicy adapter has no observation anchor for request {context.request_seq}"
            )

        groups: dict[str, np.ndarray] = {}
        for arm_index, group in enumerate(self._group_order):
            state_start = arm_index * GROUP_DIM
            seed_state = anchor[state_start : state_start + GROUP_DIM]
            if context.measured_state is not None:
                measured = context.measured_state.groups.get(group)
                if measured is None:
                    raise ValueError(
                        f"SAPolicy measured state is missing group {group!r}"
                    )
                seed_state = np.asarray(measured, dtype=np.float64)
                if seed_state.shape != (GROUP_DIM,) or not np.isfinite(seed_state).all():
                    raise ValueError(
                        f"SAPolicy measured state group {group!r} is invalid"
                    )
            groups[group] = self._solve_arm(
                group,
                actions[:, arm_index * 8 : arm_index * 8 + 8],
                np.asarray(seed_state[:ARM_JOINTS], dtype=np.float64),
            )

        return ActionChunk(
            plan_id=f"sapolicy-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
        )

    def _solve_arm(
        self, group: str, actions: np.ndarray, seed: np.ndarray
    ) -> np.ndarray:
        horizon = actions.shape[0]
        out = np.empty((horizon, GROUP_DIM), dtype=np.float64)
        current = self._kinematics.clip_arm_joints(seed)
        failures = 0
        for step, row in enumerate(actions):
            gripper = float(row[7])
            model_target = _wire_endpose_to_pose(row[:7])
            target = self._kinematics_from_model[group] @ model_target
            converged, raw_solved = self._kinematics.ik(target, current, float(gripper))
            solved = np.asarray(raw_solved, dtype=np.float64)
            if converged and solved.shape == (ARM_JOINTS,) and np.isfinite(solved).all():
                current = self._kinematics.clip_arm_joints(solved)
            else:
                failures += 1
            out[step, :ARM_JOINTS] = current
            out[step, ARM_JOINTS] = float(gripper)
        if failures:
            log.warning("%s: IK did not converge on %d/%d chunk steps", group, failures, horizon)
        if not np.isfinite(out).all():
            raise ValueError(f"SAPolicy IK produced non-finite joints for {group}")
        return out

    def validate(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        del policy
        if tuple(robot.group_dims) != self._group_order:
            raise ValueError(
                "SAPolicy YAM requires robot groups in order "
                f"{list(self._group_order)}, got {list(robot.group_dims)}"
            )
        if any(robot.group_dims[name] != GROUP_DIM for name in self._group_order):
            raise ValueError("SAPolicy YAM requires two 7-value arm+gripper groups")


def build_adapter(robot: RobotConfig, policy: PolicyConfig) -> SAPolicyYamAdapter:
    return SAPolicyYamAdapter(robot, policy)
