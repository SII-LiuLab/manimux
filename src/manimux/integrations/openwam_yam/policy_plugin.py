"""OpenWAM policy transport and absolute-EEF adapter for the dual-arm YAM.

The model process stays in OpenWAM's own environment. ManiMux sends three RGB
cameras plus a raw 20-D achieved EEF state and receives a 32-step raw EEF
chunk. Each arm occupies ``xyz(3) + rot6d(6) + gripper(1)``. Positions are
metres in that arm's robot-base frame; grippers are physical ``0=closed,
1=open`` values after the OpenWAM server applies inverse normalization.

Unlike Xiaomi XR-1, these are absolute EEF targets, not anchor-relative deltas.
The adapter converts every target rotation to a matrix and solves YAM IK to
produce ManiMux's canonical 14-D joint-position trajectory.
"""

from __future__ import annotations

import base64
import io
import logging
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from manimux.config import PolicyConfig, RobotConfig
from manimux.integrations.openwam_yam.ws_client import OpenWAMWsClient
from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import (
    ActionChunk,
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
)

log = logging.getLogger("manimux.policies.openwam")

DEFAULT_SERVER = "ws://127.0.0.1:8848"
DEFAULT_GROUP_ORDER = ("left_arm", "right_arm")
DEFAULT_CAMERA_MAP = {
    "head_camera": "front_camera",
    "left_wrist_camera": "left_camera",
    "right_wrist_camera": "right_camera",
}
DEFAULT_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following instruction: "
    "{instruction}"
)
ARM_JOINTS = 6
GROUP_DIM = 7
ARM_ACTION_DIM = 10
ACTION_DIM = 20


@dataclass(slots=True)
class OpenWAMInferenceRequest(InferenceRequest):
    """Inference request carrying the YAM-specific OpenWAM wire contract."""

    openwam_state: np.ndarray | None = None
    openwam_prompt: str | None = None


def _string_option(options: Mapping[str, object], name: str, default: str) -> str:
    value = options.get(name, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"policy.options.{name} must be a non-empty string")
    return value


def _positive_float_option(
    options: Mapping[str, object], name: str, default: float
) -> float:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"policy.options.{name} must be a positive number")
    return float(value)


def _float_option(options: Mapping[str, object], name: str, default: float) -> float:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"policy.options.{name} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"policy.options.{name} must be finite")
    return result


def _string_sequence(
    options: Mapping[str, object], name: str, default: Sequence[str]
) -> tuple[str, ...]:
    value = options.get(name, list(default))
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"policy.options.{name} must be a non-empty list of strings")
    return tuple(value)


def _camera_map(options: Mapping[str, object]) -> dict[str, str]:
    value = options.get("camera_map", dict(DEFAULT_CAMERA_MAP))
    if not isinstance(value, dict) or set(value) != set(DEFAULT_CAMERA_MAP):
        raise ValueError(
            "policy.options.camera_map must map head_camera, left_wrist_camera, "
            "and right_wrist_camera"
        )
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise ValueError("policy.options.camera_map must map strings to strings")
    return dict(value)


def matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    """Return OpenWAM's ``[column0, column1]`` 6-D rotation representation."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrix must be finite with shape (3, 3)")
    return np.concatenate([matrix[:, 0], matrix[:, 1]])


def rot6d_to_matrix(rotation: np.ndarray) -> np.ndarray:
    """Recover a proper rotation matrix with OpenWAM's Gram-Schmidt rule."""
    values = np.asarray(rotation, dtype=np.float64).reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("rot6d must be finite with shape (6,)")
    first = values[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-6:
        raise ValueError("rot6d first direction is degenerate")
    first = first / first_norm
    second = values[3:] - np.dot(first, values[3:]) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-6:
        raise ValueError("rot6d second direction is degenerate or collinear")
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def _encode_png(image: np.ndarray) -> str:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"OpenWAM RGB image must have shape (H, W, 3), got {array.shape}")
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OpenWAMWsPolicyModel:
    """Drain one official OpenWAM server buffer into each ManiMux action chunk."""

    def __init__(self, config: PolicyConfig) -> None:
        self._url = _string_option(config.options, "server", DEFAULT_SERVER)
        self._camera_map = _camera_map(config.options)
        self._horizon_steps = config.horizon_steps
        self._response_mode = _string_option(config.options, "response_mode", "chunk")
        if self._response_mode not in {"chunk", "stream"}:
            raise ValueError("policy.options.response_mode must be 'chunk' or 'stream'")
        self._connect_timeout_s = _positive_float_option(
            config.options, "connect_timeout_s", config.startup_timeout_s
        )
        self._request_timeout_s = _positive_float_option(
            config.options, "request_timeout_s", config.timeout_s
        )
        self._client: OpenWAMWsClient | None = None
        self._session_id: str | None = None
        self._backend_metadata: dict[str, object] = {}

    def reset(self, session_id: str) -> None:
        self.close()
        client = OpenWAMWsClient(
            self._url,
            connect_timeout_s=self._connect_timeout_s,
            request_timeout_s=self._request_timeout_s,
        )
        client.connect()
        pong = client.ping()
        if pong.get("representation") != "eef":
            client.close()
            raise RuntimeError(
                "OpenWAM YAM requires an EEF checkpoint; server advertises "
                f"{pong.get('representation')!r}"
            )
        if pong.get("gripper_convention") != "zero_closed_one_open":
            client.close()
            raise RuntimeError(
                "OpenWAM YAM requires gripper_convention=zero_closed_one_open; "
                f"server advertises {pong.get('gripper_convention')!r}"
            )
        if self._response_mode == "chunk":
            if pong.get("transport_mode") != "manimux_chunk":
                client.close()
                raise RuntimeError(
                    "OpenWAM chunk mode requires scripts/servers/openwam_yam_chunk_server.py"
                )
            if pong.get("action_horizon") != self._horizon_steps:
                client.close()
                raise RuntimeError(
                    "OpenWAM server action horizon does not match ManiMux: "
                    f"{pong.get('action_horizon')} != {self._horizon_steps}"
                )
        client.reset()
        self._client = client
        self._session_id = session_id
        self._backend_metadata = {
            "server": "openwam_policy_server",
            "model": {key: value for key, value in pong.items() if key != "type"},
        }

    def infer(self, request: InferenceRequest) -> object:
        if request.session_id != self._session_id or self._client is None:
            raise RuntimeError("OpenWAM session is not initialized")
        state = getattr(request, "openwam_state", None)
        prompt = getattr(request, "openwam_prompt", None)
        state_array = np.asarray(state, dtype=np.float64).reshape(-1)
        if state_array.shape != (ACTION_DIM,) or not np.isfinite(state_array).all():
            raise ValueError(f"OpenWAM YAM state must be finite with shape ({ACTION_DIM},)")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("OpenWAM YAM request is missing its wrapped prompt")

        images = {
            wire_name: _encode_png(request.observation.frames[sensor_name].data)
            for wire_name, sensor_name in self._camera_map.items()
        }
        payload: dict[str, object] = {
            "images": images,
            "prompt": prompt,
            "state": state_array.tolist(),
        }
        return self._client.infer_chunk(
            payload,
            horizon_steps=self._horizon_steps,
            response_mode=self._response_mode,
        )

    def capabilities(self) -> PolicyCapabilities:
        return PolicyCapabilities(
            sampling_modes=frozenset({"default"}),
            backend_metadata=dict(self._backend_metadata),
        )

    def close(self) -> None:
        client, self._client = self._client, None
        self._session_id = None
        if client is not None:
            client.close()


class OpenWAMYamAdapter:
    """YAM joints ↔ OpenWAM absolute EEF20 with IK at the action boundary."""

    def __init__(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        from manimux.kinematics import build_kinematics

        self._group_order = _string_sequence(
            policy.options, "group_order", DEFAULT_GROUP_ORDER
        )
        self._group_dims = dict(robot.group_dims)
        self._camera_map = _camera_map(policy.options)
        self._horizon_steps = policy.horizon_steps
        self._action_dt_ns = int(policy.effective_action_dt_s * 1_000_000_000)
        self._prompt_template = _string_option(
            policy.options, "prompt_template", DEFAULT_PROMPT_TEMPLATE
        )
        if self._prompt_template.count("{instruction}") != 1:
            raise ValueError("policy.options.prompt_template must contain {instruction} once")
        self._gripper_min = _float_option(policy.options, "gripper_min", 0.0)
        self._gripper_max = _float_option(policy.options, "gripper_max", 1.0)
        self._gripper_tolerance = _positive_float_option(
            policy.options, "gripper_tolerance", 0.1
        )
        if not self._gripper_min < self._gripper_max:
            raise ValueError("OpenWAM gripper_min must be below gripper_max")

        kinematics_name = _string_option(policy.options, "kinematics", "yam")
        kinematics_options = policy.options.get("kinematics_options", {})
        if not isinstance(kinematics_options, dict):
            raise ValueError("policy.options.kinematics_options must be a mapping")
        self._kinematics = build_kinematics(kinematics_name, **kinematics_options)
        if self._kinematics.num_arm_joints != ARM_JOINTS:
            raise ValueError(
                f"OpenWAM YAM requires {ARM_JOINTS} arm joints, kinematics reports "
                f"{self._kinematics.num_arm_joints}"
            )
        self._anchors: OrderedDict[int, np.ndarray] = OrderedDict()

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._camera_map.values() if name not in snapshot.frames]
        if missing:
            raise ValueError(f"OpenWAM YAM adapter is missing cameras: {missing}")
        return snapshot

    def prepare_request(self, request: InferenceRequest) -> InferenceRequest:
        state_parts: list[np.ndarray] = []
        anchor_parts: list[np.ndarray] = []
        for group in self._group_order:
            values = np.asarray(request.observation.state.groups[group], dtype=np.float64)
            if values.shape != (GROUP_DIM,) or not np.isfinite(values).all():
                raise ValueError(f"OpenWAM YAM group {group!r} must have shape ({GROUP_DIM},)")
            pose = self._kinematics.fk(values[:ARM_JOINTS], float(values[-1]))
            state_parts.extend([pose[:3, 3], matrix_to_rot6d(pose[:3, :3]), values[-1:]])
            anchor_parts.append(values)

        self._anchors[request.request_seq] = np.concatenate(anchor_parts)
        while len(self._anchors) > 8:
            self._anchors.popitem(last=False)
        return OpenWAMInferenceRequest(
            session_id=request.session_id,
            request_seq=request.request_seq,
            observation_time_ns=request.observation_time_ns,
            deadline_ns=request.deadline_ns,
            observation=request.observation,
            instruction=request.instruction,
            openwam_state=np.ascontiguousarray(np.concatenate(state_parts), dtype=np.float32),
            openwam_prompt=self._prompt_template.format(instruction=request.instruction),
        )

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        actions = np.asarray(raw, dtype=np.float64)
        if actions.shape != (self._horizon_steps, ACTION_DIM):
            raise ValueError(
                f"OpenWAM YAM actions must have shape ({self._horizon_steps}, {ACTION_DIM}), "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("OpenWAM YAM actions contain non-finite values")
        anchor = self._anchors.pop(context.request_seq, None)
        if anchor is None:
            raise ValueError(
                f"OpenWAM YAM adapter has no observation anchor for request {context.request_seq}"
            )

        groups: dict[str, np.ndarray] = {}
        total_failures = 0
        for arm_index, group in enumerate(self._group_order):
            seed_state = anchor[arm_index * GROUP_DIM : (arm_index + 1) * GROUP_DIM]
            if context.measured_state is not None:
                measured = context.measured_state.groups.get(group)
                if measured is None:
                    raise ValueError(f"OpenWAM measured state is missing group {group!r}")
                seed_state = np.asarray(measured, dtype=np.float64)
            arm_actions = actions[
                :, arm_index * ARM_ACTION_DIM : (arm_index + 1) * ARM_ACTION_DIM
            ]
            groups[group], failures = self._solve_arm(group, arm_actions, seed_state)
            total_failures += failures

        return ActionChunk(
            plan_id=f"openwam-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
            metadata={
                "native_action_space": "absolute_eef20",
                "ik_failures": total_failures,
            },
        )

    def _solve_arm(
        self,
        group: str,
        actions: np.ndarray,
        seed_state: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        seed_state = np.asarray(seed_state, dtype=np.float64)
        if seed_state.shape != (GROUP_DIM,) or not np.isfinite(seed_state).all():
            raise ValueError(f"OpenWAM IK seed for {group!r} must have shape ({GROUP_DIM},)")
        out = np.empty((self._horizon_steps, GROUP_DIM), dtype=np.float64)
        current = np.ascontiguousarray(seed_state[:ARM_JOINTS])
        failures = 0
        for step, row in enumerate(actions):
            gripper = float(row[9])
            lower = self._gripper_min - self._gripper_tolerance
            upper = self._gripper_max + self._gripper_tolerance
            if not lower <= gripper <= upper:
                raise ValueError(
                    f"OpenWAM gripper for {group!r} is outside the physical contract: {gripper}"
                )
            gripper = float(np.clip(gripper, self._gripper_min, self._gripper_max))
            target = np.eye(4, dtype=np.float64)
            target[:3, 3] = row[:3]
            target[:3, :3] = rot6d_to_matrix(row[3:9])
            converged, raw_solved = self._kinematics.ik(target, current, gripper)
            solved = np.asarray(raw_solved, dtype=np.float64)
            if converged and solved.shape == (ARM_JOINTS,) and np.isfinite(solved).all():
                current = np.ascontiguousarray(solved)
            else:
                failures += 1
            out[step, :ARM_JOINTS] = current
            out[step, ARM_JOINTS] = gripper
        if failures:
            log.warning(
                "%s: IK did not converge on %d/%d OpenWAM steps",
                group,
                failures,
                len(actions),
            )
        return np.ascontiguousarray(out), failures

    def validate(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        del policy
        if tuple(robot.group_dims) != self._group_order:
            raise ValueError(
                "OpenWAM YAM requires robot groups in order "
                f"{list(self._group_order)}, got {list(robot.group_dims)}"
            )
        if any(robot.group_dims[name] != GROUP_DIM for name in self._group_order):
            raise ValueError("OpenWAM YAM requires two 7-value arm+gripper groups")


def build_model(config: PolicyConfig) -> OpenWAMWsPolicyModel:
    return OpenWAMWsPolicyModel(config)


def build_adapter(robot: RobotConfig, policy: PolicyConfig) -> OpenWAMYamAdapter:
    return OpenWAMYamAdapter(robot, policy)
