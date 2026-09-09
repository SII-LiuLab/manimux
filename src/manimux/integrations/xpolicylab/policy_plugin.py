"""ManiMux policy plugins backed by an XPolicyLab WebSocket policy server.

Same split as the MolmoAct/ABC/XR-1 plugins: the model plugin only speaks the
wire protocol, and every robot semantic -- group order, arm/gripper layout,
camera naming, chunk timing -- lives in the adapter.

What is different is that the server on the other end is not ours. XPolicyLab
serves one ``Model`` per policy out of its own environment, so a single plugin
pair reaches every policy in its zoo; which policy is running is decided when
that server is launched, not here.

Each request sends the observation and sampling parameters together through
XPolicyLab's ``INFER`` message. The server updates the model observation and
computes its action under one model lock, so observations cannot be interleaved
between separate RPCs.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence

import numpy as np

from manimux.config import PolicyConfig, RobotConfig
from manimux.integrations.xpolicylab.aac import (
    AacPreviousAction,
    EeActionStats,
    load_ee_action_stats,
    select_ee_chunk,
)
from manimux.integrations.xpolicylab.obs_codec import (
    GroupLayout,
    build_layouts,
    decode_action_steps,
    encode_observation,
)
from manimux.integrations.xpolicylab.ws_client import XPolicyLabWsClient
from manimux.kinematics.base import ArmKinematics
from manimux.policies.capabilities import PolicyCapabilities
from manimux.types import ActionChunk, ActionContext, InferenceRequest, ObservationSnapshot

DEFAULT_SERVER = "ws://127.0.0.1:8500"
DEFAULT_GROUP_ORDER = ("left_arm", "right_arm")
DEFAULT_GROUP_PREFIXES = {"left_arm": "left", "right_arm": "right"}
DEFAULT_CAMERA_MAP = {
    "cam_head": "front_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
DEFAULT_GRIPPER_DOFS = 1


def _string_option(options: Mapping[str, object], name: str, default: str) -> str:
    value = options.get(name, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"policy.options.{name} must be a non-empty string")
    return value


def _string_sequence(
    options: Mapping[str, object],
    name: str,
    default: Sequence[str],
) -> tuple[str, ...]:
    value = options.get(name, list(default))
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"policy.options.{name} must be a non-empty list of strings")
    return tuple(value)


def _string_mapping(
    options: Mapping[str, object],
    name: str,
    default: Mapping[str, str],
) -> dict[str, str]:
    value = options.get(name, dict(default))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"policy.options.{name} must be a non-empty mapping")
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise ValueError(f"policy.options.{name} must map strings to strings")
    return dict(value)


def _positive_float_option(options: Mapping[str, object], name: str, default: float) -> float:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"policy.options.{name} must be a number")
    if value <= 0:
        raise ValueError(f"policy.options.{name} must be positive")
    return float(value)


def _positive_int_option(options: Mapping[str, object], name: str, default: int) -> int:
    value = options.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"policy.options.{name} must be a non-negative integer")
    return value


def _bool_option(options: Mapping[str, object], name: str, default: bool) -> bool:
    value = options.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"policy.options.{name} must be a boolean")
    return value


def _layouts_from_options(
    options: Mapping[str, object],
    group_dims: Mapping[str, int],
) -> tuple[GroupLayout, ...]:
    return build_layouts(
        _string_sequence(options, "group_order", DEFAULT_GROUP_ORDER),
        _string_mapping(options, "group_prefixes", DEFAULT_GROUP_PREFIXES),
        group_dims,
        gripper_dofs=_positive_int_option(options, "gripper_dofs", DEFAULT_GRIPPER_DOFS),
    )


class XPolicyLabWsPolicyModel:
    """WebSocket inference backend; robot semantics stay in the adapter."""

    def __init__(self, config: PolicyConfig) -> None:
        options = config.options
        self._url = _string_option(options, "server", DEFAULT_SERVER)
        self._camera_map = _string_mapping(options, "camera_map", DEFAULT_CAMERA_MAP)
        # The observation carries the rate *we* run at. XPolicyLab treats the
        # environment as the owner of the control rate, so this is ours to
        # declare, and it comes from the same value the adapter uses for dt.
        self._frequency = 1.0 / config.effective_action_dt_s
        self._request_timeout_s = _positive_float_option(
            options, "request_timeout_s", config.timeout_s
        )
        self._connect_timeout_s = _positive_float_option(
            options, "connect_timeout_s", config.startup_timeout_s
        )
        # The model plugin never sees RobotConfig, so the arm/gripper split is
        # resolved from the first observation and then held fixed.
        self._group_order = _string_sequence(options, "group_order", DEFAULT_GROUP_ORDER)
        self._group_prefixes = _string_mapping(options, "group_prefixes", DEFAULT_GROUP_PREFIXES)
        self._gripper_dofs = _positive_int_option(options, "gripper_dofs", DEFAULT_GRIPPER_DOFS)
        self._horizon_steps = config.horizon_steps
        self._aac_kinematics_name = _string_option(options, "aac_kinematics", "yam")
        self._aac_kinematics: ArmKinematics | None = None
        self._aac_ee_stats: EeActionStats | None = None
        self._aac_ee_stats_path: str | None = None
        self._aac_previous: AacPreviousAction | None = None
        self._layouts: tuple[GroupLayout, ...] = ()
        self._session_id: str | None = None
        self._client: XPolicyLabWsClient | None = None

    def reset(self, session_id: str) -> None:
        self.close()
        client = XPolicyLabWsClient(
            url=self._url,
            evaluation_id=session_id,
            trial_id=session_id,
            connect_timeout_s=self._connect_timeout_s,
            request_timeout_s=self._request_timeout_s,
        )
        client.connect()
        client.reset()
        self._client = client
        self._session_id = session_id
        self._aac_previous = None

    def infer(self, request: InferenceRequest) -> object:
        if request.session_id != self._session_id:
            raise RuntimeError("XPolicyLab session is not initialized")
        client = self._client
        if client is None:
            raise RuntimeError("XPolicyLab client is not connected")

        snapshot = request.observation
        if not self._layouts:
            self._layouts = build_layouts(
                self._group_order,
                self._group_prefixes,
                {name: int(values.size) for name, values in snapshot.state.groups.items()},
                gripper_dofs=self._gripper_dofs,
            )
        observation = encode_observation(
            snapshot,
            layouts=self._layouts,
            camera_map=self._camera_map,
            instruction=request.instruction,
            frequency=self._frequency,
        )
        extra_state = getattr(request, "xpolicylab_state", None)
        if extra_state is not None:
            if not isinstance(extra_state, Mapping) or set(extra_state) & set(observation["state"]):
                raise ValueError("XPolicy additional state must not replace canonical joint state")
            observation["state"].update(extra_state)
        # Embodiment adapters (e.g. sapolicy_yam) may attach EE poses / intrinsics
        # that the generic joint-state codec does not carry.
        extra_info = getattr(request, "xpolicylab_additional_info", None)
        if isinstance(extra_info, Mapping) and extra_info:
            merged = dict(observation.get("additional_info") or {})
            merged.update(dict(extra_info))
            observation["additional_info"] = merged
        condition = getattr(request, "action_condition", None)
        weights = getattr(request, "condition_weights", None)
        if (condition is None) != (weights is None):
            raise ValueError("RTC action_condition and condition_weights must be provided together")
        aac_num_samples = getattr(request, "aac_num_samples", None)
        autohorizon = bool(getattr(request, "autohorizon", False))
        dvac = bool(getattr(request, "dvac", False))
        paint_prefix = getattr(request, "paint_action_prefix", None)
        paint_delay_steps = getattr(request, "paint_delay_steps", None)
        if (paint_prefix is None) != (paint_delay_steps is None):
            raise ValueError(
                "PAINT paint_action_prefix and paint_delay_steps must be provided together"
            )
        if paint_prefix is not None:
            if condition is not None or aac_num_samples is not None or autohorizon or dvac:
                raise ValueError(
                    "PAINT cannot be combined with RTC, AAC, AutoHorizon, or DVAC sampling"
                )
            prefix_array = np.asarray(paint_prefix, dtype=np.float32)
            assert paint_delay_steps is not None
            delay_steps = int(paint_delay_steps)
            expected_width = sum(layout.dim for layout in self._layouts)
            if (
                delay_steps <= 0
                or prefix_array.shape != (delay_steps, expected_width)
                or not np.isfinite(prefix_array).all()
            ):
                raise ValueError(
                    "PAINT action prefix must be finite with shape "
                    f"({delay_steps}, {expected_width}), got {prefix_array.shape}"
                )
            return client.infer(
                observation,
                sampling={
                    "mode": "paint",
                    "action_prefix": prefix_array,
                    "delay_steps": delay_steps,
                },
            )
        if aac_num_samples is not None:
            if condition is not None or autohorizon or dvac:
                raise ValueError("AAC cannot be combined with RTC, AutoHorizon, or DVAC sampling")
            result = client.infer(
                observation,
                sampling={
                    "mode": "aac",
                    "num_samples": int(aac_num_samples),
                },
            )
            if not isinstance(result, Mapping):
                raise ValueError("AAC server response must be a mapping")
            candidates = result.get("actions")
            if not isinstance(candidates, Sequence) or isinstance(candidates, str | bytes):
                raise ValueError("AAC server response must contain candidate chunks")
            if len(candidates) != int(aac_num_samples):
                raise ValueError(
                    f"AAC expected {int(aac_num_samples)} candidates, got {len(candidates)}"
                )
            if self._aac_kinematics is None:
                from manimux.kinematics import build_kinematics

                self._aac_kinematics = build_kinematics(self._aac_kinematics_name)
            stats_path = getattr(request, "aac_ee_stats_path", None)
            if not isinstance(stats_path, str) or not stats_path:
                raise ValueError("AAC requires aac_ee_stats_path")
            if self._aac_ee_stats is None or self._aac_ee_stats_path != stats_path:
                self._aac_ee_stats = load_ee_action_stats(stats_path, layouts=self._layouts)
                self._aac_ee_stats_path = stats_path
            selected, selection, self._aac_previous = select_ee_chunk(
                candidates,
                layouts=self._layouts,
                current_groups=snapshot.state.groups,
                kinematics=self._aac_kinematics,
                ee_stats=self._aac_ee_stats,
                motion_threshold=float(getattr(request, "aac_motion_threshold", 3.0)),
                chunk_id_selector=str(getattr(request, "aac_chunk_id_selector", "0")),
                previous=self._aac_previous,
                backward_beta=float(getattr(request, "aac_backward_beta", 0.99)),
            )
            return {"actions": selected, "aac": selection.metadata()}
        if autohorizon:
            if condition is not None or dvac:
                raise ValueError("AutoHorizon cannot be combined with RTC or DVAC sampling")
            return client.infer(observation, sampling={"mode": "autohorizon"})
        if dvac:
            if condition is not None:
                raise ValueError("DVAC and RTC sampling cannot be requested together")
            return client.infer(
                observation,
                sampling={
                    "mode": "dvac",
                    "tail_steps": int(getattr(request, "dvac_tail_steps", 5)),
                    "alpha": float(getattr(request, "dvac_alpha", 2.0)),
                    "rolling_window_size": int(
                        getattr(request, "dvac_rolling_window_size", 5)
                    ),
                    "min_execution_steps": int(
                        getattr(request, "dvac_min_execution_steps", 1)
                    ),
                    "max_execution_steps": int(
                        getattr(request, "dvac_max_execution_steps", self._horizon_steps)
                    ),
                },
            )
        if condition is None:
            return client.infer(observation, sampling={"mode": "default"})

        beta = float(getattr(request, "rtc_beta", 5.0))
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError(f"rtc_beta must be finite and positive, got {beta}")
        condition_array = np.asarray(condition, dtype=np.float32)
        weights_array = np.asarray(weights, dtype=np.float32)
        if (
            condition_array.ndim != 2
            or condition_array.shape[0] != self._horizon_steps
            or not np.isfinite(condition_array).all()
        ):
            raise ValueError(
                f"RTC action_condition must have shape ({self._horizon_steps}, native_dim), "
                f"got {condition_array.shape}"
            )
        if (
            weights_array.shape != (condition_array.shape[0],)
            or not np.isfinite(weights_array).all()
        ):
            raise ValueError(
                f"RTC condition_weights must have shape {(condition_array.shape[0],)}, "
                f"got {weights_array.shape}"
            )
        return client.infer(
            observation,
            sampling={
                "mode": "rtc",
                "action_condition": condition_array,
                "condition_weights": weights_array,
                "beta": beta,
            },
        )

    def close(self) -> None:
        client, self._client = self._client, None
        self._session_id = None
        if client is not None:
            client.close()

    def capabilities(self) -> PolicyCapabilities:
        client = self._client
        modes = (
            frozenset({"default"})
            if client is None
            else getattr(client, "sampling_modes", frozenset({"default"}))
        )
        metadata = (
            {}
            if client is None
            else getattr(client, "backend_metadata", {})
        )
        return PolicyCapabilities(sampling_modes=modes, backend_metadata=dict(metadata))


class XPolicyLabAdapter:
    """Translate canonical snapshots and XPolicyLab per-step action dictionaries."""

    def __init__(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        self._layouts = _layouts_from_options(policy.options, robot.group_dims)
        self._camera_map = _string_mapping(policy.options, "camera_map", DEFAULT_CAMERA_MAP)
        self._required_cameras = tuple(self._camera_map.values())
        self._action_dt_ns = int(policy.effective_action_dt_s * 1_000_000_000)
        self._horizon_steps = policy.horizon_steps
        self._allow_short_horizon = _bool_option(policy.options, "allow_short_horizon", False)

    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        missing = [name for name in self._required_cameras if name not in snapshot.frames]
        if missing:
            raise ValueError(f"XPolicyLab adapter is missing cameras: {missing}")
        return snapshot

    def decode_action(self, raw: object, context: ActionContext) -> ActionChunk:
        action_payload = raw.get("actions") if isinstance(raw, Mapping) else raw
        groups = decode_action_steps(action_payload, layouts=self._layouts)
        horizons = {values.shape[0] for values in groups.values()}
        horizon = next(iter(horizons)) if len(horizons) == 1 else None
        valid_short_horizon = (
            self._allow_short_horizon
            and horizon is not None
            and 2 <= horizon <= self._horizon_steps
        )
        if horizons != {self._horizon_steps} and not valid_short_horizon:
            raise ValueError(
                f"XPolicyLab action horizon must be {self._horizon_steps}"
                f"{' or 2..' + str(self._horizon_steps) if self._allow_short_horizon else ''}, "
                f"got {sorted(horizons)}"
            )
        return ActionChunk(
            plan_id=f"xpolicylab-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            request_seq=context.request_seq,
            observation_time_ns=context.observation_time_ns,
            created_time_ns=context.created_time_ns,
            action_space="joint_position",
            dt_ns=self._action_dt_ns,
            groups=groups,
        )

    def validate(self, robot: RobotConfig, policy: PolicyConfig) -> None:
        del policy
        expected = tuple(layout.group for layout in self._layouts)
        if tuple(robot.group_dims) != expected:
            raise ValueError(
                "XPolicyLab requires robot groups in order "
                f"{list(expected)}, got {list(robot.group_dims)}"
            )
        for layout in self._layouts:
            if robot.group_dims[layout.group] != layout.dim:
                raise ValueError(
                    f"group {layout.group!r} is {robot.group_dims[layout.group]} values "
                    f"but the layout describes {layout.dim}"
                )


def build_model(config: PolicyConfig) -> XPolicyLabWsPolicyModel:
    return XPolicyLabWsPolicyModel(config)


def build_adapter(robot: RobotConfig, policy: PolicyConfig) -> XPolicyLabAdapter:
    return XPolicyLabAdapter(robot, policy)
