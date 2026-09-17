"""Contract tests for the XPolicyLab bridge.

No server, no robot, no msgpack: every test here exercises the mapping between
canonical ManiMux types and XPolicyLab dictionaries, which is where an
integration like this actually goes wrong.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from manimux.embodiments.robot import robot_parameters
from manimux.integrations.xpolicylab.obs_codec import (
    DATA_FORMAT_VERSION,
    GroupLayout,
    build_layouts,
    decode_action_steps,
    encode_observation,
)
from manimux.integrations.xpolicylab.policy_plugin import build_adapter, build_model
from manimux.integrations.xpolicylab.ws_client import XPolicyLabWsClient, normalize_url
from manimux.integrations.xr1_yam.policy_plugin import build_adapter as build_xr1_adapter
from manimux.integrations.xr1_yam.policy_plugin import joint_condition_to_xr1_actions
from manimux.policies.base import policy_parameters
from manimux.runtime.aac import AacInferenceRequest
from manimux.runtime.dvac import DvacInferenceRequest
from manimux.runtime.rtc import RtcInferenceRequest
from manimux.types import ActionContext, ObservationSnapshot, RobotState, SensorFrame

LAYOUTS = (
    GroupLayout(group="left_arm", prefix="left", arm_dofs=6, gripper_dofs=1),
    GroupLayout(group="right_arm", prefix="right", arm_dofs=6, gripper_dofs=1),
)
CAMERA_MAP = {
    "cam_head": "front_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


def _frame(name: str, value: int) -> SensorFrame:
    return SensorFrame(
        name=name,
        data=np.full((4, 6, 3), value, dtype=np.uint8),
        capture_monotonic_ns=1,
        sequence=1,
    )


def _snapshot() -> ObservationSnapshot:
    return ObservationSnapshot(
        state=RobotState(
            groups={
                "left_arm": np.arange(7, dtype=np.float64),
                "right_arm": np.arange(7, dtype=np.float64) + 100.0,
            },
            monotonic_ns=1,
            sequence=1,
        ),
        frames={name: _frame(name, index) for index, name in enumerate(CAMERA_MAP.values())},
    )


def _action_steps(horizon: int) -> list[dict[str, np.ndarray]]:
    steps = []
    for step in range(horizon):
        steps.append(
            {
                "left_arm_joint_state": np.full(6, step, dtype=np.float32),
                "left_ee_joint_state": np.array([0.5], dtype=np.float32),
                "right_arm_joint_state": np.full(6, -step, dtype=np.float32),
                "right_ee_joint_state": np.array([0.25], dtype=np.float32),
            }
        )
    return steps


def _policy_config(**options: object) -> dict:
    merged: dict[str, object] = {
        "server": "ws://127.0.0.1:8500",
        "group_order": ["left_arm", "right_arm"],
        "group_prefixes": {"left_arm": "left", "right_arm": "right"},
        "gripper_dofs": 1,
        "camera_map": dict(CAMERA_MAP),
    }
    merged.update(options)
    return policy_parameters(
        worker="xpolicylab_ws",
        adapter="xpolicylab",
        action_dt_s=0.05,
        horizon_steps=30,
        options=merged,
    )


def _robot_config() -> dict:
    return robot_parameters(
        driver="mock",
        control_hz=30.0,
        group_dims={"left_arm": 7, "right_arm": 7},
    )


# --------------------------------------------------------------------- encode


def test_encode_splits_each_group_into_arm_and_gripper() -> None:
    observation = encode_observation(
        _snapshot(),
        layouts=LAYOUTS,
        camera_map=CAMERA_MAP,
        instruction="pick it up",
        frequency=30.0,
    )
    state = observation["state"]
    assert set(state) == {
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    }
    np.testing.assert_allclose(state["left_arm_joint_state"], np.arange(6))
    np.testing.assert_allclose(state["left_ee_joint_state"], [6.0])
    np.testing.assert_allclose(state["right_arm_joint_state"], np.arange(6) + 100.0)
    np.testing.assert_allclose(state["right_ee_joint_state"], [106.0])
    assert state["left_arm_joint_state"].dtype == np.float32


def test_encode_declares_our_control_rate_and_format_version() -> None:
    observation = encode_observation(
        _snapshot(),
        layouts=LAYOUTS,
        camera_map=CAMERA_MAP,
        instruction="",
        frequency=30.0,
    )
    assert observation["additional_info"] == {"frequency": 30.0}
    assert observation["data_format_version"] == DATA_FORMAT_VERSION
    assert observation["env_idx"] == 0


def test_encode_maps_camera_names_onto_the_wire_names() -> None:
    observation = encode_observation(
        _snapshot(),
        layouts=LAYOUTS,
        camera_map=CAMERA_MAP,
        instruction="",
        frequency=30.0,
    )
    assert set(observation["vision"]) == set(CAMERA_MAP)
    head = observation["vision"]["cam_head"]
    assert head["color"].shape == (4, 6, 3)
    assert head["color"].dtype == np.uint8
    assert head["shape"] == [4, 6]


def test_encode_rejects_a_missing_camera() -> None:
    snapshot = _snapshot()
    del snapshot.frames["front_camera"]
    with pytest.raises(ValueError, match="missing camera 'front_camera'"):
        encode_observation(
            snapshot,
            layouts=LAYOUTS,
            camera_map=CAMERA_MAP,
            instruction="",
            frequency=30.0,
        )


def test_encode_rejects_a_group_of_the_wrong_width() -> None:
    snapshot = _snapshot()
    snapshot.state.groups["left_arm"] = np.zeros(6)
    with pytest.raises(ValueError, match="must have 7 values"):
        encode_observation(
            snapshot,
            layouts=LAYOUTS,
            camera_map=CAMERA_MAP,
            instruction="",
            frequency=30.0,
        )


# --------------------------------------------------------------------- decode


def test_decode_stacks_per_step_dicts_into_group_trajectories() -> None:
    groups = decode_action_steps(_action_steps(30), layouts=LAYOUTS)
    assert set(groups) == {"left_arm", "right_arm"}
    assert groups["left_arm"].shape == (30, 7)
    np.testing.assert_allclose(groups["left_arm"][3], [3, 3, 3, 3, 3, 3, 0.5])
    np.testing.assert_allclose(groups["right_arm"][3], [-3, -3, -3, -3, -3, -3, 0.25])


def test_decode_rejects_an_empty_chunk() -> None:
    with pytest.raises(ValueError, match="empty action chunk"):
        decode_action_steps([], layouts=LAYOUTS)


def test_decode_rejects_a_missing_key() -> None:
    steps = _action_steps(2)
    del steps[1]["right_ee_joint_state"]
    with pytest.raises(KeyError, match="right_ee_joint_state"):
        decode_action_steps(steps, layouts=LAYOUTS)


def test_decode_rejects_a_wrong_width() -> None:
    steps = _action_steps(2)
    steps[0]["left_arm_joint_state"] = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError, match="must have 6 values"):
        decode_action_steps(steps, layouts=LAYOUTS)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_decode_rejects_non_finite_actions(bad: float) -> None:
    steps = _action_steps(2)
    steps[0]["left_arm_joint_state"] = np.full(6, bad, dtype=np.float64)
    with pytest.raises(ValueError, match="not finite"):
        decode_action_steps(steps, layouts=LAYOUTS)


# -------------------------------------------------------------------- adapter


def test_adapter_produces_a_canonical_joint_position_chunk() -> None:
    adapter = build_adapter(_robot_config(), _policy_config())
    chunk = adapter.decode_action(
        _action_steps(30),
        ActionContext(request_seq=7, observation_time_ns=11, created_time_ns=12),
    )
    assert chunk.action_space == "joint_position"
    assert chunk.dt_ns == 50_000_000
    assert chunk.horizon_steps == 30
    assert chunk.request_seq == 7
    assert chunk.observation_time_ns == 11
    assert chunk.plan_id.startswith("xpolicylab-7-")
    assert set(chunk.groups) == {"left_arm", "right_arm"}


def test_adapter_rejects_an_unexpected_action_horizon() -> None:
    adapter = build_adapter(_robot_config(), _policy_config())
    with pytest.raises(ValueError, match="action horizon must be 30"):
        adapter.decode_action(
            _action_steps(29),
            ActionContext(request_seq=7, observation_time_ns=11, created_time_ns=12),
        )


def test_adapter_accepts_aac_selected_short_horizon_when_enabled() -> None:
    adapter = build_adapter(_robot_config(), _policy_config(allow_short_horizon=True))
    chunk = adapter.decode_action(
        _action_steps(7),
        ActionContext(request_seq=7, observation_time_ns=11, created_time_ns=12),
    )
    assert chunk.horizon_steps == 7


def test_adapter_unwraps_aac_metadata_without_changing_actions() -> None:
    adapter = build_adapter(_robot_config(), _policy_config(allow_short_horizon=True))
    chunk = adapter.decode_action(
        {
            "actions": _action_steps(5),
            "aac": {"chunk_id": 0, "entropy_elbow": 2, "motion_floor": 5},
        },
        ActionContext(request_seq=7, observation_time_ns=11, created_time_ns=12),
    )
    assert chunk.horizon_steps == 5


def test_adapter_validate_rejects_a_group_order_mismatch() -> None:
    adapter = build_adapter(_robot_config(), _policy_config())
    swapped = robot_parameters(
        driver="mock",
        control_hz=30.0,
        group_dims={"right_arm": 7, "left_arm": 7},
    )
    with pytest.raises(ValueError, match="requires robot groups in order"):
        adapter.validate(swapped, _policy_config())


def test_adapter_build_observation_rejects_a_missing_camera() -> None:
    adapter = build_adapter(_robot_config(), _policy_config())
    snapshot = _snapshot()
    del snapshot.frames["left_camera"]
    with pytest.raises(ValueError, match="missing cameras"):
        adapter.build_observation(snapshot)


# --------------------------------------------------------------------- layout


def test_build_layouts_derives_the_arm_width_from_group_dims() -> None:
    layouts = build_layouts(
        ["left_arm"],
        {"left_arm": "left"},
        {"left_arm": 7},
        gripper_dofs=1,
    )
    assert layouts[0].arm_dofs == 6
    assert layouts[0].arm_key == "left_arm_joint_state"
    assert layouts[0].gripper_key == "left_ee_joint_state"


def test_build_layouts_rejects_a_group_with_no_arm_joints_left() -> None:
    with pytest.raises(ValueError, match="leaves no arm joints"):
        build_layouts(["left_arm"], {"left_arm": "left"}, {"left_arm": 1}, gripper_dofs=1)


def test_build_layouts_rejects_an_unmapped_group() -> None:
    with pytest.raises(KeyError, match="right_arm"):
        build_layouts(
            ["right_arm"],
            {"left_arm": "left"},
            {"right_arm": 7},
            gripper_dofs=1,
        )


def test_single_arm_layout_drops_the_prefix() -> None:
    layout = GroupLayout(group="arm", prefix="", arm_dofs=6, gripper_dofs=1)
    assert layout.arm_key == "arm_joint_state"
    assert layout.gripper_key == "ee_joint_state"


# ------------------------------------------------------------------ ws client


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("ws://127.0.0.1:8500", "ws://127.0.0.1:8500"),
        ("http://127.0.0.1:8500", "ws://127.0.0.1:8500"),
        ("https://host:9000", "wss://host:9000"),
        ("127.0.0.1:8500", "ws://127.0.0.1:8500"),
        ("ws://127.0.0.1:8500/", "ws://127.0.0.1:8500"),
    ],
)
def test_normalize_url_accepts_the_usual_spellings(given: str, expected: str) -> None:
    assert normalize_url(given) == expected


def test_normalize_url_rejects_an_unusable_scheme() -> None:
    with pytest.raises(ValueError, match="ws:// or wss://"):
        normalize_url("ftp://host:1")


def test_ws_client_preserves_aac_metadata_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = XPolicyLabWsClient(
        url="ws://127.0.0.1:8500",
        evaluation_id="evaluation",
        trial_id="trial",
    )
    monkeypatch.setattr(
        client,
        "request",
        lambda *_args, **_kwargs: {
            "payload": {
                "actions": [_action_steps(5), _action_steps(5)],
                "latency_ms": 12.0,
            }
        },
    )

    result = client.infer({}, sampling={"mode": "aac"})
    assert isinstance(result, dict)
    assert result["latency_ms"] == 12.0
    assert len(result["actions"]) == 2
    assert len(result["actions"][0]) == 5


def test_ws_client_preserves_dvac_metadata_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = XPolicyLabWsClient(
        url="ws://127.0.0.1:8500",
        evaluation_id="evaluation",
        trial_id="trial",
    )
    monkeypatch.setattr(
        client,
        "request",
        lambda *_args, **_kwargs: {
            "payload": {
                "actions": _action_steps(30),
                "dvac": {"execution_steps": 7, "threshold": 0.01},
            }
        },
    )

    result = client.infer({}, sampling={"mode": "dvac"})

    assert isinstance(result, dict)
    assert len(result["actions"]) == 30
    assert result["dvac"]["execution_steps"] == 7


def test_ws_client_preserves_explicit_action_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = XPolicyLabWsClient(
        url="ws://127.0.0.1:8500",
        evaluation_id="evaluation",
        trial_id="trial",
    )
    monkeypatch.setattr(
        client,
        "request",
        lambda *_args, **_kwargs: {
            "payload": {
                "actions": _action_steps(5),
                "action_semantics": "anchor_relative_arm_absolute_gripper",
            }
        },
    )

    result = client.infer({}, sampling={"mode": "default"})

    assert isinstance(result, dict)
    assert result["action_semantics"] == "anchor_relative_arm_absolute_gripper"
    assert len(result["actions"]) == 5


# --------------------------------------------------------------- model wiring


def test_model_is_constructible_without_a_server() -> None:
    model = build_model(_policy_config())
    assert model is not None


def test_model_capabilities_preserve_policy_server_fingerprint() -> None:
    model = build_model(_policy_config())
    model._client = SimpleNamespace(  # type: ignore[attr-defined]
        sampling_modes=frozenset({"default", "rtc"}),
        backend_metadata={
            "server_revision": "xpolicy-sha",
            "model": {"model_root": "/checkpoints/pi05-step-1000"},
        },
    )

    capabilities = model.capabilities()

    assert capabilities.sampling_modes == frozenset({"default", "rtc"})
    assert capabilities.backend_metadata["server_revision"] == "xpolicy-sha"
    assert capabilities.backend_metadata["model"] == {"model_root": "/checkpoints/pi05-step-1000"}


def test_model_infer_refuses_an_uninitialised_session() -> None:
    from manimux.types import InferenceRequest

    model = build_model(_policy_config())
    request = InferenceRequest(
        session_id="nope",
        request_seq=1,
        observation_time_ns=1,
        deadline_ns=2,
        observation=_snapshot(),
    )
    with pytest.raises(RuntimeError, match="session is not initialized"):
        model.infer(request)


def test_model_maps_aac_request_to_xpolicy_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import manimux.kinematics

    monkeypatch.setattr(
        manimux.kinematics,
        "build_kinematics",
        lambda *_args, **_kwargs: _LinearKinematics(),
    )
    model = build_model(_policy_config(allow_short_horizon=True))
    model._session_id = "session"
    captured: dict[str, object] = {}

    class _Client:
        def infer(self, observation: object, *, sampling: object) -> dict[str, object]:
            captured["observation"] = observation
            captured["sampling"] = sampling
            return {"actions": [_action_steps(30) for _ in range(20)]}

    model._client = _Client()
    request = AacInferenceRequest(
        session_id="session",
        request_seq=1,
        observation_time_ns=1,
        deadline_ns=2**62,
        observation=_snapshot(),
        instruction="task",
        aac_num_samples=20,
        aac_motion_threshold=3.0,
        aac_ee_stats_path=(
            "src/manimux/integrations/xpolicylab/norm_stats/yam_60ep_ee_increment.json"
        ),
        aac_chunk_id_selector="mean",
        aac_backward_beta=0.99,
    )
    result = model.infer(request)

    assert captured["sampling"] == {
        "mode": "aac",
        "num_samples": 20,
    }
    assert isinstance(result, dict)
    assert 2 <= len(result["actions"]) <= 30
    assert result["aac"]["metric_space"] == "dual_arm_incremental_ee_action_mean"


def test_model_maps_dvac_request_to_xpolicy_sampling() -> None:
    model = build_model(_policy_config())
    model._session_id = "session"
    captured: dict[str, object] = {}

    class _Client:
        def infer(self, observation: object, *, sampling: object) -> dict[str, object]:
            captured["observation"] = observation
            captured["sampling"] = sampling
            return {
                "actions": _action_steps(30),
                "dvac": {"execution_steps": 7},
            }

    model._client = _Client()
    request = DvacInferenceRequest(
        session_id="session",
        request_seq=1,
        observation_time_ns=1,
        deadline_ns=2**62,
        observation=_snapshot(),
        instruction="task",
        dvac_tail_steps=5,
        dvac_alpha=2.0,
        dvac_rolling_window_size=5,
        dvac_min_execution_steps=1,
        dvac_max_execution_steps=30,
    )

    result = model.infer(request)

    assert captured["sampling"] == {
        "mode": "dvac",
        "tail_steps": 5,
        "alpha": 2.0,
        "rolling_window_size": 5,
        "min_execution_steps": 1,
        "max_execution_steps": 30,
    }
    assert result["dvac"]["execution_steps"] == 7


class _LinearKinematics:
    num_arm_joints = 6

    def fk(self, joints: np.ndarray, gripper: float) -> np.ndarray:
        del gripper
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(joints)[:3]
        return pose


def test_xr1_condition_codec_maps_joint_waypoints_to_native_ee_deltas() -> None:
    anchors = {
        "left_arm": np.zeros(7),
        "right_arm": np.zeros(7),
    }
    condition = np.zeros((30, 14))
    condition[:, 0] = 0.1
    condition[:, 6] = 0.2
    condition[:, 7 + 1] = -0.3
    condition[:, 13] = -0.4

    actions = joint_condition_to_xr1_actions(
        condition,
        anchors,
        group_order=("left_arm", "right_arm"),
        kinematics=_LinearKinematics(),
    )

    assert actions.shape == (30, 60)
    np.testing.assert_allclose(actions[:, 0], 0.1)
    np.testing.assert_allclose(actions[:, 6], 0.2)
    np.testing.assert_allclose(actions[:, 9], -0.3)
    np.testing.assert_allclose(actions[:, 14], -0.4)
    np.testing.assert_allclose(actions[:, 20:], 0.0)


def test_xr1_adapter_encodes_rtc_condition_before_generic_xpolicy_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import manimux.kinematics

    monkeypatch.setattr(
        manimux.kinematics,
        "build_kinematics",
        lambda *_args, **_kwargs: _LinearKinematics(),
    )
    policy = _policy_config(action_codec="xr1_ee_delta", kinematics="yam")
    model = build_model(policy)
    adapter = build_xr1_adapter(_robot_config(), policy)
    model._session_id = "session"
    captured: dict[str, object] = {}

    class _Client:
        def infer(self, observation: object, *, sampling: object) -> np.ndarray:
            captured["observation"] = observation
            captured["sampling"] = sampling
            return np.zeros((30, 60), dtype=np.float32)

    model._client = _Client()
    snapshot = _snapshot()
    condition = np.concatenate(
        [snapshot.state.groups["left_arm"], snapshot.state.groups["right_arm"]]
    )
    condition = np.tile(condition, (30, 1))
    request = RtcInferenceRequest(
        session_id="session",
        request_seq=1,
        observation_time_ns=1,
        deadline_ns=2**62,
        observation=snapshot,
        instruction="task",
        action_condition=condition,
        condition_weights=np.ones(30),
        rtc_beta=5.0,
    )
    prepared = adapter.prepare_request(request)
    raw = model.infer(prepared)

    sampling = captured["sampling"]
    assert isinstance(sampling, dict)
    assert sampling["mode"] == "rtc"
    assert sampling["action_condition"].shape == (30, 60)
    np.testing.assert_allclose(sampling["action_condition"], 0.0)
    assert isinstance(raw, np.ndarray)
    assert raw.shape == (30, 60)
    np.testing.assert_allclose(adapter._anchors[1], condition[0])
