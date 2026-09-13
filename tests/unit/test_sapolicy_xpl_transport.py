from __future__ import annotations

import time

import numpy as np
import pytest

from manimux.config import load_config
from manimux.integrations.sapolicy_yam.policy_plugin import (
    SAPolicyXPolicyRequest,
    build_adapter,
)
from manimux.types import ObservationSnapshot, RobotState, SensorFrame

CONFIG = "configs/sapolicy/yam/infra/manimux-xpl.yaml"


def _snapshot(now_ns: int) -> ObservationSnapshot:
    return ObservationSnapshot(
        state=RobotState(
            groups={
                "left_arm": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
                "right_arm": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
            },
            monotonic_ns=now_ns,
            sequence=1,
        ),
        frames={
            name: SensorFrame(
                name=name,
                data=np.zeros((48, 64, 3), dtype=np.uint8),
                capture_monotonic_ns=now_ns,
                sequence=1,
            )
            for name in ("front_camera", "left_camera", "right_camera")
        },
    )


def test_sapolicy_xpl_infra_uses_ws_transport() -> None:
    config = load_config(CONFIG)
    assert config.policy.worker == "xpolicylab_ws"
    assert config.policy.adapter == "sapolicy_yam"
    assert config.policy.horizon_steps == 16


def test_sapolicy_adapter_prepares_xpolicylab_additional_info() -> None:
    config = load_config(CONFIG)
    adapter = build_adapter(config.robot, config.policy)
    now_ns = time.monotonic_ns()
    from manimux.types import InferenceRequest

    prepared = adapter.prepare_request(
        InferenceRequest(
            session_id="session",
            request_seq=1,
            observation_time_ns=now_ns,
            deadline_ns=now_ns + 1_000_000_000,
            observation=_snapshot(now_ns),
            instruction="put bottles in bin",
        )
    )

    assert isinstance(prepared, SAPolicyXPolicyRequest)
    sap = prepared.xpolicylab_additional_info["sapolicy"]
    assert np.asarray(sap["left_endpose"]).shape == (7,)
    assert np.asarray(sap["right_endpose"]).shape == (7,)
    assert np.asarray(sap["intrinsics"]["top"]).shape == (3, 3)
    assert set(sap["intrinsics"]) == {"top", "left", "right"}
    # Wire RGB is stretch-resized; K stays at the native calibration.
    assert sap["image_native_hw"] == {
        "top": [48, 64],
        "left": [48, 64],
        "right": [48, 64],
    }
    for name in ("front_camera", "left_camera", "right_camera"):
        assert prepared.observation.frames[name].data.shape == (168, 224, 3)
        assert prepared.observation.frames[name].data.dtype == np.uint8


def test_rtc_condition_preserves_joint_fk_calibration_and_shared_transport():
    from manimux.integrations.xpolicylab.policy_plugin import build_model
    from manimux.runtime.rtc.request import RtcInferenceRequest

    config = load_config("configs/sapolicy/yam/infra/teleopMV51/top-rtc.yaml")
    adapter = build_adapter(config.robot, config.policy)
    now = time.monotonic_ns()
    snapshot = _snapshot(now)
    snapshot.state.groups["left_arm"][:6] = [0.2, -0.1, 0.3, 0.1, -0.2, 0.1]
    snapshot.state.groups["right_arm"][:6] = [-0.2, 0.1, -0.3, 0.2, 0.1, -0.1]
    rows = np.tile(np.concatenate(list(snapshot.state.groups.values())), (50, 1))
    request = RtcInferenceRequest(
        session_id="rtc",
        request_seq=2,
        observation_time_ns=now,
        deadline_ns=now + 5_000_000_000,
        observation=snapshot,
        instruction="bottles",
        action_condition=rows,
        condition_weights=np.linspace(1, 0, 50),
        rtc_beta=4.5,
    )
    prepared = adapter.prepare_request(request)
    assert isinstance(prepared, RtcInferenceRequest)
    for arm, side in enumerate(("left", "right")):
        np.testing.assert_allclose(
            prepared.action_condition[:, arm * 8 : arm * 8 + 7],
            np.tile(prepared.xpolicylab_state[f"{side}_ee_pose"], (50, 1)),
        )
        np.testing.assert_array_equal(
            prepared.action_condition[:, arm * 8 + 7], rows[:, arm * 7 + 6]
        )
    captured = {}

    class Client:
        def infer(self, observation, *, sampling):
            captured.update(sampling)
            return []

    worker = build_model(config.policy)
    worker._client = Client()
    worker._session_id = "rtc"
    worker.infer(prepared)
    assert captured["mode"] == "rtc" and captured["beta"] == 4.5
    np.testing.assert_allclose(captured["action_condition"], prepared.action_condition, atol=1e-7)
    np.testing.assert_allclose(captured["condition_weights"], request.condition_weights)
    with pytest.raises(ValueError, match="joint condition"):
        adapter._rtc_condition_to_ee(rows[:, :13])


@pytest.mark.parametrize("view", ["top", "gemini305", "gemini335"])
@pytest.mark.parametrize("suffix", ["", "-rtc"])
def test_mv51_smoothing_and_rtc_profiles_keep_rate_caps_disabled(view, suffix):
    from manimux.runtime.executors.smooth import SmoothExecutor
    from manimux.runtime.rtc.strategy import RtcInferenceStrategy

    config = load_config(f"configs/sapolicy/yam/infra/teleopMV51/{view}{suffix}.yaml")
    smooth = config.execution.smooth
    executor = SmoothExecutor(smooth, 1 / config.robot.control_hz)
    assert not executor.braking_tracking
    assert smooth.cutoff_hz == 8
    assert smooth.max_velocity is None and smooth.max_acceleration is None
    assert smooth.gripper.max_velocity is None and smooth.gripper.max_acceleration is None
    assert smooth.gripper.max_closing_velocity is None
    assert smooth.grasp_guard is None and smooth.release_guard is None
    assert config.policy.action_decoding == "process"
    assert config.policy.expected_backend.model["action_horizon"] == 50
    if suffix:
        assert config.execution.runtime == "rtc"
        assert config.execution.max_chunk_steps is None
        strategy = RtcInferenceStrategy(config)
        assert strategy.execution_horizon(50, 4) == 25
        assert strategy.required_sampling_modes == {"rtc"}
    else:
        assert config.execution.max_chunk_steps == 25
