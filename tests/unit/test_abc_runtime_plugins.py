from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from manimux.embodiments.sensor.camera_server import CameraServerSensorDriver
from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot
from manimux.embodiments.robot.yam import YamRobot
from manimux.embodiments.sensor import build_sensor
from manimux.policies.abc import AbcHttpPolicyModel
from manimux.policies import build_policy_model
from manimux.policies.base import action_interval
from manimux.policy_adapter import build_policy_adapter
from manimux.policy_adapter.abc_yam import AbcYamAdapter
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)


def test_abc_run_config_swaps_only_the_policy_layer() -> None:
    """The point of the plugin split: ABC reuses YAM, the cameras and the viewer."""
    abc = load_config(Path("manimux/configs/experiments/put_bottles/yam_abc_manimux.yaml"))
    molmoact = load_config(Path("manimux/configs/experiments/pick_red_object/yam_molmoact2_manimux.yaml"))

    assert isinstance(build_robot(abc["robot"], SystemClock()), YamRobot)
    assert isinstance(build_sensor(abc["sensors"][0], SystemClock()), CameraServerSensorDriver)
    assert isinstance(build_policy_model(abc["policy"]), AbcHttpPolicyModel)
    assert isinstance(build_policy_adapter(abc["robot"], abc["policy"]), AbcYamAdapter)

    assert abc["robot"]["type"] == molmoact["robot"]["type"]
    assert abc["sensors"][0]["driver"] == molmoact["sensors"][0]["driver"]
    assert abc["viewer"]["robot"] == molmoact["viewer"]["robot"]
    assert (abc["policy"]["worker"], abc["policy"]["adapter"]) != (
        molmoact["policy"]["worker"],
        molmoact["policy"]["adapter"],
    )


def test_abc_adapter_splits_raw_actions_into_canonical_yam_groups() -> None:
    config = load_config("manimux/configs/experiments/put_bottles/yam_abc_manimux.yaml")
    adapter = build_policy_adapter(config["robot"], config["policy"])
    raw = np.arange(30 * 14, dtype=np.float64).reshape(30, 14)

    chunk = adapter.decode_action(
        raw,
        ActionContext(request_seq=4, observation_time_ns=10, created_time_ns=20),
    )

    assert chunk.action_space == "joint_position"
    assert chunk.request_seq == 4
    assert chunk.plan_id.startswith("abc-")
    assert chunk.dt_ns == int(config["policy"]["action_dt_s"] * 1_000_000_000)
    np.testing.assert_array_equal(chunk.groups["left_arm"], raw[:, :7])
    np.testing.assert_array_equal(chunk.groups["right_arm"], raw[:, 7:])


def test_abc_adapter_rejects_wrong_action_width() -> None:
    config = load_config("manimux/configs/experiments/put_bottles/yam_abc_manimux.yaml")
    adapter = build_policy_adapter(config["robot"], config["policy"])

    with pytest.raises(ValueError, match="shape"):
        adapter.decode_action(
            np.zeros((30, 13)),
            ActionContext(request_seq=1, observation_time_ns=10, created_time_ns=20),
        )


def test_abc_live_config_matches_the_checkpoint_timing() -> None:
    config = load_config("manimux/configs/experiments/put_bottles/yam_abc_manimux.yaml")

    # ABC-DiT was trained at 30 Hz with a fixed chunk_length of 30.
    assert config["run"]["task"] == "put the plastic bottles in the bin"
    assert config["policy"]["horizon_steps"] == 30
    assert action_interval(config["policy"]) == pytest.approx(1.0 / 30.0, abs=1e-4)
    assert config["executor"]["type"] == "direct"
    assert config["inference"]["blend_steps"] == 0
    assert config["robot"]["options"]["home_on_close"] is True


class _StubResponse:
    def __init__(self, body: str) -> None:
        self.status_code = 200
        self.text = body


def _snapshot() -> ObservationSnapshot:
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    return ObservationSnapshot(
        state=RobotState(
            groups={"left_arm": np.arange(7.0), "right_arm": np.arange(7.0, 14.0)},
            monotonic_ns=1,
            sequence=1,
        ),
        frames={
            name: SensorFrame(name=name, data=frame, capture_monotonic_ns=1, sequence=1)
            for name in ("left_camera", "front_camera", "right_camera")
        },
    )


def test_abc_http_model_posts_the_server_wire_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    import json_numpy

    config = load_config("manimux/configs/experiments/put_bottles/yam_abc_manimux.yaml")
    model = build_policy_model(config["policy"])
    model._session_id = "session"

    captured: dict[str, object] = {}

    def _post(url: str, **kwargs: object) -> _StubResponse:
        captured["url"] = url
        captured["payload"] = json_numpy.loads(kwargs["data"])
        return _StubResponse(json_numpy.dumps({"actions": np.zeros((30, 14), dtype=np.float32)}))

    stub = types.ModuleType("requests")
    stub.post = _post  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", stub)

    actions = model.infer(
        InferenceRequest(
            session_id="session",
            request_seq=1,
            observation_time_ns=0,
            deadline_ns=2**62,
            observation=_snapshot(),
            instruction="put the bottle into the bin",
        )
    )

    assert captured["url"] == "http://127.0.0.1:8300/act"
    payload = captured["payload"]
    assert set(payload) == {
        "top_cam",
        "left_cam",
        "right_cam",
        "timestamp",
        "instruction",
        "state",
        "num_steps",
    }
    # ABC's `top` camera is the front-facing scene camera, not a wrist camera.
    assert payload["top_cam"].shape == (4, 5, 3)
    assert payload["instruction"] == "put the bottle into the bin"
    np.testing.assert_array_equal(payload["state"], np.arange(14.0))
    assert payload["num_steps"] == 10
    assert np.asarray(actions).shape == (30, 14)


def test_abc_letterbox_matches_the_training_time_ffmpeg_geometry() -> None:
    """ABC's cache was built with ffmpeg decrease-then-centre-pad to 224x224."""
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from manimux.integrations.abc_yam.host_server_abc import _letterbox, _to_chw_uint8

    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    out = _letterbox(frame)

    assert out.shape == (224, 224, 3)
    ratio = max(640 / 224, 360 / 224)
    inner_h = int(360 / ratio)
    top = (224 - inner_h) // 2
    content_rows = np.where(out.reshape(224, -1).any(axis=1))[0]
    assert content_rows.min() == top
    assert content_rows.max() == top + inner_h - 1
    # zero padding, not edge replication
    assert not out[:top].any() and not out[top + inner_h :].any()

    # already-224 frames pass straight through, and the model gets CHW
    square = np.zeros((224, 224, 3), dtype=np.uint8)
    assert _letterbox(square) is square
    assert _to_chw_uint8(frame).shape == (3, 224, 224)
