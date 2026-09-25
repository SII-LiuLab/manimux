from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from manimux.cli import load_config, prepare_experiment
from manimux.policies.base import action_interval
from manimux.runtime.inference import RequestState, build_inference_strategy
from manimux.runtime.paint import PaintInferenceRequest
from manimux.runtime.safety import RuntimeState
from manimux.runtime.timeline import ActionTimeline
from manimux.types import (
    ActionChunk,
    InferenceRequest,
    InferenceResponse,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)


class _Adapter:
    def build_observation(self, snapshot: ObservationSnapshot) -> ObservationSnapshot:
        return snapshot


def _config() -> dict:
    return load_config("manimux/configs/experiments/pick_red_object/pi05/yam_pi05_paint_step1000.yaml")


def _snapshot(now_ns: int) -> ObservationSnapshot:
    return ObservationSnapshot(
        state=RobotState(
            groups={"left_arm": np.zeros(7), "right_arm": np.zeros(7)},
            monotonic_ns=now_ns,
            sequence=1,
        ),
        frames={
            name: SensorFrame(
                name=name,
                data=np.zeros((2, 2, 3), dtype=np.uint8),
                capture_monotonic_ns=now_ns,
                sequence=1,
            )
            for name in ("left_camera", "front_camera", "right_camera")
        },
    )


def _chunk(*, request_seq: int, observation_time_ns: int, dt_ns: int) -> ActionChunk:
    rows = np.arange(50 * 14, dtype=np.float64).reshape(50, 14)
    return ActionChunk(
        plan_id=f"plan-{request_seq}",
        request_seq=request_seq,
        observation_time_ns=observation_time_ns,
        created_time_ns=observation_time_ns,
        action_space="joint_position",
        dt_ns=dt_ns,
        groups={"left_arm": rows[:, :7], "right_arm": rows[:, 7:]},
    )


def test_paint_request_preserves_core_worker_contract() -> None:
    request = PaintInferenceRequest(
        session_id="session",
        request_seq=2,
        observation_time_ns=1,
        deadline_ns=2,
        observation=_snapshot(1),
        instruction="task",
        paint_action_prefix=np.zeros((4, 14)),
        paint_delay_steps=4,
        paint_execution_steps=12,
    )

    assert isinstance(request, InferenceRequest)
    restored = pickle.loads(pickle.dumps(request))
    assert restored.paint_action_prefix.shape == (4, 14)
    assert restored.paint_delay_steps == 4


def test_xpolicy_plugin_maps_paint_request_to_sampling_payload() -> None:
    from manimux.policies import build_policy_model

    config = _config()
    model = build_policy_model(config["policy"])
    model._session_id = "session"
    captured: dict[str, object] = {}

    class _Client:
        def infer(self, observation: object, *, sampling: object) -> dict[str, object]:
            captured["observation"] = observation
            captured["sampling"] = sampling
            return {"actions": [], "paint": {"delay_steps": 3}}

    model._client = _Client()
    request = PaintInferenceRequest(
        session_id="session",
        request_seq=2,
        observation_time_ns=1,
        deadline_ns=2,
        observation=_snapshot(1),
        instruction="task",
        paint_action_prefix=np.arange(3 * 14, dtype=np.float64).reshape(3, 14),
        paint_delay_steps=3,
        paint_execution_steps=12,
    )

    result = model.infer(request)

    sampling = captured["sampling"]
    assert isinstance(sampling, dict)
    assert sampling["mode"] == "paint"
    assert sampling["delay_steps"] == 3
    np.testing.assert_array_equal(
        sampling["action_prefix"],
        request.paint_action_prefix.astype(np.float32),
    )
    assert result["paint"] == {"delay_steps": 3}


def test_paint_strategy_sends_exact_old_chunk_prefix() -> None:
    config = _config()
    strategy = build_inference_strategy(config)
    timeline = ActionTimeline(config["robot"]["group_dims"])
    dt_ns = int(action_interval(config["policy"]) * 1_000_000_000)
    chunk = _chunk(request_seq=1, observation_time_ns=0, dt_ns=dt_ns)
    commit = timeline.commit(
        chunk,
        now_ns=0,
        commit_lead_ns=0,
        max_plan_age_ns=10**12,
        current_command={"left_arm": np.zeros(7), "right_arm": np.zeros(7)},
        blend_steps=0,
    )
    assert commit.accepted
    strategy.on_plan_accepted(
        chunk=chunk,
        result=commit,
        response=InferenceResponse(
            session_id="session",
            request_seq=1,
            observation_time_ns=0,
            finished_time_ns=0,
            inference_ms=0.0,
            raw_action=[],
        ),
        now_ns=0,
    )

    now_ns = config["inference"]["paint"]["execution_policy_steps"] * dt_ns
    submission = strategy.build_submission(
        session_id="session",
        request_seq=2,
        now_ns=now_ns,
        snapshot=_snapshot(now_ns),
        adapter=_Adapter(),
        timeline=timeline,
        request_state=RequestState(False, 1, 0),
        runtime_state=RuntimeState.RUNNING,
    )

    assert submission is not None
    request = submission.request
    assert isinstance(request, PaintInferenceRequest)
    assert request.paint_execution_steps == 12
    assert request.paint_delay_steps == 10
    packed = np.concatenate([chunk.groups["left_arm"], chunk.groups["right_arm"]], axis=1)
    np.testing.assert_array_equal(request.paint_action_prefix, packed[12:22])


def test_paint_rejects_response_beyond_anchored_prefix() -> None:
    config = _config()
    strategy = build_inference_strategy(config)
    strategy._request_forecast[2] = 3
    dt_ns = int(action_interval(config["policy"]) * 1_000_000_000)
    chunk = _chunk(request_seq=2, observation_time_ns=0, dt_ns=dt_ns)
    response = InferenceResponse(
        session_id="session",
        request_seq=2,
        observation_time_ns=0,
        finished_time_ns=4 * dt_ns,
        inference_ms=100.0,
        raw_action=[],
    )

    with pytest.raises(ValueError, match="beyond its anchored prefix"):
        strategy.prepare_chunk(chunk=chunk, response=response, now_ns=4 * dt_ns)


def test_paint_config_requires_feasible_s_and_d() -> None:
    payload = deepcopy(_config())
    payload["inference"].pop("inference_schedule")
    payload["inference"].pop("refill_threshold_s")
    payload["inference"]["paint"]["execution_policy_steps"] = 45

    with pytest.raises(ValueError, match="PAINT requires"):
        prepare_experiment(**payload)


def test_paint_uses_configured_seam_blending() -> None:
    payload = deepcopy(_config())
    payload["inference"].pop("inference_schedule")
    payload["inference"].pop("refill_threshold_s")
    payload["inference"]["blend_policy_steps"] = 1

    config = prepare_experiment(**payload)
    settings = build_inference_strategy(config).commit_settings(
        response=InferenceResponse("s", 1, 1, 1.0, raw_action=[]),
        measured={"left_arm": np.zeros(7), "right_arm": np.zeros(7)},
        last_command={"left_arm": np.ones(7), "right_arm": np.ones(7)},
    )
    assert settings.blend_steps == 1


def test_paint_config_is_loadable_and_uses_edge_runtime(tmp_path: Path) -> None:
    from manimux.runtime import build_runtime
    from manimux.runtime.edge import EdgeRuntime

    config = _config()
    # 验证调度构造，不依赖 YAM 硬件 SDK 或相机。
    config["robot"]["type"] = "tests.support.robot:build_robot"
    config["policy"]["adapter"]["type"] = "manimux.policies.fake:FakePolicyAdapter"
    config["sensors"] = []
    runtime = build_runtime(config, tmp_path)

    assert isinstance(runtime, EdgeRuntime)
    assert runtime._strategy.name == "paint"
