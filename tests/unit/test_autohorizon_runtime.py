from __future__ import annotations

import numpy as np

from manimux.cli import load_config
from manimux.runtime.autohorizon import AutoHorizonInferenceStrategy
from manimux.types import ActionChunk, InferenceResponse


def _chunk(horizon: int = 50) -> ActionChunk:
    return ActionChunk(
        plan_id="plan",
        request_seq=1,
        observation_time_ns=10,
        created_time_ns=20,
        action_space="joint_position",
        dt_ns=33_333_333,
        groups={
            "left_arm": np.arange(horizon * 7, dtype=np.float64).reshape(horizon, 7),
            "right_arm": np.arange(horizon * 7, dtype=np.float64).reshape(horizon, 7),
        },
    )


def test_autohorizon_truncates_only_after_full_chunk_decode() -> None:
    config = load_config(
        "manimux/configs/experiments/pick_red_object/yam_pi05_autohorizon_step1000.yaml"
    )
    strategy = AutoHorizonInferenceStrategy(config)
    response = InferenceResponse(
        session_id="session",
        request_seq=1,
        finished_time_ns=100,
        inference_ms=50.0,
        raw_action={"actions": [], "autohorizon": {"execution_steps": 7}},
    )

    prepared = strategy.prepare_chunk(chunk=_chunk(), response=response, now_ns=200)

    assert prepared.horizon_steps == 7
    assert prepared.observation_time_ns == 10
    np.testing.assert_array_equal(prepared.groups["left_arm"], _chunk().groups["left_arm"][:7])


def test_autohorizon_rejects_missing_or_invalid_server_horizon() -> None:
    config = load_config(
        "manimux/configs/experiments/pick_red_object/yam_pi05_autohorizon_step1000.yaml"
    )
    strategy = AutoHorizonInferenceStrategy(config)
    for raw_action in ({"actions": []}, {"autohorizon": {"execution_steps": 0}}):
        response = InferenceResponse(
            session_id="session",
            request_seq=1,
            finished_time_ns=100,
            inference_ms=50.0,
            raw_action=raw_action,
        )
        try:
            strategy.prepare_chunk(chunk=_chunk(), response=response, now_ns=200)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid AutoHorizon metadata was accepted")
