from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from manimux.cli import load_config, prepare_experiment
from manimux.runtime.dvac import DvacInferenceStrategy
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


def test_dvac_truncates_only_after_full_chunk_decode() -> None:
    config = load_config("manimux/configs/experiments/pick_red_object/pi05/yam_pi05_dvac_step1000.yaml")
    strategy = DvacInferenceStrategy(config)
    response = InferenceResponse(
        session_id="session",
        request_seq=1,
        finished_time_ns=100,
        inference_ms=50.0,
        raw_action={"actions": [], "dvac": {"execution_steps": 7}},
    )

    prepared = strategy.prepare_chunk(chunk=_chunk(), response=response, now_ns=200)

    assert prepared.horizon_steps == 7
    assert prepared.observation_time_ns == 10
    np.testing.assert_array_equal(prepared.groups["left_arm"], _chunk().groups["left_arm"][:7])


def test_dvac_rejects_invalid_metadata_and_uses_configured_blending() -> None:
    config = load_config("manimux/configs/experiments/pick_red_object/pi05/yam_pi05_dvac_step1000.yaml")
    strategy = DvacInferenceStrategy(config)
    for raw_action in ({"actions": []}, {"dvac": {"execution_steps": 0}}):
        response = InferenceResponse(
            session_id="session",
            request_seq=1,
            finished_time_ns=100,
            inference_ms=50.0,
            raw_action=raw_action,
        )
        with pytest.raises(ValueError):
            strategy.prepare_chunk(chunk=_chunk(), response=response, now_ns=200)

    payload = deepcopy(config)
    payload["inference"].pop("inference_schedule")
    payload["inference"].pop("refill_threshold_s")
    payload["inference"]["blend_policy_steps"] = 2
    resolved = prepare_experiment(**payload)
    configured = DvacInferenceStrategy(resolved).commit_settings(
        response=response,
        measured={"left_arm": np.zeros(7), "right_arm": np.zeros(7)},
        last_command={"left_arm": np.ones(7), "right_arm": np.ones(7)},
    )
    assert configured.blend_steps == 2


def test_dvac_config_preserves_paper_defaults_and_pi05_contract() -> None:
    config = load_config("manimux/configs/experiments/pick_red_object/pi05/yam_pi05_dvac_step1000.yaml")

    assert config["inference"]["algorithm"] == "dvac"
    assert config["inference"]["dvac"]["tail_policy_steps"] == 5
    assert config["inference"]["dvac"]["alpha"] == pytest.approx(2.0)
    assert config["inference"]["dvac"]["rolling_window_size"] == 5
    assert config["inference"]["dvac"]["min_execution_policy_steps"] == 1
    assert config["inference"]["dvac"]["max_execution_policy_steps"] == 50
    assert config["policy"]["horizon_policy_steps"] == 50
    assert config["inference"]["blend_policy_steps"] == 0
