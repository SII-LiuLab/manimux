from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from manimux.cli import load_config, prepare_experiment
from manimux.policies.base import action_interval
from manimux.policies.fake import FakePolicyAdapter
from manimux.runtime.inference import RequestState, build_inference_strategy
from manimux.runtime.safety import RuntimeState
from manimux.runtime.temporal_ensemble import ACTTemporalEnsembler
from manimux.runtime.timeline import ActionTimeline
from manimux.types import ActionChunk, ObservationSnapshot, RobotState


def _chunk(
    *,
    request_seq: int,
    observation_time_ns: int,
    values: list[float],
    dt_ns: int = 100,
) -> ActionChunk:
    return ActionChunk(
        plan_id=f"plan-{request_seq}",
        request_seq=request_seq,
        observation_time_ns=observation_time_ns,
        created_time_ns=observation_time_ns + 1,
        action_space="joint_position",
        dt_ns=dt_ns,
        groups={"arm": np.asarray(values, dtype=np.float64)[:, None]},
    )


def _act_config(*, query_interval_steps: int = 4) -> dict:
    payload = deepcopy(load_config("configs/mock.yaml"))
    payload["execution"].pop("inference_schedule")
    payload["execution"].pop("refill_threshold_s")
    payload["execution"]["runtime"] = "act_temporal_ensemble"
    payload["execution"]["blend_steps"] = 0
    payload["execution"]["temporal_ensemble"] = {
        "coefficient": 0.01,
        "query_interval_steps": query_interval_steps,
    }
    return prepare_experiment(**payload)


def test_temporal_ensemble_matches_official_act_weighting() -> None:
    ensembler = ACTTemporalEnsembler(coefficient=0.01)
    first = ensembler.aggregate(
        _chunk(request_seq=1, observation_time_ns=1_000, values=[0, 1, 2, 3, 4])
    )
    np.testing.assert_allclose(first.groups["arm"][:, 0], [0, 1, 2, 3, 4])

    second = ensembler.aggregate(
        _chunk(request_seq=2, observation_time_ns=1_200, values=[10, 11, 12, 13, 14])
    )
    weights = np.exp(-0.01 * np.arange(2, dtype=np.float64))
    weights /= weights.sum()
    expected_overlap = np.asarray(
        [
            2 * weights[0] + 10 * weights[1],
            3 * weights[0] + 11 * weights[1],
            4 * weights[0] + 12 * weights[1],
        ]
    )
    np.testing.assert_allclose(second.groups["arm"][:3, 0], expected_overlap)
    np.testing.assert_allclose(second.groups["arm"][3:, 0], [13, 14])
    assert ensembler.last_contributor_counts == (2, 2, 2, 1, 1)


def test_temporal_ensemble_queries_in_policy_steps_not_control_ticks() -> None:
    config = _act_config(query_interval_steps=4)
    strategy = build_inference_strategy(config)
    snapshot = ObservationSnapshot(
        state=RobotState(
            groups={name: np.zeros(dim) for name, dim in config["robot"]["group_dims"].items()},
            monotonic_ns=1_000_000_000,
            sequence=1,
        )
    )
    request_state = RequestState(
        in_flight=False,
        last_submitted_seq=-1,
        last_deadline_ns=0,
    )
    kwargs = {
        "session_id": "session",
        "snapshot": snapshot,
        "adapter": FakePolicyAdapter(),
        "timeline": ActionTimeline(config["robot"]["group_dims"]),
        "request_state": request_state,
        "runtime_state": RuntimeState.RUNNING,
    }

    first = strategy.build_submission(request_seq=1, now_ns=1_000_000_000, **kwargs)
    assert first is not None
    assert first.event_fields["query_interval_ms"] == pytest.approx(200.0)
    assert strategy.build_submission(request_seq=2, now_ns=1_199_999_999, **kwargs) is None
    assert strategy.build_submission(request_seq=2, now_ns=1_200_000_000, **kwargs) is not None


def test_temporal_ensemble_rejects_double_blending_and_nonoverlap() -> None:
    payload = deepcopy(_act_config())
    payload["execution"].pop("inference_schedule")
    payload["execution"].pop("refill_threshold_s")
    payload["execution"]["blend_steps"] = 2
    with pytest.raises(ValueError, match="blend_steps=0"):
        prepare_experiment(**payload)

    payload = deepcopy(_act_config())
    payload["execution"].pop("inference_schedule")
    payload["execution"].pop("refill_threshold_s")
    payload["execution"]["temporal_ensemble"]["query_interval_steps"] = payload["policy"][
        "horizon_steps"
    ]
    with pytest.raises(ValueError, match="consecutive chunks overlap"):
        prepare_experiment(**payload)


def test_pi05_temporal_ensemble_config_loads_with_four_step_queries() -> None:
    config = load_config("configs/pi05/yam/infra/act-temporal-ensemble.yaml")

    assert config["execution"]["runtime"] == "act_temporal_ensemble"
    assert config["execution"]["temporal_ensemble"]["coefficient"] == pytest.approx(0.01)
    assert config["execution"]["temporal_ensemble"]["query_interval_steps"] == 4
    assert config["execution"]["blend_steps"] == 0
    assert action_interval(config["policy"]) * 4 == pytest.approx(0.13333333333333333)


def test_pi05_step1000_temporal_ensemble_preserves_checkpoint_contract() -> None:
    config = load_config(
        "configs/pi05/yam/infra/pick-red-ball-box/act-temporal-ensemble-step1000.yaml"
    )

    assert config["execution"]["runtime"] == "act_temporal_ensemble"
    assert config["execution"]["temporal_ensemble"]["query_interval_steps"] == 4
    assert config["policy"]["horizon_steps"] == 50
    assert config["robot"]["control_hz"] == pytest.approx(100.0)
    assert config["run"]["task"] == "Pick the red ball up and place it into the box."
