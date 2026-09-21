"""RTC source clocks and Tianji history after asynchronous embodiment decoding."""

from types import SimpleNamespace

import numpy as np
import pytest

from manimux.cli import load_config
from manimux.integrations.umi_dp_tianji.history import HistoryStrategy, align_rtc_condition
from manimux.runtime.inference import RequestState
from manimux.runtime.safety import RuntimeState
from manimux.runtime.timeline import ActionTimeline
from manimux.types import ActionChunk, InferenceResponse, ObservationSnapshot, RobotState


def setup_plan(commit_lead_s=0.0):
    config = load_config("configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml")
    config["policy"]["options"]["history_strategy"] = "rtc"
    config["policy"]["action_dt_s"] = 0.1
    config["execution"]["commit_lead_s"] = commit_lead_s
    strategy = HistoryStrategy(config).delegate
    dt = 100_000_000
    origin = 1_000_000_000
    groups = {
        name: np.tile(np.arange(5, 16)[:, None], (1, 8)).astype(float)
        for name in config["robot"]["group_dims"]
    }
    chunk = ActionChunk(
        "suffix",
        1,
        origin,
        origin + 550_000_000,
        "joint_position",
        dt,
        groups,
        source_offset_steps=5,
    )
    timeline = ActionTimeline(config["robot"]["group_dims"])
    now = origin + 650_000_000
    result = timeline.commit(
        chunk,
        now_ns=now,
        commit_lead_ns=0,
        max_plan_age_ns=2_000_000_000,
        current_command={name: np.zeros(8) for name in groups},
        blend_steps=0,
    )
    assert result.accepted and result.trimmed_steps == 2
    response = InferenceResponse(
        "test", 1, chunk.created_time_ns, 20.0, None, observation_time_ns=origin
    )
    return config, strategy, timeline, chunk, response, result, now


def submit(strategy, config, timeline, now, seq=2):
    state = RobotState({name: np.zeros(8) for name in config["robot"]["group_dims"]}, now, seq)
    return strategy.build_submission(
        session_id="test",
        request_seq=seq,
        now_ns=now,
        snapshot=ObservationSnapshot(state),
        adapter=SimpleNamespace(build_observation=lambda snapshot: snapshot),
        timeline=timeline,
        request_state=RequestState(False, seq - 1, 0),
        runtime_state=RuntimeState.RUNNING,
    )


def test_rtc_keeps_source_horizon_across_both_trims_and_conditions_committed_clock():
    config, strategy, timeline, chunk, response, result, now = setup_plan()
    strategy.prepare_chunk(chunk=chunk, response=response, now_ns=now)
    event = strategy.on_plan_accepted(chunk=chunk, result=result, response=response, now_ns=now)
    assert event["rtc_source_horizon"] == 16
    assert event["rtc_executed_steps_at_commit"] == 7
    assert submit(strategy, config, timeline, now) is None
    now += chunk.dt_ns
    submission = submit(strategy, config, timeline, now)
    assert submission.event_fields["executed_steps"] == 8
    request = submission.request
    assert request.action_condition.shape == (16, 16)
    np.testing.assert_array_equal(request.action_condition[0], 8)
    align_rtc_condition(
        request,
        timeline,
        now,
        offset_ns=33_333_333,
        dt_ns=chunk.dt_ns,
        group_order=tuple(config["robot"]["group_dims"]),
        horizon=16,
    )
    np.testing.assert_allclose(request.action_condition[0], 8 + 33_333_333 / chunk.dt_ns)
    np.testing.assert_array_equal(request.condition_weights[8:], 0)


def test_rtc_delay_includes_observation_age_decode_and_commit_lead_rounding_up():
    config, strategy, timeline, chunk, response, result, now = setup_plan(commit_lead_s=0.02)
    strategy._request_started_ns[1] = now - 120_000_000
    strategy._request_observation_ns[1] = now - 181_000_000
    strategy._request_forecast[1] = 2
    response.inference_ms = 20.0
    event = strategy.on_plan_accepted(chunk=chunk, result=result, response=response, now_ns=now)
    assert event["request_to_commit_ms"] == 120
    assert event["rtc_delay_ms"] == 201
    assert event["measured_delay"] == 3


def test_reset_allows_fresh_unconditioned_request_after_plan_discard():
    config, strategy, timeline, chunk, response, result, now = setup_plan()
    strategy.on_plan_accepted(chunk=chunk, result=result, response=response, now_ns=now)
    assert submit(strategy, config, timeline, now + 2 * chunk.dt_ns) is not None
    strategy.reset()
    strategy.on_response_rejected(response)  # A pre-pause result can arrive after reset.
    submission = submit(
        strategy, config, ActionTimeline(config["robot"]["group_dims"]), now + 10**9, 3
    )
    assert submission is not None and submission.request.action_condition is None
    assert not submission.event_fields["conditioned"]
    assert (
        strategy.commit_settings(
            response=SimpleNamespace(request_seq=3), measured={}, last_command={}
        ).blend_steps
        == 2
    )


def test_rtc_rejects_missing_source_tail():
    _, strategy, _, chunk, response, _, now = setup_plan()
    chunk.source_offset_steps -= 1
    with pytest.raises(ValueError, match="source horizon"):
        strategy.prepare_chunk(chunk=chunk, response=response, now_ns=now)


def test_rtc_rejects_independent_arm_holds():
    _, strategy, _, chunk, response, _, now = setup_plan()
    chunk.hold_from_step = {"right_arm": 0}
    with pytest.raises(ValueError, match="complete joint plan"):
        strategy.prepare_chunk(chunk=chunk, response=response, now_ns=now)


def test_empty_aligned_overlap_restores_unconditioned_commit(monkeypatch):
    config, strategy, timeline, chunk, response, result, now = setup_plan()
    strategy.on_plan_accepted(chunk=chunk, result=result, response=response, now_ns=now)
    config["policy"]["options"]["first_action_offset_s"] = 1.0
    wrapper = HistoryStrategy(config)
    wrapper.delegate = strategy
    now += 2 * chunk.dt_ns
    state = RobotState({name: np.zeros(8) for name in config["robot"]["group_dims"]}, now, 2)
    snapshot = ObservationSnapshot(state)
    monkeypatch.setattr(wrapper.history, "observe", lambda snapshot: None)
    monkeypatch.setattr(wrapper.history, "window", lambda now: snapshot)
    submission = wrapper.build_submission(
        session_id="test",
        request_seq=2,
        now_ns=now,
        snapshot=snapshot,
        adapter=SimpleNamespace(build_observation=lambda snapshot: snapshot),
        timeline=timeline,
        request_state=RequestState(False, 1, 0),
        runtime_state=RuntimeState.RUNNING,
    )
    assert submission.request.action_condition is None
    assert submission.event_fields["condition_reason"] == "no_committed_overlap"
    assert not submission.event_fields["conditioned"]
    assert (
        strategy.commit_settings(
            response=SimpleNamespace(request_seq=2), measured={}, last_command={}
        ).blend_steps
        == 2
    )
