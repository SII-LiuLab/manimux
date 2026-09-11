from __future__ import annotations

import numpy as np

from manimux.runtime.timeline import ActionTimeline
from manimux.types import ActionChunk


def _chunk(request_seq: int, observation_time_ns: int = 0) -> ActionChunk:
    values = np.arange(10, dtype=np.float64).reshape(5, 2)
    return ActionChunk(
        plan_id=f"plan-{request_seq}",
        request_seq=request_seq,
        observation_time_ns=observation_time_ns,
        created_time_ns=observation_time_ns,
        action_space="joint_position",
        dt_ns=10,
        groups={"left_arm": values, "right_arm": -values},
    )


def test_commit_is_atomic_and_rejects_stale_sequence() -> None:
    timeline = ActionTimeline({"left_arm": 2, "right_arm": 2})
    current = {"left_arm": np.zeros(2), "right_arm": np.zeros(2)}
    accepted = timeline.commit(
        _chunk(1),
        now_ns=0,
        commit_lead_ns=0,
        max_plan_age_ns=100,
        current_command=current,
        blend_steps=0,
    )
    stale = timeline.commit(
        _chunk(1),
        now_ns=1,
        commit_lead_ns=0,
        max_plan_age_ns=100,
        current_command=current,
        blend_steps=0,
    )
    assert accepted.accepted
    assert not stale.accepted
    assert stale.reason == "stale_request_seq"
    assert timeline.active_plan_id == "plan-1"


def test_commit_trims_obsolete_prefix_and_interpolates() -> None:
    timeline = ActionTimeline({"left_arm": 2, "right_arm": 2})
    current = {"left_arm": np.zeros(2), "right_arm": np.zeros(2)}
    result = timeline.commit(
        _chunk(1),
        now_ns=20,
        commit_lead_ns=0,
        max_plan_age_ns=100,
        current_command=current,
        blend_steps=0,
    )
    assert result.accepted
    assert result.trimmed_steps == 2
    sample = timeline.sample(25)
    assert sample is not None
    np.testing.assert_allclose(sample["left_arm"], [5.0, 6.0])
    np.testing.assert_allclose(sample["right_arm"], [-5.0, -6.0])

    committed = timeline.active_horizon()
    assert committed is not None
    assert committed.start_time_ns == 20
    assert committed.groups["left_arm"].shape == (3, 2)
    np.testing.assert_array_equal(committed.groups["left_arm"][0], [4.0, 5.0])
    assert timeline.cursor(20) == 0
    assert timeline.cursor(31) == 1
    assert timeline.cursor(100) == 3


def test_commit_does_not_trim_adapter_source_offset_twice() -> None:
    timeline = ActionTimeline({"left_arm": 2, "right_arm": 2})
    current = {"left_arm": np.zeros(2), "right_arm": np.zeros(2)}
    chunk = _chunk(1)
    chunk.groups = {name: values[2:].copy() for name, values in chunk.groups.items()}
    chunk.source_offset_steps = 2

    result = timeline.commit(
        chunk,
        now_ns=20,
        commit_lead_ns=0,
        max_plan_age_ns=100,
        current_command=current,
        blend_steps=0,
    )

    assert result.accepted
    assert result.trimmed_steps == 0
    committed = timeline.active_horizon()
    assert committed is not None
    np.testing.assert_array_equal(committed.groups["left_arm"][0], [4.0, 5.0])


def test_source_offset_does_not_hide_original_plan_age() -> None:
    timeline = ActionTimeline({"left_arm": 2, "right_arm": 2})
    current = {"left_arm": np.zeros(2), "right_arm": np.zeros(2)}
    chunk = _chunk(1)
    chunk.source_offset_steps = 2

    result = timeline.commit(
        chunk,
        now_ns=25,
        commit_lead_ns=0,
        max_plan_age_ns=20,
        current_command=current,
        blend_steps=0,
    )

    assert not result.accepted
    assert result.reason == "plan_too_old"


def test_dimension_mismatch_rejects_entire_chunk() -> None:
    timeline = ActionTimeline({"left_arm": 2, "right_arm": 2})
    current = {"left_arm": np.zeros(2), "right_arm": np.zeros(2)}
    chunk = _chunk(2)
    chunk.groups["right_arm"] = np.zeros((5, 3))
    result = timeline.commit(
        chunk,
        now_ns=0,
        commit_lead_ns=0,
        max_plan_age_ns=100,
        current_command=current,
        blend_steps=0,
    )
    assert not result.accepted
    assert result.reason == "dimension_mismatch:right_arm"
    assert timeline.active_plan_id is None


def test_execution_prefix_caps_original_rows_before_latency_trimming():
    # Policy still returns 50; never accidentally execute 25 AFTER the trim.
    timeline = ActionTimeline({'arm': 1}, max_source_steps=25)
    original = np.arange(50, dtype=float).reshape(50, 1)
    chunk = ActionChunk('plan', 1, 0, 0, 'joint_position', 10, {'arm': original})
    result = timeline.commit(chunk, now_ns=30, commit_lead_ns=0, max_plan_age_ns=1000,
                             current_command={'arm': np.zeros(1)}, blend_steps=0)
    assert result.accepted and result.trimmed_steps == 3
    np.testing.assert_array_equal(timeline.active_horizon().groups['arm'][:, 0], np.arange(3, 25))
    assert chunk.horizon_steps == 50
    assert timeline.sample(240)['arm'][0] == 24
    assert timeline.sample(250) is None
    late = ActionChunk('late', 2, 0, 0, 'joint_position', 10, {'arm': original})
    rejected = timeline.commit(late, now_ns=250, commit_lead_ns=0, max_plan_age_ns=1000,
                               current_command={'arm': np.zeros(1)}, blend_steps=0)
    assert not rejected.accepted and rejected.reason == 'no_future_horizon'
    assert timeline.active_plan_id == 'plan'


def test_execution_prefix_respects_already_removed_source_offset():
    timeline = ActionTimeline({'arm': 1}, max_source_steps=25)
    chunk = ActionChunk('plan', 1, 0, 0, 'joint_position', 10,
                        {'arm': np.arange(5, 50, dtype=float).reshape(-1, 1)},
                        source_offset_steps=5)
    result = timeline.commit(chunk, now_ns=70, commit_lead_ns=0, max_plan_age_ns=1000,
                             current_command={'arm': np.zeros(1)}, blend_steps=0)
    assert result.accepted and result.trimmed_steps == 2
    np.testing.assert_array_equal(timeline.active_horizon().groups['arm'][:, 0], np.arange(7, 25))


def test_group_hold_survives_source_and_commit_trim_and_clears_on_new_plan():
    timeline = ActionTimeline({'left_arm': 2, 'right_arm': 2}, max_source_steps=25)
    current = {'left_arm': np.zeros(2), 'right_arm': np.zeros(2)}
    chunk = ActionChunk('first', 1, 0, 0, 'joint_position', 100,
        {name: np.zeros((22, 2)) for name in current}, source_offset_steps=3,
        hold_from_step={'right_arm': 5})
    assert timeline.commit(chunk, now_ns=500, commit_lead_ns=0, max_plan_age_ns=10000,
                           current_command=current, blend_steps=4).accepted
    # Source 8 fails. Hold before the 7->8 interpolation and 10ns feedforward enter it.
    assert timeline.reference_horizon(now_ns=689, dt_ns=10, horizon_steps=2).hold_groups == ()
    assert timeline.reference_horizon(now_ns=690, dt_ns=10, horizon_steps=2).hold_groups == ('right_arm',)
    chunk.request_seq = 2
    chunk.hold_from_step = {}
    assert timeline.commit(chunk, now_ns=800, commit_lead_ns=0, max_plan_age_ns=10000,
                           current_command=current, blend_steps=4).accepted
    assert timeline.reference_horizon(now_ns=800, dt_ns=10, horizon_steps=2).hold_groups == ()


def test_tracking_reference_is_unblended_and_respects_source_trim():
    timeline = ActionTimeline({'arm':2}, max_source_steps=25)
    values = np.column_stack([np.arange(22)*.01, np.ones(22)])
    chunk = ActionChunk('p',1,0,0,'joint_position',100,{'arm':values},source_offset_steps=3)
    assert timeline.commit(chunk, now_ns=500, commit_lead_ns=0, max_plan_age_ns=1000,
                           current_command={'arm':np.zeros(2)}, blend_steps=4).accepted
    ref = timeline.reference_horizon(now_ns=550,dt_ns=10,horizon_steps=2)
    np.testing.assert_allclose(ref.tracking_groups['arm'], [.025,1])
    assert ref.groups['arm'][0,0] < ref.tracking_groups['arm'][0]


def test_reference_preserves_observation_time_for_release_completion_barrier():
    timeline = ActionTimeline({'arm':2})
    chunk = ActionChunk('p',1,100,110,'joint_position',100,{'arm':np.zeros((25,2))})
    assert timeline.commit(chunk,now_ns=220,commit_lead_ns=0,max_plan_age_ns=1000,
                           current_command={'arm':np.zeros(2)},blend_steps=4).accepted
    ref = timeline.reference_horizon(now_ns=230,dt_ns=10,horizon_steps=2)
    assert ref.observation_time_ns == 100
    assert timeline.active_horizon().observation_time_ns == 100
