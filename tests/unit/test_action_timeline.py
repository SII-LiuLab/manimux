from __future__ import annotations

import numpy as np
import pytest

from manimux.runtime.timeline import ActionTimeline, CommitResult
from manimux.types import ActionChunk


def _chunk(
    values: list[float],
    *,
    request_seq: int = 0,
    observation_time_ns: int = 0,
    source_offset_steps: int = 0,
    hold_from_step: int | None = None,
) -> ActionChunk:
    return ActionChunk(
        plan_id=f"plan-{request_seq}",
        request_seq=request_seq,
        observation_time_ns=observation_time_ns,
        created_time_ns=observation_time_ns,
        action_space="joint_position",
        dt_ns=10,
        groups={"arm": np.asarray(values, dtype=np.float64)[:, None]},
        source_offset_steps=source_offset_steps,
        hold_from_step={} if hold_from_step is None else {"arm": hold_from_step},
    )


def _commit(
    timeline: ActionTimeline,
    chunk: ActionChunk,
    *,
    now_ns: int,
    blend_steps: int = 0,
) -> CommitResult:
    return timeline.commit(
        chunk,
        now_ns=now_ns,
        max_plan_age_ns=1_000,
        current_command={"arm": np.array([0.0])},
        blend_steps=blend_steps,
    )


def test_commit_replaces_active_plan_immediately() -> None:
    timeline = ActionTimeline({"arm": 1}, action_start_mode="first_step_when_ready")
    assert _commit(timeline, _chunk([1, 2, 3, 4]), now_ns=0).accepted
    assert _commit(timeline, _chunk([10, 20, 30], request_seq=1), now_ns=5).accepted

    committed = timeline.active_horizon()
    assert committed is not None
    assert committed.start_time_ns == 5
    assert timeline.active_plan_id == "plan-1"
    assert timeline.accepted_request_seq == 1
    assert timeline.cursor(5) == 0
    assert timeline.remaining_ns(5) == 20
    sample = timeline.sample(5)
    assert sample is not None
    np.testing.assert_array_equal(sample["arm"], [10])
    horizon = timeline.reference_horizon(now_ns=5, dt_ns=5, horizon_steps=3)
    assert horizon is not None
    assert horizon.plan_id == "plan-1"
    np.testing.assert_array_equal(horizon.groups["arm"], [[10], [15], [20]])


def test_motion_target_survives_missed_final_tick_but_does_not_extend_next_policy():
    timeline = ActionTimeline(
        {"arm": 1}, action_start_mode="first_step_when_ready", hold_last_step=True
    )
    assert timeline.commit(
        _chunk([0, 1, 2]), now_ns=0, max_plan_age_ns=1000,
        current_command={"arm": np.array([0.0])}, blend_steps=0,
        retain_terminal_target=True,
    ).accepted
    # A delayed control cycle misses both the final source row and its interval.
    reference = timeline.reference_horizon(now_ns=35, dt_ns=5, horizon_steps=3)
    assert reference is not None
    np.testing.assert_array_equal(reference.groups["arm"], [[2], [2], [2]])
    # A subsequent policy owns its normal finite horizon; the motion must not revive.
    assert _commit(timeline, _chunk([2, 3], request_seq=1), now_ns=40).accepted
    np.testing.assert_array_equal(timeline.sample_runtime(40)["arm"], [2])
    assert timeline.reference_horizon(now_ns=61, dt_ns=5, horizon_steps=3) is None


@pytest.mark.parametrize("source_offset_steps", [0, 2])
def test_drop_infer_latency_trims_only_elapsed_source_rows(source_offset_steps: int) -> None:
    timeline = ActionTimeline({"arm": 1}, action_start_mode="drop_infer_latency")
    chunk = _chunk(list(range(source_offset_steps, 7)), source_offset_steps=source_offset_steps)

    result = _commit(timeline, chunk, now_ns=25)

    assert result.accepted
    assert result.trimmed_steps == 3 - source_offset_steps
    assert result.timeline_latency_ns == 25
    committed = timeline.active_horizon()
    assert committed is not None
    assert committed.start_time_ns == 30
    np.testing.assert_array_equal(committed.groups["arm"], [[3], [4], [5], [6]])
    sample = timeline.sample(30)
    assert sample is not None
    np.testing.assert_array_equal(sample["arm"], [3.0])


def test_first_step_when_ready_keeps_first_row_despite_inference_age() -> None:
    timeline = ActionTimeline({"arm": 1}, action_start_mode="first_step_when_ready")

    result = _commit(timeline, _chunk([10, 20, 30]), now_ns=25)

    assert result.accepted
    assert result.trimmed_steps == 0
    assert result.timeline_latency_ns == 25
    committed = timeline.active_horizon()
    assert committed is not None
    assert committed.start_time_ns == 25
    np.testing.assert_array_equal(committed.groups["arm"], [[10], [20], [30]])


def test_source_limit_and_rejected_replacement_preserve_active_plan() -> None:
    timeline = ActionTimeline({"arm": 1}, max_source_steps=5)
    first = _chunk([2, 3, 4, 5, 6], source_offset_steps=2)
    assert _commit(timeline, first, now_ns=25).accepted
    committed = timeline.active_horizon()
    assert committed is not None
    np.testing.assert_array_equal(committed.groups["arm"], [[3], [4]])

    exhausted = _chunk([2, 3, 4, 5, 6], request_seq=1, source_offset_steps=2)
    result = _commit(timeline, exhausted, now_ns=50)

    assert not result.accepted
    assert result.reason == "no_future_horizon"
    assert timeline.active_plan_id == "plan-0"
    assert timeline.accepted_request_seq == 0


def test_serial_last_row_holds_for_one_interval() -> None:
    timeline = ActionTimeline(
        {"arm": 1},
        max_source_steps=2,
        action_start_mode="first_step_when_ready",
        hold_last_step=True,
    )
    assert _commit(timeline, _chunk([1, 2, 3]), now_ns=25).accepted

    final_sample = timeline.sample(44)
    assert final_sample is not None
    np.testing.assert_array_equal(final_sample["arm"], [2])
    assert timeline.remaining_ns(44) == 1
    assert timeline.sample(45) is None
    assert timeline.remaining_ns(45) == 0
    result = _commit(
        timeline,
        _chunk([2, 3], request_seq=1, source_offset_steps=1),
        now_ns=45,
    )
    assert not result.accepted
    assert result.reason == "serial_requires_untrimmed_chunk"


def test_blending_preserves_unblended_tracking_and_invalid_row_hold() -> None:
    timeline = ActionTimeline({"arm": 1}, action_start_mode="first_step_when_ready")
    chunk = _chunk([10, 20, 30], hold_from_step=2)
    assert _commit(timeline, chunk, now_ns=25, blend_steps=2).accepted

    horizon = timeline.reference_horizon(now_ns=25, dt_ns=5, horizon_steps=3)

    assert horizon is not None
    np.testing.assert_array_equal(horizon.groups["arm"], [[5], [12.5], [20]])
    assert horizon.tracking_groups is not None
    np.testing.assert_array_equal(horizon.tracking_groups["arm"], [10])
    assert horizon.hold_groups == ("arm",)
    np.testing.assert_array_equal(chunk.groups["arm"], [[10], [20], [30]])
