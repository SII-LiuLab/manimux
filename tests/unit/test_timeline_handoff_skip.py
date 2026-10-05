"""Intentional source-row skips retain the ordinary Timeline handoff clock."""

import numpy as np
import zarr

from manimux.recording import EpisodeRecorder
from manimux.runtime.timeline import ActionTimeline
from manimux.types import ActionChunk, ActionHorizon, RuntimeTrajectory

DT_NS = 100_000_000
ORIGIN_NS = 1_000_000_000


def _chunk(seq: int, observation_ns: int, *, source_offset: int = 0, runtime=None):
    rows = np.arange(source_offset, 12, dtype=np.float64)[:, None]
    return ActionChunk(
        plan_id=f"plan-{seq}",
        request_seq=seq,
        observation_time_ns=observation_ns,
        created_time_ns=observation_ns,
        action_space="joint_position",
        dt_ns=DT_NS,
        groups={"arm": rows},
        source_offset_steps=source_offset,
        runtime_trajectory=runtime,
    )


def _commit(timeline, chunk, now_ns, *, lead_ns=0, skip=0):
    return timeline.commit(
        chunk,
        now_ns=now_ns,
        commit_lead_ns=lead_ns,
        max_plan_age_ns=2_000_000_000,
        current_command={"arm": np.zeros(1)},
        blend_steps=0,
        handoff_skip_steps=skip,
    )


def _timeline(*, serial=False):
    timeline = ActionTimeline({"arm": 1}, start_on_commit=serial)
    first = _commit(timeline, _chunk(1, ORIGIN_NS), ORIGIN_NS, skip=5)
    assert first.accepted
    assert first.handoff_skipped_steps == 0
    return timeline


def test_handoff_skip_advances_content_without_delaying_handoff():
    baseline = _timeline()
    skipped = _timeline()
    second = _chunk(2, ORIGIN_NS + 300_000_000)
    now_ns = ORIGIN_NS + 550_000_000
    ordinary = _commit(baseline, second, now_ns, lead_ns=50_000_000)
    result = _commit(skipped, second, now_ns, lead_ns=50_000_000, skip=2)

    assert ordinary.accepted and result.accepted
    assert result.time_trimmed_steps == 3
    assert result.handoff_skipped_steps == 2
    assert result.trimmed_steps == 5
    assert result.timeline_latency_ns == ordinary.timeline_latency_ns == 300_000_000
    assert skipped.active_horizon().start_time_ns == baseline.active_horizon().start_time_ns
    assert skipped.active_horizon().start_time_ns == ORIGIN_NS + 600_000_000
    assert baseline.active_horizon().groups["arm"][0, 0] == 3
    assert skipped.active_horizon().groups["arm"][0, 0] == 5
    assert skipped.sample(now_ns)["arm"][0] == baseline.sample(now_ns)["arm"][0]


def test_handoff_skip_counts_after_adapter_source_offset():
    timeline = _timeline()
    observation_ns = ORIGIN_NS + 300_000_000
    result = _commit(
        timeline,
        _chunk(2, observation_ns, source_offset=5),
        observation_ns + 650_000_000,
        skip=2,
    )

    assert result.accepted
    assert (result.time_trimmed_steps, result.handoff_skipped_steps) == (2, 2)
    assert timeline.active_horizon().start_time_ns == observation_ns + 700_000_000
    assert timeline.active_horizon().groups["arm"][0, 0] == 9


def test_serial_handoff_skip_keeps_commit_start():
    timeline = _timeline(serial=True)
    now_ns = ORIGIN_NS + 550_000_000
    result = _commit(
        timeline,
        _chunk(2, ORIGIN_NS + 300_000_000),
        now_ns,
        lead_ns=50_000_000,
        skip=2,
    )

    assert result.accepted
    assert (result.time_trimmed_steps, result.handoff_skipped_steps) == (0, 2)
    assert timeline.active_horizon().start_time_ns == now_ns + 50_000_000
    assert timeline.active_horizon().groups["arm"][0, 0] == 2


def test_dense_runtime_trajectory_skips_matching_substeps_without_time_shift():
    timeline = _timeline()
    observation_ns = ORIGIN_NS + 300_000_000
    dense = RuntimeTrajectory(
        start_time_ns=observation_ns - 75_000_000,
        dt_ns=25_000_000,
        groups={"arm": np.arange(48, dtype=np.float64)[:, None]},
    )
    result = _commit(
        timeline,
        _chunk(2, observation_ns, runtime=dense),
        ORIGIN_NS + 550_000_000,
        lead_ns=50_000_000,
        skip=2,
    )

    assert result.accepted
    start_ns = timeline.active_horizon().start_time_ns
    assert start_ns == ORIGIN_NS + 600_000_000
    assert timeline.active_horizon().groups["arm"][0, 0] == 5
    assert timeline._sample_runtime(start_ns)["arm"][0] == 23


def test_handoff_skip_rejects_chunk_with_no_remaining_rows():
    timeline = _timeline()
    observation_ns = ORIGIN_NS + 300_000_000
    result = _commit(
        timeline,
        _chunk(2, observation_ns),
        observation_ns + 850_000_000,
        lead_ns=50_000_000,
        skip=5,
    )

    assert not result.accepted
    assert result.reason == "no_future_horizon"
    assert timeline.active_plan_id == "plan-1"


def test_recording_preserves_time_trim_and_intentional_skip_separately(tmp_path):
    recorder = EpisodeRecorder(tmp_path, "episode-001", {"arm": 1}, {})
    raw = _chunk(1, ORIGIN_NS)
    output = _chunk(1, ORIGIN_NS)
    output.metadata.update(time_trimmed_steps=3, handoff_skipped_steps=2)
    committed = ActionHorizon(
        start_time_ns=ORIGIN_NS + 3 * DT_NS,
        dt_ns=DT_NS,
        plan_id=output.plan_id,
        groups={"arm": output.groups["arm"][5:]},
        observation_time_ns=ORIGIN_NS,
    )
    recorder.record_plan(canonical_raw=raw, infra_output=output, committed=committed)
    episode = recorder.finish(success=True, terminal_reason="test", steps=0, wall_time_s=0.0)

    attrs = zarr.open_group(str(episode / "data.zarr"), mode="r")
    attrs = attrs["plans"]["000000"]["committed"].attrs
    assert attrs["time_trimmed_steps"] == 3
    assert attrs["handoff_skipped_steps"] == 2
