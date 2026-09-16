"""Shared feedback lifetime and command-only recording; fake drivers only."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from manimux.collection.yam.backend import CollectionBackend, FollowerView, load_backend_config
from manimux.collection.yam.config import StationConfig, build_station_config
from manimux.collection.yam.data.recorder import EpisodeRecorder
from manimux.collection.yam.robot.yam_adapter import YamLeaderPolicy
from manimux.collection.yam.runtime import ArmUnit
from manimux.collection.yam.teleop.loop import ControlLoop
from manimux.types import RobotState


@pytest.fixture
def rig(tmp_path, monkeypatch):
    cfg = build_station_config("configs/collection/yam/station.yaml")
    cfg.cameras = []
    cfg.save_root = str(tmp_path / "episodes")
    backend = CollectionBackend(load_backend_config(cfg, mock=True), mock=True, lock_dir=tmp_path)
    backend.connect(start_thread=False)
    reads, bilateral, units = [], {"left": [], "right": []}, []

    def read():
        index = len(reads) + 1
        state = RobotState(
            groups={side + "_arm": np.r_[np.full(6, index * 0.001), 0.4]
                    for side in bilateral},
            monotonic_ns=backend.clock.now_ns(), sequence=index,
        )
        reads.append(state)
        return state

    monkeypatch.setattr(backend.driver, "get_state", read)
    for side in bilateral:
        leader = SimpleNamespace(
            get_state=lambda: (np.full(6, 0.02), 0.4, [False, False]),
            command_arm=lambda q, side=side: bilateral[side].append(q.copy()),
        )
        follower = FollowerView(backend, side + "_arm")
        agent = YamLeaderPolicy(leader, follower, control_hz=100, bilateral_kp=0.2)
        units.append(ArmUnit(side, follower, agent))
    loop = ControlLoop(units, [], 100)
    loop.sync_enabled = True
    try:
        yield SimpleNamespace(cfg=cfg, backend=backend, loop=loop, reads=reads, bilateral=bilateral)
    finally:
        backend.close()


def test_one_read_each_cycle_shared_by_feedback_executor_and_recording(rig, monkeypatch):
    backend, loop = rig.backend, rig.loop
    observed = []
    execute = backend.executor.step

    def step(now, state, reference):
        observed.append(state)
        return execute(now, state, reference)

    monkeypatch.setattr(backend.executor, "step", step)
    backend.start_trace()
    for index in range(3):
        loop._step()
        assert len(rig.reads) == index + 1  # includes first activation/reset
        state = rig.reads[index]
        assert observed[-1] is state
        assert loop._last_obs["left"]["feedback_timestamp_ns"] == (
            loop._last_obs["right"]["feedback_timestamp_ns"]
        )
        for side in rig.bilateral:
            np.testing.assert_array_equal(rig.bilateral[side][-1], state.groups[side + "_arm"][:6])
        timing = loop.timings.snapshot()["latest"]
        assert "observation_left_arm.state_read" in timing["stages"]
        assert "observation_right_arm.state_reuse" in timing["stages"]
        assert "precommand.state_reuse" in timing["stages"]
        submit = timing["stages"]["backend.follower_sdk_submit"]
        assert submit["feedback_monotonic_ns"] == state.monotonic_ns
        assert submit["feedback_sequence"] == state.sequence
        assert submit["feedback_age_ns"] >= 0
    trace = backend.finish_trace()
    assert len(trace) == 3
    assert [x["feedback"]["left_arm"][0] for x in trace] == [0.001, 0.002, 0.003]
    # Outside a teleop step, independent callers must still get a new read.
    backend.observation("left_arm")
    backend.observation("right_arm")
    assert len(rig.reads) == 5


def test_exception_clears_snapshot_and_state_validation_still_runs(rig, monkeypatch):
    original = rig.backend.safety.validate_state

    def fail(state):
        raise ValueError("invalid measured state")

    monkeypatch.setattr(rig.backend.safety, "validate_state", fail)
    with pytest.raises(ValueError, match="invalid measured state"):
        rig.loop._step()
    assert rig.backend._last_command is None
    assert rig.backend._feedback_cycle.get() is None
    monkeypatch.setattr(rig.backend.safety, "validate_state", original)
    rig.loop._step()
    assert len(rig.reads) == 2


def test_pause_callback_skips_old_observations_and_next_step_refreshes(rig, monkeypatch):
    rig.loop._step()
    sequence = rig.backend._sequence
    callbacks = []

    def callback(_buttons):
        if not callbacks:
            callbacks.append(True)
            rig.backend.pause()

    monkeypatch.setattr(rig.loop, "_handle_button_edges", callback)
    rig.loop._step()
    assert rig.backend._sequence == sequence
    assert len(rig.reads) == 3  # previous step, interrupted step, independent hold
    rig.loop._step()
    assert len(rig.reads) == 4
    assert rig.backend._sequence == sequence + 1


def test_alignment_reads_fresh_and_invalidates_outer_step(rig, monkeypatch):
    for unit in rig.loop.units:
        def engage(_abort, unit=unit):
            for _ in range(2):
                unit.robot.get_observations()
        monkeypatch.setattr(unit.agent, "engage", engage)
    backend = rig.backend
    with backend.feedback_cycle():
        backend.observation("left_arm")
        rig.loop.engage_all()
        assert not backend.feedback_cycle_valid()
        backend.observation("right_arm")
    assert len(rig.reads) == 6  # 1 + 4 independent alignment reads + refreshed outer read


def test_background_caller_cannot_borrow_loop_snapshot(rig):
    from concurrent.futures import ThreadPoolExecutor

    backend = rig.backend
    with backend.feedback_cycle():
        first = backend.observation("left_arm")
        with ThreadPoolExecutor(max_workers=1) as pool:
            other = pool.submit(backend.observation, "left_arm").result(timeout=2)
        same = backend.observation("left_arm")
    assert len(rig.reads) == 2
    np.testing.assert_array_equal(first["joint_pos"], same["joint_pos"])
    assert other["joint_pos"][0] != first["joint_pos"][0]
    assert same["feedback_timestamp_ns"] == first["feedback_timestamp_ns"]


@pytest.mark.parametrize("record_achieved", [False, True])
def test_recording_flag_controls_arrays_trace_and_eepose(rig, record_achieved):
    rig.cfg.record_achieved = record_achieved
    rec = EpisodeRecorder(rig.cfg.save_root, rig.cfg, [], ["left", "right"], backend=rig.backend)
    rig.loop.attach_recorder(rec)
    rec.start("shared_feedback", include_eepose=True)
    for _ in range(3):
        rig.loop._step()
    path = rec.stop()
    assert (path / "write_complete.flag").exists()
    assert rec.last_stop_warning is None
    meta = json.loads((path / "metadata.json").read_text())
    assert meta["extra"]["record_achieved"] is record_achieved
    assert meta["extra"]["control_timing"]["samples"] == 3
    trace = [json.loads(line) for line in (path / "manimux-control.jsonl").read_text().splitlines()]
    assert len(trace) == 3
    assert all(("feedback" in row) is record_achieved for row in trace)
    for side in ("left", "right"):
        for suffix in (
            "joint_pos", "gripper_pos", "feedback-timestamp-ns", "ee_pos", "ee_transform",
        ):
            assert (path / f"{side}-{suffix}.npy").exists() is record_achieved
        assert np.load(path / f"action-{side}-joint.npy").shape == (3, 6)
        assert np.load(path / f"controller-{side}-joint.npy").shape == (3, 6)
        assert np.load(path / f"action-{side}-ee_pos.npy").shape == (3, 3)


@pytest.mark.parametrize("kwargs", [{"writer_name": "abc"}, {"writer_name": "lerobot"},
                                    {"record_native_joints": True}])
def test_command_only_recording_rejects_formats_requiring_feedback(tmp_path, kwargs):
    cfg = StationConfig(record_achieved=False)
    rec = EpisodeRecorder(tmp_path, cfg, [], ["left"])
    with pytest.raises(ValueError, match="record_achieved"):
        rec.start("invalid", **kwargs)
    assert not rec.is_recording
    assert not list(tmp_path.iterdir())
