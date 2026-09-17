"""Timing instrumentation must preserve device calls and retain every submission."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from manimux.collection.yam.backend import CollectionBackend, FollowerView, load_backend_config
from manimux.collection.yam.config import CameraConfig, build_station_config
from manimux.collection.yam.data.control_timing import ControlTimings, recording_snapshot
from manimux.collection.yam.data.recorder import EpisodeRecorder
from manimux.collection.yam.robot.yam_adapter import YamLeaderPolicy
from manimux.collection.yam.runtime import ArmUnit
from manimux.collection.yam.teleop.loop import ControlLoop
from manimux.robots.yam.base import BimanualRobot
from manimux.timing import stage
from manimux.types import RobotState


class FakeArm:
    def __init__(self):
        self.q = np.zeros(7)
        self.reads = 0
        self.commands = []
        self.fail_submit = False

    def num_dofs(self):
        return 7

    def get_joint_state(self):
        self.reads += 1
        return self.q.copy()

    def command_joint_state(self, q):
        if self.fail_submit:
            raise OSError("injected right submit failure")
        self.q = q.copy()
        self.commands.append(q.copy())


@contextmanager
def diagnostic_loop(tmp_path):
    cfg = build_station_config("configs/collection/yam/station.yaml")
    cfg.collection_hz = 100
    cfg.cameras = []
    cfg.save_root = str(tmp_path / "episodes")
    backend = CollectionBackend(load_backend_config(cfg, mock=True), mock=True, lock_dir=tmp_path)
    arms = [FakeArm(), FakeArm()]
    robot = BimanualRobot(*arms)

    class Driver:
        def connect(self):
            pass

        def get_state(self):
            q = robot.get_joint_state()
            return RobotState(groups={"left_arm": q[:7], "right_arm": q[7:]},
                              monotonic_ns=backend.clock.now_ns(), sequence=arms[0].reads)

        def send_command(self, command):
            robot.command_joint_state(np.concatenate([command.groups[g] for g in
                                                       ("left_arm", "right_arm")]))

        def stop(self):
            pass

        def close(self):
            pass

    backend.driver = Driver()
    backend.connect(start_thread=False)
    leaders, units = [], []
    for side in ("left", "right"):
        leader = SimpleNamespace(q=np.zeros(6), feedback=[])
        leader.get_state = lambda leader=leader: (leader.q.copy(), 0.4, [False, False])
        leader.command_arm = lambda q, leader=leader: leader.feedback.append(q.copy())
        leaders.append(leader)
        follower = FollowerView(backend, f"{side}_arm")
        units.append(ArmUnit(side, follower,
                             YamLeaderPolicy(leader, follower, control_hz=100, bilateral_kp=0.2)))
    loop = ControlLoop(units, [], 100)
    loop.sync_enabled = True
    loop.timings.configure_output(tmp_path / "diagnostics", {"mock": True})
    try:
        yield cfg, backend, loop, arms, leaders
    finally:
        loop.timings.flush("test_end")
        loop.timings.wait_for_saves()
        backend.close()


def saved_rows(timings):
    timings.wait_for_saves()
    result = timings.output_status()["latest"]
    assert result["status"] == "saved", result
    path = Path(result["path"])
    assert (path / "write_complete.flag").exists()
    return [json.loads(line) for line in (path / "control-timing.jsonl").read_text().splitlines()]


def test_every_command_saved_beyond_ui_window_without_changing_reads_or_targets(tmp_path):
    with diagnostic_loop(tmp_path) as (_, backend, loop, arms, leaders):
        initial_reads = [arm.reads for arm in arms]
        backend.start_trace()
        expected = []
        for i in range(205):
            value = 0.02 if i % 2 else -0.03
            leaders[0].q[:] = value
            leaders[1].q[:] = -value
            expected.append(value)
            loop._step()
        # Three reads per arm per steady cycle; the first activation has its
        # existing additional executor-reset read. Instrumentation adds none.
        assert [arm.reads - start for arm, start in zip(arms, initial_reads, strict=True)] == [
            4 + 204 * 3, 4 + 204 * 3,
        ]
        assert [len(leader.feedback) for leader in leaders] == [205, 205]
        assert [row[0] for row in arms[0].commands] == expected
        assert [row[0] for row in arms[1].commands] == [-v for v in expected]
        trace = backend.finish_trace()
        assert len(trace) == 205
        assert loop.timings.snapshot()["samples"] == 200
        loop.sync_enabled = False
        loop.timings.request_save("pause")
        loop._step()
        rows = saved_rows(loop.timings)
        commands = [r for r in rows if "backend.follower_sdk_submit" in r["stages"]]
        assert len(commands) == 205
        assert [r["stages"]["backend.follower_sdk_submit"]["sequence"] for r in commands] == [
            t["sequence"] for t in trace
        ]
        for row in commands:
            for scope in ("observation_left_arm", "observation_right_arm", "precommand"):
                for side in ("left", "right"):
                    assert f"{scope}.{side}_sdk_state" in row["stages"]
            for side in ("left", "right"):
                span = row["stages"][f"submit.{side}_sdk_submit"]
                assert span["offset_ns"] >= 0
                assert span["duration_ns"] >= span["cpu_ns"] >= 0


def test_failed_right_submit_keeps_left_evidence_and_propagates_error(tmp_path):
    with diagnostic_loop(tmp_path) as (_, backend, loop, arms, _):
        arms[1].fail_submit = True
        with pytest.raises(OSError, match="injected right"):
            loop._step()
        assert backend._halted
        loop.timings.flush("loop_error")
        row = saved_rows(loop.timings)[0]
        assert "injected right" in row["error"]
        assert "error" not in row["stages"]["submit.left_sdk_submit"]
        assert "injected right" in row["stages"]["submit.right_sdk_submit"]["error"]
        assert "error" in row["stages"]["backend.follower_sdk_submit"]
        assert len(arms[0].commands) == 1 and not arms[1].commands


def test_repeated_alignment_spans_keep_all_calls_and_frozen_copy(tmp_path):
    timings = ControlTimings()
    timings.configure_output(tmp_path, {"mock": True})
    with timings.cycle(0.01, kind="alignment"):
        for _ in range(2):
            with stage("submit"):
                pass
        frozen = recording_snapshot()
        with stage("submit"):
            pass
    assert len(frozen["stages"]["submit"]["spans"]) == 2
    timings.flush("pause")
    row = saved_rows(timings)[0]
    assert row["kind"] == "alignment"
    spans = row["stages"]["submit"]["spans"]
    assert len(spans) == 3
    assert row["stages"]["submit"]["duration_ns"] == sum(s["duration_ns"] for s in spans)


def test_sleep_diagnostics_keep_pacing_and_ema_semantics(monkeypatch):
    import manimux.collection.yam.teleop.loop as loop_module

    loop = ControlLoop([], [], 100)
    now = [0.003]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds + 0.002  # deterministic scheduler oversleep

    monkeypatch.setattr(loop_module, "time", SimpleNamespace(
        perf_counter=lambda: now[0], monotonic_ns=lambda: round(now[0] * 1e9), sleep=sleep,
    ))
    loop._step()
    start = loop.timings.snapshot()["latest"]["cycle_monotonic_ns"]
    loop._sleep_to_rate(0)
    assert sleeps == [pytest.approx(0.007)]
    assert loop.actual_hz == pytest.approx(1 / 0.012)
    assert loop.overruns == 0
    loop._step()
    pacing = loop.timings.snapshot()["latest"]["previous_pacing"]
    assert pacing["cycle_monotonic_ns"] == start
    assert pacing["requested_sleep_ns"] == 7_000_000
    assert pacing["actual_sleep_ns"] == 9_000_000
    assert pacing["sleep_overshoot_ns"] == 2_000_000
    now[0] = 0.018
    loop._sleep_to_rate(0)
    assert len(sleeps) == 1
    assert loop.overruns == 1
    assert loop.actual_hz == pytest.approx(0.8 / 0.012 + 0.2 / 0.018)


def test_episode_timing_rows_match_commands_and_work_cpu_is_saved(tmp_path):
    with diagnostic_loop(tmp_path) as (cfg, backend, loop, _, _):
        recorder = EpisodeRecorder(cfg.save_root, cfg, [], ["left", "right"], backend=backend)
        loop.attach_recorder(recorder)
        episode = recorder.start("timing")
        for _ in range(3):
            t0 = time.perf_counter()
            loop._step()
            loop._sleep_to_rate(t0)
        recorder.stop()
        rows = [json.loads(s) for s in (episode / "control-timing.jsonl").read_text().splitlines()]
        assert len(rows) == 3
        assert [r["sample_index"] for r in rows] == [0, 1, 2]
        assert all(r["work_cpu_ns"] >= 0 for r in rows)
        assert rows[1]["previous_pacing"]["cycle_monotonic_ns"] == rows[0]["cycle_monotonic_ns"]
        assert np.load(episode / "controller-left-joint.npy").shape[0] == len(rows)
        meta = json.loads((episode / "metadata.json").read_text())
        assert meta["extra"]["control_timing"]["schema_version"] == 2


def test_gui_pause_saves_diagnostics_without_record_button(tmp_path):
    from fastapi.testclient import TestClient

    from manimux.collection.yam.gui.server import create_app

    cfg = build_station_config("configs/collection/yam/station.yaml")
    cfg.save_root = str(tmp_path)
    cfg.cameras = [CameraConfig("top", "mock", "top", width=32, height=24)]
    app = create_app(cfg, mock=True)
    with TestClient(app) as client:
        assert client.post("/api/collect/start-teleop").status_code == 200
        time.sleep(0.08)
        assert not app.state.session.recorder.is_recording
        assert client.post("/api/collect/stop-teleop").status_code == 200
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = client.get("/api/collect/status").json()
            latest = status["teleop_diagnostics"]["latest"]
            if latest and latest["status"] == "saved":
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"diagnostics did not save: {status}")
        rows = saved_rows(app.state.session.loop.timings)
        assert {r["kind"] for r in rows} >= {"cycle", "alignment", "pause"}
        commands = [r for r in rows if "backend.follower_sdk_submit" in r["stages"]]
        assert len(commands) == app.state.session.units[0].robot.backend._sequence > 0


def test_diagnostic_save_failure_is_reported_without_raising_in_control(tmp_path, monkeypatch):
    timings = ControlTimings()
    timings.configure_output(tmp_path / "diagnostics", {})
    with timings.cycle(0.01, sync_enabled=True), stage("command"):
        pass

    def refuse(*args, **kwargs):
        raise OSError("injected full disk")

    monkeypatch.setattr(Path, "mkdir", refuse)
    timings.flush("pause")
    timings.wait_for_saves()
    status = timings.output_status()["latest"]
    assert status["status"] == "error"
    assert "injected full disk" in status["error"]
