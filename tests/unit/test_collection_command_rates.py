"""Command-rate semantics with deterministic clocks and mock drivers only."""

from types import SimpleNamespace

import numpy as np
import pytest

from manimux.collection.yam.backend import CollectionBackend, load_backend_config
from manimux.collection.yam.config import build_station_config
from manimux.collection.yam.data.command_rates import CommandRates


def test_window_counts_successes_over_time_including_stalls():
    now = [0]
    rates = CommandRates(clock=lambda: now[0])
    assert rates.snapshot()["submitted_hz"] == 0
    for sequence in range(1, 201):
        now[0] = sequence * 10_000_000
        rates.record(sequence)
    s = rates.snapshot()
    assert s["submitted_count"] == s["fresh_target_count"] == 200
    assert s["submitted_hz"] == s["fresh_target_hz"] == 100
    assert s["fresh_target_interval_ms"]["p95"] == 10
    now[0] = 3_000_000_000
    assert rates.snapshot()["fresh_target_hz"] == 50
    now[0] = 4_000_000_000
    s = rates.snapshot()
    assert s["submitted_hz"] == s["fresh_target_hz"] == 0
    assert s["last_submit_age_ms"] == 2000
    assert s["submit_interval_ms"]["p95"] is None


def test_repeats_and_skipped_targets_do_not_inflate_new_target_rate():
    now = [0]
    rates = CommandRates(clock=lambda: now[0])
    for index in range(200):
        now[0] = (index + 1) * 10_000_000
        # 100 Hz executor, 50 Hz delivered new targets; many source sequences
        # were overwritten before the executor could submit them.
        rates.record((index // 2) * 3)
    s = rates.snapshot()
    assert s["submitted_hz"] == 100
    assert s["fresh_target_hz"] == 50
    assert s["repeated_count"] == 100
    assert s["fresh_target_interval_ms"]["p95"] == 20


def test_long_new_target_gap_is_retained_despite_repeated_submissions():
    now = [0]
    rates = CommandRates(clock=lambda: now[0])
    rates.record(1)
    for index in range(1, 300):
        now[0] = index * 10_000_000
        rates.record(1)
    now[0] = 3_000_000_000
    rates.record(2)
    s = rates.snapshot()
    assert s["submitted_hz"] == 100
    assert s["fresh_target_hz"] == 0.5
    assert s["fresh_target_interval_ms"]["max"] == 3000
    assert len(rates._events) == 200


def test_startup_does_not_extrapolate_a_single_event_to_high_hz():
    rates = CommandRates(clock=lambda: 123)
    rates.record(1)
    assert rates.snapshot()["submitted_hz"] == 0.5
    assert rates.snapshot()["submit_interval_ms"]["p95"] is None


@pytest.fixture(params=["synchronous", "threaded"])
def backend(tmp_path, request):
    station = build_station_config("configs/collection/yam/station.yaml")
    instance = CollectionBackend(
        load_backend_config(station, mock=True), mock=True, lock_dir=tmp_path,
        execution_mode=request.param,
    )
    instance.connect(start_thread=False)
    try:
        yield instance
    finally:
        instance.close()


def test_count_only_after_both_arm_submissions_succeed(backend, monkeypatch):
    now = [0]
    backend.command_rates = CommandRates(clock=lambda: now[0])
    original_send = backend.driver.send_command

    def slow_success(command):
        # The in-flight submission must not already appear as successful.
        assert backend.command_rates.snapshot()["submitted_count"] == 0
        original_send(command)
        now[0] = 20_000_000

    monkeypatch.setattr(backend.driver, "send_command", slow_success)
    backend.set_target("left_arm", np.zeros(7))
    if backend.execution_mode == "threaded":
        assert backend.command_rates.snapshot()["submitted_count"] == 0
        backend.tick()
    s = backend.command_rates.snapshot()
    assert s["submitted_count"] == s["fresh_target_count"] == 1
    assert s["last_submit_age_ms"] == 0

    def fail_after_partial_submission(command):
        raise RuntimeError("right SDK submit failed")

    monkeypatch.setattr(backend.driver, "send_command", fail_after_partial_submission)
    with pytest.raises(RuntimeError, match="right SDK"):
        backend.set_target("left_arm", np.zeros(7))
        if backend.execution_mode == "threaded":
            backend.tick()
    assert backend.command_rates.snapshot()["submitted_count"] == 1


def test_threaded_overwrite_repeat_and_unchanged_positions(backend):
    if backend.execution_mode != "threaded":
        pytest.skip("explicit threaded executor scheduling")
    backend.set_target("left_arm", np.zeros(7))
    backend.set_target("left_arm", np.zeros(7))
    assert backend.command_rates.snapshot()["submitted_count"] == 0
    backend.tick()
    backend.tick()
    s = backend.command_rates.snapshot()
    assert s["submitted_count"] == 2
    assert s["fresh_target_count"] == 1  # overwritten target never reached SDK
    backend.set_target("left_arm", np.zeros(7))
    backend.tick()
    s = backend.command_rates.snapshot()
    assert s["submitted_count"] == 3
    assert s["fresh_target_count"] == 2  # same positions, newly sampled target
    backend.pause()
    assert backend.command_rates.snapshot()["submitted_count"] == 3  # hold excluded


def test_session_reports_delivered_targets_instead_of_loop_ema(backend, tmp_path):
    from manimux.collection.yam.gui.session import CollectSession
    from manimux.collection.yam.teleop.loop import ControlLoop

    cfg = build_station_config("configs/collection/yam/station.yaml")
    cfg.save_root = str(tmp_path / "episodes")
    cfg.cameras = []
    session = CollectSession(cfg, mock=True)
    session.units = [SimpleNamespace(name=side, robot=SimpleNamespace(backend=backend),
                                     agent=SimpleNamespace())
                     for side in ("left", "right")]
    session.loop = ControlLoop(session.units, [], 100)
    session.loop.sync_enabled = True
    session.loop.actual_hz = 99.9  # misleading legacy input-loop estimate
    now = [0]
    backend.command_rates = CommandRates(clock=lambda: now[0])
    for index in range(1, 201):
        now[0] = index * 10_000_000
        backend.command_rates.record(1)  # executor repeats one target at 100 Hz
    status = session.status()
    assert status["collection_actual_hz"] == 0.5
    assert status["collection_command_rates"]["submitted_hz"] == 100
    session.loop.sync_enabled = False
    assert session.status()["collection_actual_hz"] == 0
