"""SDK probe tests use fake owners/locks and uninitialized SDK objects, never CAN."""

import ast
import copy
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest

from manimux.robots.yam.sdk_lock_probe import (
    FIELDS,
    HELPER,
    LockProbe,
    getter_breakdown,
    install,
    instrument_function,
    prepare,
    strip_instrumentation,
)
from manimux.timing import active_timing


def events(probe):
    return [dict(zip(FIELDS, e, strict=True)) for b in probe.buffers for e in b.events]


class FakeRobot:
    def get_joint_pos(self):
        """Preserve both the object identity and exception handling."""
        with self._state_lock:
            if self.fail:
                raise ValueError("original error")
            return self._joint_state.pos


def test_ast_preserves_control_and_return_value(tmp_path):
    probe = LockProbe(tmp_path)
    tree, count, _ = instrument_function(FakeRobot.get_joint_pos, "MotorChainRobot")
    assert count == 1
    stripped = strip_instrumentation(copy.deepcopy(tree))
    assert ast.unparse(stripped).startswith("def get_joint_pos(self):")
    namespace = {HELPER: probe}
    exec(compile(tree, "test", "exec"), namespace)
    robot = SimpleNamespace(_state_lock=threading.Lock(), fail=False,
                            _joint_state=SimpleNamespace(pos=object()),
                            motor_chain=SimpleNamespace(channel="can_left"))
    original_lock = robot._state_lock
    probe.arm()
    assert namespace["get_joint_pos"](robot) is robot._joint_state.pos
    robot.fail = True
    with pytest.raises(ValueError, match="original error"):
        namespace["get_joint_pos"](robot)
    assert robot._state_lock is original_lock
    assert original_lock.acquire(blocking=False)
    original_lock.release()
    assert len(events(probe)) == 4
    assert events(probe)[-1]["error"] == "ValueError"


def test_only_normal_active_cycle_triggers_and_carries_cycle_id(tmp_path):
    probe = LockProbe(tmp_path)
    owner = SimpleNamespace(channel="can_lead_l")
    for kind, sync in [("alignment", True), ("cycle", False), ("pause", False)]:
        token = active_timing.set({"kind": kind, "sync_enabled": sync, "cycle_monotonic_ns": 123})
        try:
            with probe.call(owner, "MotorChainRobot.get_joint_pos"):
                pass
        finally:
            active_timing.reset(token)
        assert probe.started_ns is None
    token = active_timing.set({"kind": "cycle", "sync_enabled": True, "cycle_monotonic_ns": 456})
    try:
        with probe.call(owner, "MotorChainRobot.get_joint_pos"):
            pass
    finally:
        active_timing.reset(token)
    assert probe.started_ns is not None
    assert events(probe)[0]["cycle_ns"] == 456
    assert events(probe)[0]["channel"] == "can_lead_l"


def test_contention_is_wait_not_hold_and_objects_are_separate(tmp_path):
    probe = LockProbe(tmp_path)
    probe.arm()
    lock = threading.Lock()
    waiting = threading.Event()
    owner = SimpleNamespace(channel="can_right")
    lock.acquire()

    class SignalledLock:
        def __enter__(self):
            waiting.set()
            return lock.__enter__()

        def __exit__(self, *exc):
            return lock.__exit__(*exc)

    original = SignalledLock()

    def reader():
        with probe.lock(owner, original, "getter.state_lock"):
            pass

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        assert waiting.wait(2)
        time.sleep(0.025)
    finally:
        lock.release()
        thread.join(timeout=2)
    assert not thread.is_alive()
    row = events(probe)[0]
    assert row["acquired_ns"] - row["start_ns"] >= 20_000_000
    assert row["lock_id"] == id(original)
    assert row["hold_cpu_ns"] < row["acquired_ns"] - row["start_ns"]
    second = threading.Lock()
    with probe.lock(SimpleNamespace(channel="can_left"), second, "getter.state_lock"):
        pass
    assert events(probe)[-1]["lock_id"] != row["lock_id"]


def test_nested_wait_links_to_original_owner_hold(tmp_path):
    probe = LockProbe(tmp_path)
    probe.arm()
    owner = SimpleNamespace(channel="can_left")
    state, command = threading.Lock(), threading.Lock()
    with (probe.lock(owner, state, "update.state_lock"),
          probe.lock(owner, command, "set_commands.command_lock")):
        pass
    inner, outer = events(probe)
    assert inner["parent_id"] == outer["span_id"]
    assert outer["acquired_ns"] <= inner["start_ns"] < inner["released_ns"] <= outer["body_end_ns"]
    assert not probe.buffers[0].stack


def test_bounded_buffer_reports_loss_and_saves_after_capture(tmp_path):
    probe = LockProbe(tmp_path, capacity=2)
    probe.arm()
    for _ in range(4):
        with probe.call(SimpleNamespace(channel="can_left"), "fake"):
            pass
    assert not list(tmp_path.iterdir())
    probe.capturing = False
    probe.save()
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["rows"] == 2
    assert report["threads"][0]["dropped"] == 2
    assert (tmp_path / "write_complete.flag").exists()
    assert len((tmp_path / "sdk-locks.jsonl").read_text().splitlines()) == 2


def test_original_context_suppression_and_enter_errors(tmp_path):
    probe = LockProbe(tmp_path)
    probe.arm()

    class Context:
        entered = exited = 0

        def __enter__(self):
            self.entered += 1
            return "original enter value"

        def __exit__(self, *args):
            self.exited += 1
            return True

    lock = Context()
    with probe.lock(SimpleNamespace(channel="test"), lock, "test") as value:
        assert value == "original enter value"
        raise ValueError("suppressed by original context")
    assert (lock.entered, lock.exited) == (1, 1)

    class Broken(Context):
        def __enter__(self):
            raise RuntimeError("acquisition failed")

    with (pytest.raises(RuntimeError, match="acquisition failed"),
          probe.lock(SimpleNamespace(channel="test"), Broken(), "test")):
        pytest.fail("body must not run")
    assert not probe.buffers[0].stack


def test_installed_sdk_is_audited_without_constructing_hardware(tmp_path):
    probe = LockProbe(tmp_path / "audit")
    plan = prepare(probe)
    manifest = json.loads((probe.output / "manifest.json").read_text())
    assert manifest["control_statements_unchanged_ast_check"]
    assert {s["module"] for s in manifest["sources"]} == {
        "i2rt.robots.motor_chain_robot", "i2rt.motor_drivers.dm_driver",
    }
    install(probe, plan)
    try:
        from i2rt.robots.motor_chain_robot import MotorChainRobot
        # __new__ deliberately bypasses SDK initialization and all hardware.
        robot = MotorChainRobot.__new__(MotorChainRobot)
        robot._state_lock = threading.Lock()
        robot._joint_state = SimpleNamespace(pos=object())
        robot.motor_chain = SimpleNamespace(channel="can_left")
        probe.arm()
        assert robot.get_joint_pos() is robot._joint_state.pos
        assert {r["site"] for r in events(probe)} == {
            "MotorChainRobot.get_joint_pos", "MotorChainRobot.get_joint_pos._state_lock",
        }
    finally:
        for module, cls, name, original, _ in plan:
            setattr(cls, name, original)
            module.__dict__.pop(HELPER, None)
    for source in manifest["sources"]:
        from pathlib import Path
        assert hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == source["sha256"]


def test_finite_writer_closes_without_any_teleop(tmp_path):
    probe = LockProbe(tmp_path)
    probe.start_writer()
    probe.close()
    assert not probe.worker.is_alive()
    assert probe.error is None
    assert json.loads((tmp_path / "summary.json").read_text())["rows"] == 0


def test_finished_capture_does_not_rearm(tmp_path):
    probe = LockProbe(tmp_path)
    probe.arm()
    probe.capturing = False
    first_start = probe.started_ns
    token = active_timing.set({"kind": "cycle", "sync_enabled": True, "cycle_monotonic_ns": 123})
    try:
        lock = threading.Lock()
        with (probe.call(SimpleNamespace(channel="test"), "MotorChainRobot.get_joint_pos"),
              probe.lock(SimpleNamespace(channel="test"), lock, "test")):
            assert lock.locked()
    finally:
        active_timing.reset(token)
    assert probe.started_ns == first_start
    assert events(probe) == []


def test_report_correlates_same_lock_owner_and_nested_chain_wait():
    def row(span_id, parent_id, site, tid, lock_id, start, acquired, end, released):
        return {"span_id": span_id, "parent_id": parent_id, "site": site,
                "thread_id": tid, "lock_id": lock_id, "start_ns": start,
                "acquired_ns": acquired, "body_end_ns": end, "released_ns": released,
                "channel": "can_left", "kind": "lock" if lock_id else "call",
                "cycle_ns": 42 if tid == 1 else None, "stage_prefix": "left.",
                "wait_cpu_ns": 1, "hold_cpu_ns": 2}

    rows = [
        row(1, None, "MotorChainRobot.get_joint_pos", 1, None, 100, 100, 198, 200),
        row(2, 1, "MotorChainRobot.get_joint_pos._state_lock", 1, 10, 105, 185, 190, 195),
        row(1, None, "MotorChainRobot.update._state_lock", 2, 10, 80, 85, 180, 182),
        row(2, 1, "DMChainCanInterface.set_commands.command_lock", 2, 20, 90, 170, 175, 178),
        # Another arm's state lock must never be attributed to the wait above.
        row(1, None, "MotorChainRobot.update._state_lock", 3, 99, 80, 85, 180, 182),
    ]
    report = getter_breakdown(rows)
    assert report["getter_breakdown"][0]["acquire_interval_fraction"] == pytest.approx(0.8)
    slow = report["slowest_getters"][0]
    assert slow["outside_lock_ns"] == 15
    assert slow["observed_holder_overlap_ns"] == 75
    owners = slow["observed_holders_during_wait"]
    assert len(owners) == 1 and owners[0]["thread_id"] == 2
    assert owners[0]["nested_spans"][0]["wait_ns"] == 80


def test_capture_deadline_stops_new_events(tmp_path):
    probe = LockProbe(tmp_path)
    probe.arm()
    probe.deadline_ns = time.monotonic_ns() - 1
    lock = threading.Lock()
    with probe.lock(SimpleNamespace(channel="test"), lock, "test"):
        assert lock.locked()
    assert events(probe) == []


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_duration(seconds, tmp_path):
    with pytest.raises(ValueError):
        LockProbe(tmp_path, seconds=seconds)
