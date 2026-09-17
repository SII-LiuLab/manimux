"""Physical replay scheduling with fake time and fake hardware only."""

import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from manimux.collection.yam import joint_replay as replay


class Time:
    value = 0

    def now(self):
        return self.value

    def unix(self):
        return 1_789_000_000_000_000_000 + self.value

    def sleep(self, seconds):
        self.value += round(seconds * 1e9)


class Backend:
    def __init__(self, clock, delay_ns=0):
        self.clock, self.delay_ns = clock, delay_ns
        self.targets, self.sent = {}, []

    def observation(self, group):
        return {"joint_pos": np.zeros(6), "gripper_pos": np.array([0.7]),
                "feedback_timestamp_ns": self.clock.unix()}

    @contextmanager
    def target_batch(self):
        yield
        self.clock.value += self.delay_ns
        self.sent.append((self.clock.value, {k: v.copy() for k, v in self.targets.items()}))

    def set_target(self, group, values):
        self.targets[group] = values

    def controller_input(self, group):
        return {"joint_pos": self.targets[group][:6], "timestamp_ns": self.clock.unix()}


class Recorder:
    def __init__(self):
        self.rows = []

    def tick(self, *args, **kwargs):
        self.rows.append((args, kwargs))


def dataset():
    values = np.tile(np.arange(13)[:, None], (1, 7)).astype(float)
    return SimpleNamespace(source="command", target_hz=60, time_s=np.arange(13) / 60,
                           trajectories={key: {arm: values + i * 100 for arm in ("left", "right")}
                                         for i, key in enumerate(replay.METHODS.values())})


@pytest.mark.parametrize("method", ["original", "10", "30"])
def test_physical_scheduler_selects_method_and_holds_gripper(method):
    clock, recorder = Time(), Recorder()
    backend = Backend(clock, delay_ns=1_000_000)
    data = dataset()
    journal = replay.execute_trajectory(
        backend, recorder, data, method, {"left": 0.2, "right": 0.9}, settle_s=0.02,
        now_ns=clock.now, unix_ns=clock.unix, sleep=clock.sleep,
    )
    assert len(backend.sent) == 13
    np.testing.assert_allclose(np.diff([t for t, _ in backend.sent]),
                               1e9 / 60 + backend.delay_ns, atol=1)
    for (args, kwargs), (_, targets), tick in zip(
        recorder.rows, backend.sent, journal["ticks"], strict=True,
    ):
        for arm, grip in (("left", 0.2), ("right", 0.9)):
            np.testing.assert_array_equal(targets[f"{arm}_arm"][:6],
                                          data.trajectories[replay.METHODS[method]][arm]
                                          [tick["frame"], :6])
            assert targets[f"{arm}_arm"][6] == grip
            assert args[1][arm]["feedback_timestamp_ns"] < tick["command_unix_ns"]
        assert kwargs["tick_monotonic_ns"] == tick["tick_monotonic_ns"]
    assert journal["ticks"][-1]["frame"] == 12  # final hold / settling captured


def test_late_execution_skips_expired_frames_without_catchup_bursts():
    clock, recorder = Time(), Recorder()
    backend = Backend(clock, delay_ns=40_000_000)
    result = replay.execute_trajectory(
        backend, recorder, dataset(), "30", {"left": 0.3, "right": 0.7}, settle_s=0,
        now_ns=clock.now, unix_ns=clock.unix, sleep=clock.sleep,
    )
    frames = [r["frame"] for r in result["ticks"]]
    assert frames == [0, 3, 6, 10]
    assert min(np.diff([t for t, _ in backend.sent])) >= 1e9 / 60 + backend.delay_ns


def test_late_observation_cannot_squeeze_next_command_against_grid_boundary():
    clock, recorder = Time(), Recorder()
    backend = Backend(clock, delay_ns=1_000)
    original_observation = backend.observation
    delayed = False

    def observation(group):
        nonlocal delayed
        # A feedback read finishes just before the following 60 Hz boundary.
        # Sending on that boundary would put two commands only 0.667 ms apart.
        if not delayed:
            clock.value += 16_000_000
            delayed = True
        return original_observation(group)

    backend.observation = observation
    replay.execute_trajectory(
        backend, recorder, dataset(), "original", {"left": 0.3, "right": 0.7},
        settle_s=0, now_ns=clock.now, unix_ns=clock.unix, sleep=clock.sleep,
    )
    assert len(backend.sent) > 2
    assert min(np.diff([t for t, _ in backend.sent])) >= 1e9 / 60


def test_interrupted_execution_retains_partial_journal():
    clock, recorder, journal = Time(), Recorder(), {}
    backend = Backend(clock)

    def stop_after_first(seconds):
        if backend.sent:
            raise KeyboardInterrupt
        clock.sleep(seconds)

    with pytest.raises(KeyboardInterrupt):
        replay.execute_trajectory(
            backend, recorder, dataset(), "original", {"left": 0.4, "right": 0.5},
            now_ns=clock.now, unix_ns=clock.unix, sleep=stop_after_first, journal=journal,
        )
    assert len(journal["ticks"]) == 1
    assert journal["start_unix_ns"] == clock.unix()


def test_prepare_moves_only_joints_and_reaches_common_start():
    clock = Time()
    backend = Backend(clock)
    groups = dataset().trajectories["resample_1"]
    replay.prepare_start(backend, groups, {"left": 0.1, "right": 0.8}, 0.1,
                         sleep=clock.sleep, now=lambda: clock.now() / 1e9)
    for arm, grip in (("left", 0.1), ("right", 0.8)):
        assert all(targets[f"{arm}_arm"][6] == grip for _, targets in backend.sent)
        np.testing.assert_array_equal(backend.sent[-1][1][f"{arm}_arm"][:6], groups[arm][0, :6])


def test_operator_wait_aborts_dead_driver_before_prompt():
    def failed():
        raise RuntimeError("motor chain stopped")

    called = []
    backend = SimpleNamespace(driver=SimpleNamespace(check_health=failed))
    with pytest.raises(RuntimeError, match="motor chain stopped"):
        replay.wait_for_operator("Enter to move", backend, prompt=called.append)
    assert not called


def test_physical_replay_rejects_feedback_before_connecting(tmp_path):
    data = dataset()
    data.source = "feedback"
    with pytest.raises(ValueError, match="requires saved controller commands"):
        replay.run_trials(data, None, None, method="original", save_root=tmp_path, prepare_s=5)


def test_replay_backend_fault_never_uses_bimanual_cached_stop():
    calls = []
    backend = replay.ReplayBackend.__new__(replay.ReplayBackend)
    backend._mutex = threading.RLock()
    backend._enabled = True
    backend.driver = SimpleNamespace(
        hold_healthy_arms=lambda: calls.append("hold_healthy"),
        stop=lambda: pytest.fail("must not submit both cached joint targets"),
    )
    backend._stop_on_error(RuntimeError("motor fault"))
    assert backend._fault == "motor fault"
    assert backend._halted and not backend._enabled
    assert calls == ["hold_healthy"]


def test_operator_wait_monitors_health_without_enter(monkeypatch):
    count = 0

    def health():
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("motor failed while waiting")

    monkeypatch.setattr(replay.select, "select", lambda *args: ([], [], []))
    backend = SimpleNamespace(driver=SimpleNamespace(check_health=health))
    with pytest.raises(RuntimeError, match="while waiting"):
        replay.wait_for_operator("Enter", backend)


def test_cameras_warm_before_motors_and_close_after_motors(monkeypatch, tmp_path):
    events = []

    class Camera:
        def start(self):
            events.append("camera-warm")

        def stop(self):
            events.append("camera-close")

    class TrialBackend(Backend):
        def __init__(self, *args, **kwargs):
            super().__init__(Time())
            self.driver = SimpleNamespace(check_health=lambda: None,
                                          native_joint_sources=lambda: [],
                                          get_state=lambda: SimpleNamespace(
                                              groups={"left_arm": np.zeros(7)}))

        def connect(self, **kwargs):
            events.append("motor-connect")

        def close(self):
            events.append("motor-close")

        def hold_for_shutdown(self):
            return {"left": {"status": "hold_submitted"}}

    def prompt(_):
        events.append("prompt")
        raise KeyboardInterrupt

    monkeypatch.setattr(replay, "ReplayBackend", TrialBackend)
    monkeypatch.setattr("manimux.collection.yam.runtime.build_cameras_from_config",
                        lambda _: [Camera()])
    monkeypatch.setattr("manimux.collection.yam.camera.worker.CameraWorker", lambda cam: cam)
    monkeypatch.setattr("manimux.collection.yam.data.recorder.EpisodeRecorder",
                        lambda *args, **kwargs: SimpleNamespace(is_recording=False,
                                                               is_saving=False))
    def home(_):
        events.append("home")
        return {"status": "completed"}

    monkeypatch.setattr(replay, "home_after_replay", home)
    data = dataset()
    data.episode, data.origin_ns = tmp_path / "source", 0
    with pytest.raises(KeyboardInterrupt):
        replay.run_trials(data, SimpleNamespace(manimux_config="test.yaml"), None,
                          method="10", save_root=tmp_path, prepare_s=5, prompt=prompt)
    assert events == ["camera-warm", "motor-connect", "prompt", "home",
                      "motor-close", "camera-close"]


@pytest.mark.parametrize("reached", [True, False])
def test_home_uses_actual_feedback_and_resets_collection_cache(reached):
    clock = Time()
    events = []
    pose = np.r_[np.zeros(6) if reached else np.ones(6), 0.4]
    driver = SimpleNamespace(
        home=lambda **kw: events.append(("home", kw)),
        get_state=lambda: SimpleNamespace(groups={"left_arm": pose, "right_arm": pose},
                                         monotonic_ns=clock.now()),
    )
    backend = SimpleNamespace(driver=driver, pause=lambda: events.append("reset"))
    options = {"timeout_s": 0.05, "now": lambda: clock.now() / 1e9, "sleep": clock.sleep}
    if reached:
        info = replay.home_after_replay(backend, **options)
        assert info["max_abs_joint_rad"] == 0
        assert info["feedback"]["left_arm"][6] == 0.4
    else:
        with pytest.raises(replay.HomeNotReached, match="Home 未到位"):
            replay.home_after_replay(backend, **options)
    assert events == [("home", {"release_grippers": False}), "reset"]


@pytest.mark.parametrize("outcome", ["complete", "interrupt", "software", "fault",
                                     "home_miss", "home_fault", "save_fault"])
def test_trials_freeze_then_recover_then_save_before_closing(monkeypatch, tmp_path, outcome):
    events = []
    saved_dirs = []
    healthy = True

    def check_health():
        if not healthy:
            raise RuntimeError("right motor fault")

    class Camera:
        def start(self):
            pass

        def stop(self):
            pass

    class TrialBackend(Backend):
        def __init__(self, *args, **kwargs):
            super().__init__(Time())
            self.driver = SimpleNamespace(check_health=check_health,
                                          native_joint_sources=lambda: [],
                                          get_state=lambda: SimpleNamespace(
                                              groups={"left_arm": np.zeros(7)}))

        def connect(self, **kwargs):
            pass

        def pause(self):
            pass

        def close(self):
            events.append("close")

        def hold_for_shutdown(self):
            return {"left": {"status": "hold_submitted"},
                    "right": {"status": "hold_submitted" if healthy else "unavailable"}}

    class TrialRecorder:
        is_recording = False
        is_saving = False

        def __init__(self, *args, **kwargs):
            pass

        def start(self, *args, **kwargs):
            self.is_recording = True
            self.path = tmp_path / f"episode-{len(saved_dirs)}"
            self.path.mkdir()
            saved_dirs.append(self.path)
            events.append("record")
            return self.path

        def freeze(self):
            self.is_recording = False
            self.is_saving = True
            events.append("freeze")

        def stop(self):
            assert not self.is_recording
            self.is_saving = False
            events.append("save")
            if outcome == "save_fault":
                raise OSError("video save failed")
            (self.path / "metadata.json").write_text(json.dumps(
                {"extra": {"timing": {"commands": {"left": {"actual_hz": 60}}}}}))
            return self.path

    def execute(*args, **kwargs):
        nonlocal healthy
        if outcome == "interrupt":
            raise KeyboardInterrupt
        if outcome == "fault":
            healthy = False
            raise RuntimeError("motor fault")
        if outcome == "software":
            raise ValueError("recorder error")

    def home(backend):
        nonlocal healthy
        assert healthy
        events.append("home")
        if outcome == "home_miss":
            raise replay.HomeNotReached("Home 未到位")
        if outcome == "home_fault":
            healthy = False
            raise RuntimeError("motor failed during Home")
        return {"status": "completed"}

    monkeypatch.setattr(replay, "ReplayBackend", TrialBackend)
    monkeypatch.setattr(replay, "prepare_start", lambda *a: None)
    monkeypatch.setattr(replay, "execute_trajectory", execute)
    monkeypatch.setattr(replay, "home_after_replay", home)
    monkeypatch.setattr("manimux.collection.yam.runtime.build_cameras_from_config",
                        lambda _: [Camera()])
    monkeypatch.setattr("manimux.collection.yam.camera.worker.CameraWorker", lambda cam: cam)
    monkeypatch.setattr("manimux.collection.yam.data.recorder.EpisodeRecorder", TrialRecorder)
    data = dataset()
    data.episode, data.origin_ns = tmp_path / "source", 0
    def prompt(message):
        if "支撑" in message:
            assert "close" not in events
            events.append("confirm-release")
            return "RELEASE"

    options = dict(method="all", save_root=tmp_path, prepare_s=5, prompt=prompt)
    if outcome == "complete":
        replay.run_trials(data, SimpleNamespace(manimux_config="fake.yaml"), None, **options)
        assert events == ["record", "freeze", "home", "save"] * 3 + ["close"]
        assert all(json.loads((p / "replay-experiment.json").read_text())["home"]["status"]
                   == "completed" for p in saved_dirs)
    elif outcome in {"home_miss", "home_fault", "fault"}:
        with pytest.raises(RuntimeError):
            replay.run_trials(data, SimpleNamespace(manimux_config="fake.yaml"), None, **options)
        expected_home = [] if outcome == "fault" else ["home"]
        assert events == ["record", "freeze", *expected_home, "confirm-release", "save", "close"]
        manifest = json.loads((saved_dirs[0] / "replay-experiment.json").read_text())
        assert manifest["home"]["status"] == ("skipped_unhealthy" if outcome == "fault"
                                               else "failed")
        assert manifest["home"]["support_confirmed"]
    else:
        error = {"interrupt": KeyboardInterrupt, "software": ValueError, "save_fault": OSError}
        with pytest.raises(error[outcome]):
            replay.run_trials(data, SimpleNamespace(manimux_config="fake.yaml"), None, **options)
        assert events == ["record", "freeze", "home", "save", "close"]
        manifest = json.loads((saved_dirs[0] / "replay-experiment.json").read_text())
        assert manifest["status"] == "interrupted"
        assert manifest["home"]["status"] == "completed"
        assert manifest["source"] == "command"


def test_support_requires_explicit_token_despite_enter_interrupt_or_eof():
    answers = iter(["", KeyboardInterrupt(), EOFError(), "yes", "RELEASE"])
    seen = []

    def prompt(_):
        answer = next(answers)
        seen.append(answer)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    replay.wait_for_support(prompt, sleep=lambda _: None)
    assert len(seen) == 5


def test_recovery_signals_do_not_interrupt_cleanup_and_handlers_are_restored():
    before = {sig: replay.signal.getsignal(sig)
              for sig in (replay.signal.SIGINT, replay.signal.SIGTERM)}
    with replay.protect_recovery():
        for sig in before:
            replay.signal.raise_signal(sig)
    assert all(replay.signal.getsignal(sig) == handler for sig, handler in before.items())


def test_healthy_but_unreached_home_does_not_trigger_automatic_retry(monkeypatch):
    events = []
    backend = SimpleNamespace(
        driver=SimpleNamespace(check_health=lambda: None,
                               get_state=lambda: SimpleNamespace(groups={"left_arm": np.ones(7)})),
        hold_for_shutdown=lambda: {"left": {"status": "hold_submitted"}},
        close=lambda: events.append("close"),
    )
    def prompt(_):
        assert not events
        events.append("support")
        return "RELEASE"

    guard = replay.ReplayShutdown(backend, prompt)
    guard.home_attempted = True
    guard.home_result = {"status": "completed"}
    monkeypatch.setattr(replay, "home_after_replay", lambda _: pytest.fail("must not retry"))
    guard.close()
    assert events == ["support", "close"]


def test_failure_after_home_during_save_does_not_automatically_close_or_retry_home(monkeypatch):
    events = []
    def dead():
        raise RuntimeError("motor died during encoding")

    backend = SimpleNamespace(
        driver=SimpleNamespace(check_health=dead, get_state=dead),
        hold_for_shutdown=lambda: {"left": {"status": "hold_submitted"},
                                   "right": {"status": "unavailable"}},
        close=lambda: events.append("close"),
    )
    def prompt(_):
        assert not events
        events.append("support")
        return "RELEASE"

    guard = replay.ReplayShutdown(backend, prompt)
    guard.home_attempted = True
    guard.home_result = {"status": "completed"}
    monkeypatch.setattr(replay, "home_after_replay", lambda _: pytest.fail("must not retry"))
    guard.close()
    assert events == ["support", "close"]
