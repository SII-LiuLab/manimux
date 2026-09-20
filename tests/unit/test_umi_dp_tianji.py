"""UMI observation timing, frame identity, RTC clocks and embodiment contracts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from manimux.policy_adapter.umi_dp import tianji as policy_plugin
from scipy.spatial.transform import Rotation

from manimux.cli import load_config
from manimux.policy_adapter.umi_dp.history import (
    HistoryStrategy,
    MeasuredHistory,
    WindowSnapshot,
    align_rtc_condition,
)
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.types import (
    ActionContext,
    ActionHorizon,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)

ROOT = Path(__file__).resolve().parents[2]


def snapshot(time_ns, seq, *, value=0.0, capture_ns=None):
    data = np.full((8, 8, 3), seq % 256, dtype=np.uint8)
    state = RobotState(
        {name: np.array([value, 0, 0, 0, 0, 0, 0, 0.8]) for name in ("left_arm", "right_arm")},
        time_ns,
        seq,
    )
    frames = {
        name: SensorFrame(name, data, time_ns if capture_ns is None else capture_ns, seq)
        for name in ("left_wrist", "right_wrist")
    }
    return ObservationSnapshot(state, frames)


class FakeKin:
    num_arm_joints = 7
    fail = False

    def fk(self, joints, grip):
        result = np.eye(4)
        result[:3, 3] = joints[:3]
        result[:3, :3] = Rotation.from_rotvec(joints[3:6]).as_matrix()
        return result

    def ik(self, target, seed, grip):
        return not self.fail, np.r_[
            target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_rotvec(), seed[6]
        ]


@pytest.fixture
def adapter(monkeypatch):
    kin = FakeKin()
    monkeypatch.setattr(policy_plugin, "build_kinematics", lambda *a, **k: kin)
    config = load_config(ROOT / "manimux/configs/experiments/pass_ball/tianji_umi_dp_default.yaml")
    config["robot"]["type"] = "mock"
    return policy_plugin.UmiDpTianjiAdapter(config["robot"], config["policy"]), kin, config


def request_for(adapter, *, timestamp=1000000000):
    previous = snapshot(timestamp - 100000000, 1)
    current = snapshot(timestamp, 2, value=0.01)
    frames = dict(current.frames)
    frames.update({name + "_prev": frame for name, frame in previous.frames.items()})
    window = WindowSnapshot(current.state, frames, previous)
    request = InferenceRequest("test", 1, timestamp, timestamp + 10**9, window)
    return adapter.prepare_request(request)


def actions_for(request, horizon):
    return {
        "actions": [
            {
                f"{side}_ee_pose": request.xpolicylab_state[f"{side}_ee_pose"].copy()
                for side in ("left", "right")
            }
            | {f"{side}_ee_joint_state": np.array([0.8]) for side in ("left", "right")}
            for _ in range(horizon)
        ]
    }


def test_history_uses_each_tick_and_distinct_capture_frames():
    history = MeasuredHistory(camera_names=("left_wrist", "right_wrist"))
    start = 10**9
    for tick in range(26):
        # Ten repeated polls of one camera frame retain its capture identity.
        seq = tick // 8
        history.observe(snapshot(start + tick * 4000000, seq, capture_ns=start + seq * 32000000))
    window = history.window(start + 100000000)
    assert window is not None
    assert len(history.samples) == 4
    assert window.state.monotonic_ns - window.previous.state.monotonic_ns == 96000000
    assert window.frames["left_wrist"].sequence != window.frames["left_wrist_prev"].sequence


def test_history_rejects_fake_request_history_stale_and_skewed():
    history = MeasuredHistory(camera_names=("left_wrist", "right_wrist"))
    history.observe(snapshot(10**9, 1))
    history.observe(snapshot(2 * 10**9, 2))
    assert history.window(2 * 10**9) is None
    history.reset()
    history.observe(snapshot(10**9, 1))
    history.observe(snapshot(1100000000, 2))
    assert history.window(1100000000) is not None
    assert history.window(2 * 10**9) is None
    history.reset()
    history.observe(snapshot(10**9, 1, capture_ns=800000000))
    assert not history.samples


def test_history_delegates_and_validates_rtc_constraints():
    for name in ("default", "rtc"):
        config = load_config(ROOT / f"manimux/configs/experiments/pass_ball/tianji_umi_dp_{name}.yaml")
        strategy = HistoryStrategy(config)
        assert strategy.name == ("manimux" if name == "default" else "rtc")
        assert strategy.required_sampling_modes == frozenset(
            {"default" if name == "default" else "rtc"}
        )
    config["inference"]["rtc"]["initial_delay_steps"] = 9
    with pytest.raises(ValueError, match="initial_delay"):
        HistoryStrategy(config)


@pytest.mark.parametrize("offset_ns", [33333333, 100000000])
def test_rtc_resamples_committed_rows_on_new_target_clock(offset_ns):
    dt = 100000000
    rows = np.tile(np.arange(4)[:, None], (1, 8)).astype(float)
    active = ActionHorizon(10**9, dt, "p", {"left_arm": rows, "right_arm": rows + 10})
    timeline = SimpleNamespace(active_horizon=lambda: active, cursor=lambda now: 1)
    request = RtcInferenceRequest(
        "test",
        1,
        1100000000,
        2 * 10**9,
        snapshot(1100000000, 1),
        action_condition=np.zeros((4, 16)),
        condition_weights=np.ones(4),
    )
    align_rtc_condition(
        request,
        timeline,
        1100000000,
        offset_ns=offset_ns,
        dt_ns=dt,
        group_order=("left_arm", "right_arm"),
        horizon=4,
    )
    assert request.action_condition[0, 0] == pytest.approx(1 + offset_ns / dt)
    assert request.condition_weights[-1] == 0
    assert request.condition_weights[0] == 1


def test_adapter_preserves_rtc_fields_and_previous_fk(adapter):
    model, _, _ = adapter
    prepared = request_for(model)
    assert prepared.xpolicylab_additional_info["umi_dp"]["frame_times_ns"] == [
        900000000,
        1000000000,
    ]
    assert prepared.xpolicylab_state["left_ee_pose_prev"][0] == 0
    assert prepared.xpolicylab_state["left_ee_pose"][0] == 0.01
    prepared.action_condition = np.tile(
        np.r_[
            prepared.observation.state.groups["left_arm"],
            prepared.observation.state.groups["right_arm"],
        ],
        (model.horizon, 1),
    )
    prepared.condition_weights = np.ones(model.horizon)
    prepared.rtc_beta = 2.5
    result = model.prepare_request(prepared)
    assert result.rtc_beta == 2.5
    np.testing.assert_allclose(
        result.action_condition[0, :7], result.xpolicylab_state["left_ee_pose"]
    )


def test_adapter_output_clock_reset_anchor_and_ik_failure(adapter):
    model, kin, _ = adapter
    request = request_for(model)
    raw = actions_for(request, model.horizon)
    chunk = model.decode_action(raw, ActionContext(1, 10**9, 10**9))
    assert chunk.observation_time_ns == 10**9 + model.offset_ns
    assert chunk.horizon_steps == model.horizon
    np.testing.assert_allclose(
        chunk.groups["left_arm"][0], request.observation.state.groups["left_arm"]
    )
    with pytest.raises(ValueError, match="matching observation"):
        model.decode_action(raw, ActionContext(1, 10**9, 10**9))
    request_for(model)
    kin.fail = True
    with pytest.raises(ValueError, match="IK failed"):
        model.decode_action(raw, ActionContext(1, 10**9, 10**9))


def test_adapter_accepts_the_ws_client_unwrapped_action_list(adapter):
    # XPolicyLabWebSocketClient.infer returns payload["actions"] for plain sampling.
    model, _, _ = adapter
    request = request_for(model)
    steps = actions_for(request, model.horizon)["actions"]
    chunk = model.decode_action(steps, ActionContext(1, 10**9, 10**9))
    assert chunk.horizon_steps == model.horizon


def test_partition_decode_uses_measured_context_without_parent_anchors(adapter):
    model, _, _ = adapter
    request = request_for(model)
    raw = actions_for(request, model.horizon)
    context = ActionContext(1, 10**9, 10**9, measured_state=request.observation.state)
    serial = model.decode_action(raw, context)
    assert not model.anchors
    for partition in model.decode_partitions:
        decoded = model.decode_action_partition(raw, context, partition)
        assert set(decoded.groups) == {partition}
        np.testing.assert_allclose(decoded.groups[partition], serial.groups[partition])
        assert decoded.observation_time_ns == serial.observation_time_ns
        assert decoded.source_offset_steps == serial.source_offset_steps
    with pytest.raises(ValueError, match="partition"):
        model.decode_action_partition(raw, context, "unknown")


def test_adapter_decodes_each_arm_from_measured_state_without_anchors(adapter):
    # A decoder process never sees the parent's prepare_request anchors.
    model, _, _ = adapter
    assert model.supports_context_only_decode
    raw = actions_for(request_for(model), model.horizon)
    model.anchors.clear()
    measured = RobotState(
        {name: np.array([0, 0, 0, 0, 0, 0, 0.3, 0.8]) for name in ("left_arm", "right_arm")},
        10**9,
        3,
    )
    context = ActionContext(1, 10**9, 10**9, measured_state=measured)
    whole = model.decode_action(raw, context)
    # FakeKin carries the seed's seventh joint: the measured state, not the anchor.
    np.testing.assert_array_equal(whole.groups["left_arm"][:, 6], 0.3)
    assert model.decode_partitions == ("left_arm", "right_arm")
    for name in model.decode_partitions:
        part = model.decode_action_partition(raw, context, name)
        assert set(part.groups) == {name}
        np.testing.assert_array_equal(part.groups[name], whole.groups[name])
        assert part.source_offset_steps == whole.source_offset_steps
        assert part.observation_time_ns == whole.observation_time_ns
    with pytest.raises(ValueError, match="unknown UMI decode partition"):
        model.decode_action_partition(raw, context, "left_gripper")


@pytest.mark.parametrize("failure", ["horizon", "quaternion", "gripper", "missing_history"])
def test_adapter_rejects_contract_errors(adapter, failure):
    model, _, _ = adapter
    request = request_for(model)
    raw = actions_for(request, model.horizon)
    if failure == "horizon":
        raw["actions"].pop()
    elif failure == "quaternion":
        raw["actions"][0]["left_ee_pose"][3:] = 0
    elif failure == "gripper":
        raw["actions"][0]["left_ee_joint_state"][0] = 1.1
    else:
        with pytest.raises(ValueError, match="measured history"):
            model.build_observation(snapshot(10**9, 1))
        return
    with pytest.raises(ValueError):
        model.decode_action(raw, ActionContext(1, 10**9, 10**9))


def test_timestamped_camera_retains_sequence_and_detects_clock_jump(monkeypatch):
    from manimux.embodiments.sensor.camera_server import timestamped as camera_sensor

    wall = [100000000000]
    clock = SimpleNamespace(now_ns=lambda: wall[0] - 90000000000)
    monkeypatch.setattr(camera_sensor.time, "time_ns", lambda: wall[0])
    bundle = {
        "frames": {"left_wrist": np.zeros((8, 8, 3), np.uint8)},
        "timestamps": {"left_wrist": 100.0},
    }
    client = SimpleNamespace(try_recv_bundle=lambda: bundle, close=lambda: None)
    monkeypatch.setattr(camera_sensor, "CameraSubscriber", lambda endpoint: client)
    sensor = camera_sensor.TimestampedCameraSensor(
        {"options": {"camera_names": ["left_wrist"]}}, clock
    )
    sensor.start()
    first = sensor.read()["left_wrist"]
    wall[0] += 4000000
    assert sensor.read()["left_wrist"] is first
    assert first.capture_monotonic_ns == 10000000000
    monkeypatch.setattr(camera_sensor.time, "time_ns", lambda: wall[0] + 50000000)
    with pytest.raises(RuntimeError, match="Wall clock jumped"):
        sensor.read()


def test_history_strategy_asks_runtime_to_drop_plans_while_paused():
    assert HistoryStrategy.discard_plans_while_paused is True


@pytest.mark.parametrize("discard", [True, False])
def test_pause_submits_and_commits_nothing_only_when_the_strategy_asks(
    monkeypatch, tmp_path, discard
):
    from manimux.runtime import edge
    from manimux.types import ActionChunk, InferenceResponse
    from manimux.viewer.publisher import ViewerControl

    class Clock:
        now = 10**9

        def now_ns(self):
            return self.now

        def sleep_until_ns(self, target_ns):
            self.now = max(self.now + 1, target_ns)

    clock = Clock()
    submitted_while, holder = [], {}

    class InstantPolicy:
        is_alive = True
        pending = None

        def start(self):
            pass

        def submit_latest(self, request):
            submitted_while.append(holder["runtime"]._state)
            chunk = ActionChunk(
                plan_id=f"p{request.request_seq}",
                request_seq=request.request_seq,
                observation_time_ns=request.observation_time_ns,
                created_time_ns=clock.now_ns(),
                action_space="joint_position",
                dt_ns=50_000_000,
                groups={
                    name: np.tile(values, (20, 1))
                    for name, values in request.observation.state.groups.items()
                },
            )
            self.pending = InferenceResponse(
                request.session_id,
                request.request_seq,
                clock.now_ns(),
                0.0,
                chunk,
                observation_time_ns=request.observation_time_ns,
            )

        def poll(self):
            result, self.pending = self.pending, None
            return result

        def close(self):
            self.is_alive = False

    class Strategy(edge.DefaultChunkStrategy):
        discard_plans_while_paused = discard

    monkeypatch.setattr(edge, "PolicyWorkerClient", lambda *_: InstantPolicy())
    config = load_config(ROOT / "tests/fixtures/runtime.yaml")
    config["sensors"] = []
    config["run"]["max_steps"] = 10_000
    runtime = edge.EdgeRuntime(config, tmp_path, clock=clock, strategy=Strategy(config))
    holder["runtime"] = runtime
    controls = iter(
        [ViewerControl(paused=True)] * 60
        + [ViewerControl(paused=False)] * 60
        + [ViewerControl(paused=True, finish_requested=True)]
    )
    monkeypatch.setattr(runtime._viewer, "poll_control", lambda: next(controls))
    runtime.run()
    assert edge.RuntimeState.RUNNING in submitted_while
    # Only the opt-in strategy stops inference during the pause; the default is unchanged.
    assert (edge.RuntimeState.PAUSED in submitted_while) is not discard
