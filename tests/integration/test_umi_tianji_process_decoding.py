"""Process decoding behind a delegating strategy plugin, and UMI Tianji per-arm decode."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from manimux.cli import load_config, prepare_experiment
from manimux.policies.fake import FakePolicyAdapter
from manimux.runtime import build_runtime
from manimux.runtime.edge import EdgeRuntime
from manimux.runtime.inference import DefaultChunkStrategy
from manimux.viewer import ViewerControl

ROOT = Path(__file__).resolve().parents[2]
START = np.radians([50, -40, -30, -100, -65, 0, 40])


class SlowAdapter(FakePolicyAdapter):
    supports_context_only_decode = True

    def decode_action(self, raw, context):
        # CPU work that holds the GIL, so only a process boundary keeps control ticking.
        until = time.monotonic() + 0.25
        while time.monotonic() < until:
            pass
        return super().decode_action(raw, context)


def build_slow_adapter(robot, policy, *, kinematics=None):
    return SlowAdapter({}, {})


class DelegatingStrategy:
    """Wraps a strategy the way the UMI measured-history plugin does."""

    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)


def build_delegating_strategy(config):
    return DelegatingStrategy(DefaultChunkStrategy(config))


def plugin_config():
    config = load_config(ROOT / "tests/fixtures/runtime.yaml")
    config["policy"]["adapter"]["type"] = f"{__name__}:build_slow_adapter"
    config["policy"]["action_decoding"] = "process"
    config["policy"]["inference_delay_s"] = 0.01
    config["inference"]["algorithm"] = f"{__name__}:build_delegating_strategy"
    config["inference"]["inference_schedule"] = "single_inflight"
    return config


def events_of(result):
    lines = (result.episode_dir / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


@pytest.mark.parametrize("expired", [False, True])
def test_delegating_plugin_keeps_control_ticking_during_process_decode(tmp_path, expired):
    config = plugin_config()
    config["policy"]["horizon_steps"] = 6 if expired else 20
    config["executor"]["type"] = "direct"
    config["run"]["max_steps"] = 100
    runtime = build_runtime(config, tmp_path)
    assert isinstance(runtime._strategy, DelegatingStrategy)
    result = runtime.run()
    events = events_of(result)
    sent = [e for e in events if e["kind"] == "decode_submitted"]
    assert sent
    ticks = zarr.open(str(result.episode_dir / "data.zarr"), mode="r")["ticks/monotonic_ns"][:]
    seed_ns = sent[0]["seed_time_ns"]
    assert np.count_nonzero((ticks > seed_ns) & (ticks < seed_ns + 220_000_000)) >= 8
    assert all(not p.is_alive() for p in runtime._decoder._processes)
    if expired:
        assert result.accepted_plans == 0
        assert any(e.get("reason") == "no_future_horizon" for e in events)
    else:
        accepted = [e for e in events if e["kind"] == "plan_accepted"]
        assert accepted and accepted[0]["decode_mode"] == "process"
        assert accepted[0]["decode_ms"] >= 240


class PauseDuringDecode:
    def __init__(self, runtime, home=False):
        self.runtime = runtime
        self.home = home
        self.interrupted = False
        self.pause_until = 0.0

    def poll_control(self):
        if self.runtime._decoder.busy and not self.interrupted:
            self.interrupted = True
            self.pause_until = time.monotonic() + 0.1
            if self.home:
                return ViewerControl(paused=True, home_requested=True)
        return ViewerControl(paused=time.monotonic() < self.pause_until)

    def set_state_metadata(self, *args, **kwargs):
        pass

    def publish_plan(self, *args, **kwargs):
        pass

    def publish_state(self, *args, **kwargs):
        pass

    def publish_event(self, *args, **kwargs):
        pass

    def close(self):
        pass


@pytest.mark.parametrize("home", [False, True])
def test_delegating_plugin_pause_discards_pending_decode(tmp_path, home):
    config = plugin_config()
    config["run"]["max_steps"] = 120
    runtime = build_runtime(config, tmp_path)
    runtime._viewer = PauseDuringDecode(runtime, home=home)
    result = runtime.run()
    events = events_of(result)
    assert runtime._viewer.interrupted
    assert any(e["kind"] == "inference_rejected" and e["request_seq"] == 1 for e in events)
    accepted = [e for e in events if e["kind"] == "plan_accepted"]
    assert accepted and all(e["request_seq"] > 1 for e in accepted)


def test_process_decoding_rejects_a_plugin_delegating_to_an_unsupported_strategy(tmp_path):
    # manimux and rtc delegates may use process decoding (see test_tianji_rtc_process.py).
    config = plugin_config()
    strategy = DelegatingStrategy(SimpleNamespace(name="paint"))
    with pytest.raises(ValueError, match="requires the manimux or rtc strategy"):
        EdgeRuntime(config, tmp_path, strategy=strategy)


def test_expected_decode_time_requires_process_decoding():

    data = deepcopy(load_config(ROOT / "tests/fixtures/runtime.yaml"))
    data["inference"]["expected_decode_s"] = 0.05
    with pytest.raises(ValueError, match="expected_decode_s requires process action decoding"):
        prepare_experiment(**data)


def test_decode_seed_follows_the_active_reference_to_the_expected_start(tmp_path):
    from manimux.types import ActionChunk, RobotState

    config = plugin_config()
    config["inference"]["commit_lead_s"] = 0.02
    config["inference"]["expected_decode_s"] = 0.065
    runtime = EdgeRuntime(config, tmp_path)
    state = RobotState(
        {name: np.full(dim, -1.0) for name, dim in config["robot"]["group_dims"].items()}, 10**9, 7
    )
    now = 11 * 10**8
    start = now + 85_000_000
    # No active plan: the arm holds, so the measurement stays the seed.
    assert runtime._decode_seed(state, now) == (start, state, "measured_state")
    rows = np.arange(20, dtype=float)[:, None] * 0.01
    chunk = ActionChunk(
        "p",
        1,
        10**9,
        10**9,
        "joint_position",
        50_000_000,
        {name: np.tile(rows, (1, dim)) for name, dim in config["robot"]["group_dims"].items()},
    )
    zeros = {name: np.zeros(dim) for name, dim in config["robot"]["group_dims"].items()}
    assert runtime._timeline.commit(
        chunk,
        now_ns=10**9,
        commit_lead_ns=0,
        max_plan_age_ns=10**10,
        current_command=zeros,
        blend_steps=0,
    ).accepted
    # Row time 185ms at 50ms spacing is 3.7 rows along the committed ramp.
    start_ns, seed, source = runtime._decode_seed(state, now)
    assert (start_ns, seed.monotonic_ns, seed.sequence, source) == (
        start,
        start,
        7,
        "active_reference",
    )
    for values in seed.groups.values():
        np.testing.assert_allclose(values, 0.037)
    # A plan ending before the expected start seeds from its last row.
    _, seed, source = runtime._decode_seed(state, 10**9 + 900_000_000)
    assert source == "active_reference"
    np.testing.assert_allclose(seed.groups["left_arm"], 0.19)
    # Without an expected decode time the measurement stays the seed.
    config["inference"]["expected_decode_s"] = 0.0
    idle = EdgeRuntime(config, tmp_path)
    assert idle._decode_seed(state, now) == (now + 20_000_000, state, "measured_state")


def test_decode_forecast_requires_a_positive_expected_decode_floor():
    data = deepcopy(load_config(ROOT / "tests/fixtures/runtime.yaml"))
    data["policy"]["action_decoding"] = "process"
    data["inference"]["decode_forecast_size"] = 5
    # expected_decode_s stays at its 0.0 default: there is no initial estimate.
    with pytest.raises(ValueError, match="requires a positive expected_decode_s"):
        prepare_experiment(**data)


def test_decode_seed_follows_the_measured_decode_duration(tmp_path):
    from manimux.types import RobotState

    config = plugin_config()
    config["inference"]["commit_lead_s"] = 0.02
    config["inference"]["expected_decode_s"] = 0.05
    config["inference"]["decode_forecast_size"] = 2
    runtime = EdgeRuntime(config, tmp_path)
    state = RobotState(
        {name: np.full(dim, -1.0) for name, dim in config["robot"]["group_dims"].items()}, 10**9, 7
    )
    now = 11 * 10**8
    # The configured value is the estimate until a decode has been measured.
    assert runtime._decode_seed(state, now)[0] == now + 70_000_000
    runtime._decode_forecast.observe(200.0)
    assert runtime._decode_seed(state, now)[0] == now + 220_000_000
    # A faster decode is floored by expected_decode_s, so the seed source is stable.
    runtime._decode_forecast.observe(5.0)
    runtime._decode_forecast.observe(5.0)
    assert runtime._decode_seed(state, now)[0] == now + 70_000_000


def test_runtime_forecasts_the_next_decode_from_the_previous_one(tmp_path):
    config = plugin_config()
    config["inference"]["expected_decode_s"] = 0.05
    config["inference"]["decode_forecast_size"] = 5
    config["run"]["max_steps"] = 250
    result = build_runtime(config, tmp_path).run()
    sent = [e for e in events_of(result) if e["kind"] == "decode_submitted"]
    assert len(sent) >= 2
    # SlowAdapter spends 250 ms per decode; only the first submission has to guess.
    assert sent[0]["expected_decode_ms"] == pytest.approx(50.0)
    assert all(e["expected_decode_ms"] >= 240 for e in sent[1:])


def test_runtime_seeds_later_decodes_from_the_active_reference(tmp_path):
    config = plugin_config()
    config["inference"]["expected_decode_s"] = 0.05
    config["run"]["max_steps"] = 250
    result = build_runtime(config, tmp_path).run()
    sent = [e for e in events_of(result) if e["kind"] == "decode_submitted"]
    assert len(sent) >= 2
    assert sent[0]["seed_source"] == "measured_state"
    assert "active_reference" in {e["seed_source"] for e in sent[1:]}
    assert all(
        e["expected_start_ns"] >= e["seed_time_ns"]
        for e in sent
        if e["seed_source"] == "measured_state"
    )
    assert all(
        e["seed_time_ns"] == e["expected_start_ns"]
        for e in sent
        if e["seed_source"] == "active_reference"
    )


def wait_for_decode(decoder):
    until = time.monotonic() + 30
    while time.monotonic() < until:
        result = decoder.poll()
        if result is not None:
            return result
        time.sleep(0.001)
    raise AssertionError("action decoder did not return a result")


def test_umi_tianji_per_arm_processes_match_inline_diff_decode():
    pytest.importorskip("osqp")
    from manimux.policies.decoder import ActionDecoderClient
    from manimux.policy_adapter.umi_dp.ik_config import bind_diff_ik_profile
    from manimux.policy_adapter.umi_dp.tianji import UmiDpTianjiAdapter, matrix_pose
    from manimux.types import ActionContext, InferenceResponse, RobotState

    config = load_config(ROOT / "manimux/configs/experiments/pass_ball/tianji_umi_dp_default.yaml")
    config["robot"]["type"] = "mock"
    config["policy"]["horizon_steps"] = 64
    config["policy"]["adapter"]["ik_backend"] = "diff"
    bind_diff_ik_profile(config)
    adapter = UmiDpTianjiAdapter(config["robot"], config["policy"])
    actions = []
    for index in range(64):
        action = {}
        for joint, side in enumerate(("left", "right")):
            target = START.copy()
            target[joint] += np.radians(0.04 * (index + 1))
            action[side + "_ee_pose"] = matrix_pose(adapter.kin[side].fk(target, 0.8))
            action[side + "_ee_joint_state"] = np.array([0.8])
        actions.append(action)
    now = time.monotonic_ns()
    measured = RobotState({side + "_arm": np.r_[START, 0.8] for side in ("left", "right")}, now, 1)

    def submit(seq):
        # No prepare_request: neither inline nor child decoding may need the anchors.
        context = ActionContext(seq, now, now, execution_time_ns=now, measured_state=measured)
        response = InferenceResponse(
            "test", seq, now, 1.0, {"actions": actions}, observation_time_ns=now
        )
        decoder.submit(response, context, time.monotonic_ns() + 30_000_000_000)
        return context

    decoder = ActionDecoderClient(config["robot"], config["policy"], adapter)
    try:
        decoder.start()
        context = submit(1)
        inline = adapter.decode_action({"actions": actions}, context)
        result = wait_for_decode(decoder)
        assert result.error is None
        chunk = result.chunk
        assert chunk.metadata["decode_mode"] == "process"
        assert set(chunk.metadata["decode_partition_ms"]) == {"left_arm", "right_arm"}
        assert chunk.source_offset_steps == inline.source_offset_steps
        assert chunk.observation_time_ns == inline.observation_time_ns
        for side in ("left", "right"):
            np.testing.assert_allclose(
                chunk.groups[side + "_arm"], inline.groups[side + "_arm"], atol=1e-9, rtol=0
            )
            assert chunk.metadata["diff_ik_lag"][side] == pytest.approx(
                inline.metadata["diff_ik_lag"][side], abs=1e-9
            )
        # A failure in one arm's process still rejects the whole chunk.
        actions[-1]["right_ee_pose"][0] += 0.3
        submit(2)
        failed = wait_for_decode(decoder)
        assert failed.chunk is None
        assert "rejecting the entire chunk" in failed.error
    finally:
        decoder.close()
