"""Hardware-free tests with real IK processes and RTC's shared control loop."""

import json
import math
import time

import numpy as np
import pytest
import zarr

from manimux.config import ManiMuxConfig, load_config
from manimux.integrations.umi_dp_tianji.ik_config import bind_diff_ik_profile
from manimux.integrations.umi_dp_tianji.policy_plugin import UmiDpTianjiAdapter, matrix_pose
from manimux.policies.decoder import ActionDecoderClient
from manimux.policies.fake import FakePolicyAdapter
from manimux.runtime import build_runtime
from manimux.types import ActionChunk, ActionContext, InferenceResponse, RobotState
from manimux.viewer import ViewerControl


@pytest.mark.parametrize("backend", ["analytic", "diff"])
@pytest.mark.parametrize("horizon", [16, 64])
def test_real_tianji_parallel_ik_matches_serial_and_rejects_whole_chunk(backend, horizon):
    config = load_config("configs/umi_dp/tianji/infra/pass_ball/rtc.yaml")
    config.robot.driver = "mock_dual_arm"
    config.policy.horizon_steps = horizon
    config.policy.options["ik_backend"] = backend
    bind_diff_ik_profile(config)
    adapter = UmiDpTianjiAdapter(config.robot, config.policy)
    joints = np.radians([50, -40, -30, -100, -65, 0, 40])
    now = time.monotonic_ns()
    state = RobotState({name: np.r_[joints, 0.8] for name in config.robot.group_dims}, now, 1)
    actions = []
    for row in range(horizon):
        target = joints.copy()
        target[0] += 0.0001 * row
        step = {}
        for side in ("left", "right"):
            step[f"{side}_ee_pose"] = matrix_pose(adapter.kin[side].fk(target, 0.8))
            step[f"{side}_ee_joint_state"] = np.array([0.8])
        actions.append(step)
    context = ActionContext(1, now, now, now + adapter.offset_ns + 2 * adapter.dt_ns, state)
    serial = adapter.decode_action(actions, context)
    assert serial.source_offset_steps == 2
    decoder = ActionDecoderClient(config.robot, config.policy, adapter)
    try:
        decoder.start()

        def decode(seq):
            response = InferenceResponse("test", seq, now, 0, actions, observation_time_ns=now)
            ctx = ActionContext(seq, now, now, context.execution_time_ns, state)
            deadline = time.monotonic_ns() + 5_000_000_000
            decoder.submit(response, ctx, deadline)
            result = None
            while result is None and time.monotonic_ns() < deadline:
                result = decoder.poll()
                time.sleep(0.001)
            assert result is not None
            return result

        result = decode(1)
        assert result.error is None
        assert result.chunk.source_offset_steps == 2
        assert result.chunk.horizon_steps == horizon - 2
        for name in config.robot.group_dims:
            np.testing.assert_allclose(result.chunk.groups[name], serial.groups[name], atol=1e-9)
        assert set(result.chunk.metadata["decode_partition_ms"]) == set(config.robot.group_dims)
        actions[3]["right_ee_joint_state"] = np.array([1.1])
        failed = decode(2)
        assert failed.chunk is None and "gripper" in failed.error
    finally:
        decoder.close()
    assert all(not process.is_alive() for process in decoder._processes)


class SlowPartitionAdapter(FakePolicyAdapter):
    supports_context_only_decode = True
    decode_partitions = ("left_arm", "right_arm")

    def decode_action_partition(self, raw, context, partition):
        until = time.monotonic() + (0.08 if partition == "left_arm" else 0.13)
        while time.monotonic() < until:
            pass
        skip = max(0, (context.execution_time_ns - raw.observation_time_ns) // raw.dt_ns)
        return ActionChunk(
            raw.plan_id,
            raw.request_seq,
            raw.observation_time_ns,
            raw.created_time_ns,
            raw.action_space,
            raw.dt_ns,
            {partition: raw.groups[partition][skip:]},
            source_offset_steps=skip,
        )


def build_slow_adapter(robot, policy):
    return SlowPartitionAdapter()


def rtc_config():
    data = load_config("configs/mock.yaml").model_dump(exclude_unset=True)
    data["execution"].pop("refill_threshold_s")
    data["execution"].update(runtime="rtc", commit_lead_s=0.0)
    data["robot"].update(control_hz=250, group_dims={"left_arm": 6, "right_arm": 6})
    data["policy"].update(
        adapter=f"{__name__}:build_slow_adapter",
        action_decoding="process",
        action_dt_s=1 / 30,
        horizon_steps=64,
        timeout_s=3,
        inference_delay_s=0.02,
    )
    data["sensors"] = []
    data["run"]["max_steps"] = 1400
    return ManiMuxConfig.model_validate(data)


@pytest.mark.parametrize("home", [False, True])
def test_rtc_process_loop_conditions_and_resumes_after_inflight_pause(tmp_path, monkeypatch, home):
    config = rtc_config()
    runtime = build_runtime(config, tmp_path)
    pause = {}

    def control():
        if not pause and runtime._timeline.accepted_request_seq >= 1 and runtime._decoder.busy:
            pause.update(
                seq=runtime._decoder._pending.request_seq,
                start=time.monotonic_ns(),
                until=time.monotonic() + 0.12,
            )
            if home:
                return ViewerControl(paused=True, home_requested=True)
        return ViewerControl(paused=bool(pause) and time.monotonic() < pause["until"])

    monkeypatch.setattr(runtime._viewer, "poll_control", control)
    result = runtime.run()
    events = [
        json.loads(line) for line in (result.episode_dir / "events.jsonl").read_text().splitlines()
    ]
    submissions = [e for e in events if e["kind"] == "inference_submitted"]
    accepted = [e for e in events if e["kind"] == "plan_accepted"]
    assert pause and len(accepted) >= 3
    assert all(e["request_seq"] != pause["seq"] for e in accepted)
    resumed = [e for e in submissions if e["request_seq"] > pause["seq"]]
    assert resumed and not resumed[0]["conditioned"]
    assert any(e["conditioned"] for e in resumed[1:])
    assert not [e for e in events if e["kind"] == "rtc_delay_infeasible"]
    for event in accepted:
        assert event["rtc_source_horizon"] == 64
        assert event["measured_delay"] >= math.ceil(0.13 / config.policy.effective_action_dt_s)
        assert event["request_to_commit_ms"] >= event["decode_stage_ms"]
    ticks = zarr.open(str(result.episode_dir / "data.zarr"), mode="r")["ticks/monotonic_ns"][:]
    decodes = [e for e in events if e["kind"] == "decode_submitted"]
    # CPU-bound solves execute in the two children while the main loop ticks.
    assert all(
        np.count_nonzero((ticks > e["seed_time_ns"]) & (ticks < e["seed_time_ns"] + 70_000_000))
        >= 5
        for e in decodes
        if e["request_seq"] in {a["request_seq"] for a in accepted}
    )
    inflight = None
    for event in events:
        if event["kind"] == "inference_submitted":
            assert inflight is None
            inflight = event["request_seq"]
        elif event["kind"] in {"plan_accepted", "plan_rejected", "inference_rejected"}:
            assert event["request_seq"] == inflight
            inflight = None
    assert all(not process.is_alive() for process in runtime._decoder._processes)


def test_rtc_decoder_timeout_closes_children_and_mock_robot(tmp_path):
    config = rtc_config()
    config.policy.timeout_s = 0.08
    config.policy.inference_delay_s = 0.001
    runtime = build_runtime(config, tmp_path)
    with pytest.raises(TimeoutError, match="decoder exceeded"):
        runtime.run()
    assert not runtime._robot._connected
    assert all(not process.is_alive() for process in runtime._decoder._processes)
