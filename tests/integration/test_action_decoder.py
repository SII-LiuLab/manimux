from __future__ import annotations

import json
import time

import numpy as np
import pytest
import zarr

from manimux.config import load_config
from manimux.policies.fake import FakePolicyAdapter
from manimux.runtime.edge import EdgeRuntime
from manimux.viewer import ViewerControl


class SlowAdapter(FakePolicyAdapter):
    supports_context_only_decode = True

    def decode_action(self, raw, context):
        # CPU work, not a sleep that releases the GIL: the process boundary is
        # essential to isolate Python-heavy numerical iterations from control.
        until = time.monotonic() + 0.25
        while time.monotonic() < until:
            pass
        return super().decode_action(raw, context)


def build_slow_adapter(robot, policy):
    return SlowAdapter()


@pytest.mark.parametrize("expired", [False, True])
def test_process_decoding_keeps_control_ticking_and_checks_actual_expiry(tmp_path, expired):
    config = load_config("configs/mock.yaml")
    config.policy.adapter = f"{__name__}:build_slow_adapter"
    config.policy.action_decoding = "process"
    config.policy.horizon_steps = 6 if expired else 20
    config.policy.inference_delay_s = 0.01
    config.execution.executor = "direct"
    config.execution.inference_schedule = "single_inflight"
    config.run.max_steps = 100
    runtime = EdgeRuntime(config, tmp_path)
    result = runtime.run()
    events = [json.loads(s) for s in (result.episode_dir / "events.jsonl").read_text().splitlines()]
    sent = [e for e in events if e["kind"] == "decode_submitted"]
    assert sent
    ticks = zarr.open(str(result.episode_dir / "data.zarr"), mode="r")["ticks/monotonic_ns"][:]
    # Commands continue while the first 250 ms CPU-bound decode is still active.
    assert (
        np.count_nonzero(
            (ticks > sent[0]["seed_time_ns"]) & (ticks < sent[0]["seed_time_ns"] + 220_000_000)
        )
        >= 8
    )
    assert not runtime._decoder.busy
    assert all(not p.is_alive() for p in runtime._decoder._processes)
    if expired:
        assert result.accepted_plans == 0
        assert any(e.get("reason") == "no_future_horizon" for e in events)
    else:
        accepted = [e for e in events if e["kind"] == "plan_accepted"]
        assert accepted
        assert accepted[0]["decode_ms"] >= 240
        assert accepted[0]["observation_to_commit_ms"] >= 240
        boundary = next(e for e in events if e["kind"] == "plan_boundary")
        assert boundary["trimmed_steps"] >= 4


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
def test_pause_discards_pending_decode_and_resume_needs_fresh_observation(tmp_path, home):
    config = load_config("configs/mock.yaml")
    config.policy.adapter = f"{__name__}:build_slow_adapter"
    config.policy.action_decoding = "process"
    config.policy.inference_delay_s = 0.01
    config.execution.inference_schedule = "single_inflight"
    config.run.max_steps = 120
    runtime = EdgeRuntime(config, tmp_path)
    runtime._viewer = PauseDuringDecode(runtime, home=home)
    result = runtime.run()
    events = [json.loads(s) for s in (result.episode_dir / "events.jsonl").read_text().splitlines()]
    assert runtime._viewer.interrupted
    assert any(e["kind"] == "inference_rejected" and e["request_seq"] == 1 for e in events)
    accepted = [e for e in events if e["kind"] == "plan_accepted"]
    assert accepted and all(e["request_seq"] > 1 for e in accepted)


def test_decode_deadline_failure_closes_robot_and_children(tmp_path):
    config = load_config("configs/mock.yaml")
    config.policy.adapter = f"{__name__}:build_slow_adapter"
    config.policy.action_decoding = "process"
    config.policy.inference_delay_s = 0.001
    config.policy.timeout_s = 0.12
    config.execution.inference_schedule = "single_inflight"
    config.run.max_steps = 100
    runtime = EdgeRuntime(config, tmp_path)
    with pytest.raises(TimeoutError, match="decoder exceeded"):
        runtime.run()
    assert not runtime._robot._connected
    assert all(not p.is_alive() for p in runtime._decoder._processes)


def test_parallel_sapolicy_ik_matches_serial_including_failed_waypoints():
    pytest.importorskip("mujoco")
    pytest.importorskip("i2rt")
    from manimux.integrations.sapolicy_yam.policy_plugin import (
        SAPolicyYamAdapter,
        _pose_to_wire_endpose,
    )
    from manimux.policies.decoder import ActionDecoderClient
    from manimux.types import ActionContext, InferenceResponse, RobotState

    cfg = load_config("configs/sapolicy/yam/infra/manimux-xpl.yaml")
    adapter = SAPolicyYamAdapter(cfg.robot, cfg.policy)
    groups = {name: np.array([0.1, 0.8, 1.0, -0.2, 0.1, 0.2, 0.5]) for name in cfg.robot.group_dims}
    wire = np.empty((16, 16))
    for j, name in enumerate(groups):
        for i in range(16):
            q = groups[name].copy()
            q[0] += 0.002 * i
            pose = adapter._model_from_kinematics[name] @ adapter._kinematics.fk(q[:6], q[6])
            wire[i, j * 8 : j * 8 + 7] = _pose_to_wire_endpose(pose)
            wire[i, j * 8 + 7] = 0.5
    # Repeated unreachable targets must preserve the same preceding valid seed.
    wire[-2:, 0] += 10
    wire[-2:, 8] += 10
    decoder = ActionDecoderClient(cfg.robot, cfg.policy, adapter)
    try:
        decoder.start()
        now = time.monotonic_ns()
        context = ActionContext(1, now, now, measured_state=RobotState(groups, now, 1))
        serial = adapter.decode_action(wire, context)
        response = InferenceResponse("test", 1, now, 1.0, wire, observation_time_ns=now)
        decoder.submit(response, context, time.monotonic_ns() + 10_000_000_000)
        parallel = None
        until = time.monotonic() + 10
        while parallel is None and time.monotonic() < until:
            parallel = decoder.poll()
            time.sleep(0.001)
        assert parallel is not None and parallel.error is None
        for name in groups:
            np.testing.assert_allclose(
                parallel.chunk.groups[name], serial.groups[name], atol=1e-10, rtol=0
            )
            assert parallel.chunk.metadata["ik"][name]["failed_steps"] == 2
            assert (
                parallel.chunk.metadata["ik"][name]["converged"]
                == serial.metadata["ik"][name]["converged"]
            )
        assert set(parallel.chunk.groups) == set(groups)
        assert set(parallel.chunk.metadata["raw_model_eef"]) == set(groups)
        # A malformed partition cannot produce a half-arm plan.
        invalid = wire.copy()
        invalid[0, 3:7] = 0
        response.request_seq = 2
        response.raw_action = invalid
        context = ActionContext(2, now, now, measured_state=RobotState(groups, now, 1))
        decoder.submit(response, context, time.monotonic_ns() + 10_000_000_000)
        failed = None
        until = time.monotonic() + 10
        while failed is None and time.monotonic() < until:
            failed = decoder.poll()
            time.sleep(0.001)
        assert failed is not None and failed.error is not None and failed.chunk is None
    finally:
        decoder.close()


class PartitionAdapter:
    supports_context_only_decode = True
    supports_independent_group_decode = True
    decode_partitions = ('left_arm', 'right_arm')

    def validate(self, robot, policy):
        pass

    def build_observation(self, snapshot):
        return snapshot

    def decode_action_partition(self, raw, context, partition):
        from manimux.types import ActionChunk
        if partition == 'right_arm' and context.request_seq == 1:
            time.sleep(.4)
        return ActionChunk(str(context.request_seq), context.request_seq,
                           context.observation_time_ns, context.created_time_ns,
                           'joint_position', 33_333_333, {partition: np.ones((25, 2))})

    def decode_hold_partition(self, raw, context, partition, reason):
        from manimux.types import ActionChunk
        return ActionChunk(str(context.request_seq), context.request_seq,
                           context.observation_time_ns, context.created_time_ns,
                           'joint_position', 33_333_333,
                           {partition: np.tile(context.measured_state.groups[partition], (25, 1))},
                           hold_from_step={partition: 0}, metadata={'holds': {partition: reason}})


def build_partition_adapter(robot, policy):
    return PartitionAdapter()


def test_late_arm_holds_without_blocking_other_arm_and_late_result_cannot_replace_new_plan():
    from manimux.policies.decoder import ActionDecoderClient
    from manimux.types import ActionContext, InferenceResponse, RobotState
    config = load_config('configs/mock.yaml')
    config.policy.adapter = f'{__name__}:build_partition_adapter'
    decoder = ActionDecoderClient(config.robot, config.policy, PartitionAdapter())
    try:
        decoder.start()
        def decode(seq):
            now = time.monotonic_ns()
            context = ActionContext(seq, now, now,
                measured_state=RobotState({name: np.zeros(2) for name in PartitionAdapter.decode_partitions}, now, 0),
                independent_groups=True, decode_budget_ms=40)
            decoder.submit(InferenceResponse('test', seq, now, 0, None, observation_time_ns=now),
                           context, now + 3_000_000_000)
            result = None
            while result is None and time.monotonic_ns() < now + 1_000_000_000:
                result = decoder.poll()
                time.sleep(.001)
            assert result is not None and result.error is None
            return result.chunk, (time.monotonic_ns() - now) / 1e9
        first, duration = decode(1)
        assert duration < .3  # Does not wait for the 400ms right-arm job.
        assert first.hold_from_step == {'right_arm': 0}
        np.testing.assert_array_equal(first.groups['left_arm'], 1)
        second, _ = decode(2)
        assert second.metadata['holds']['right_arm'] == 'worker_busy'
        time.sleep(.45)
        assert decoder.poll() is None  # Drain the old result while no request is pending.
        third, _ = decode(3)
        assert third.request_seq == 3 and third.hold_from_step == {}
    finally:
        decoder.close()


def test_independent_runtime_executes_left_while_right_worker_times_out(tmp_path):
    from manimux.config import GripperHysteresisConfig
    config = load_config('configs/mock.yaml')
    config.robot.group_dims = {'left_arm': 2, 'right_arm': 2}
    config.policy.adapter = f'{__name__}:build_partition_adapter'
    config.policy.action_decoding = 'process'
    config.policy.horizon_steps = 25
    config.execution.max_chunk_steps = 25
    config.execution.independent_group_decoding = True
    config.execution.decode_budget_ms = 40
    config.execution.inference_schedule = 'single_inflight'
    config.execution.smooth.tracking_mode = 'braking'
    config.execution.smooth.max_velocity = .8
    config.execution.smooth.max_acceleration = 3
    config.execution.smooth.gripper = GripperHysteresisConfig(
        mode='continuous', group_indices={'left_arm': 1, 'right_arm': 1},
        max_velocity=1, max_acceleration=12)
    config.run.max_steps = 110
    runtime = EdgeRuntime(config, tmp_path)
    result = runtime.run()
    assert result.accepted_plans >= 2
    events = [json.loads(s) for s in (result.episode_dir/'events.jsonl').read_text().splitlines()]
    accepted = [e for e in events if e['kind'] == 'plan_accepted']
    assert accepted[0]['hold_from_step'] == {'right_arm': 0}
    assert any(e['hold_from_step'] == {} for e in accepted[1:])
    z = zarr.open_group(str(result.episode_dir/'data.zarr'), mode='r')
    ticks = z['ticks/monotonic_ns'][:]
    # Verify recorded commands after publication and before the late right result.
    submitted = next(e['seed_time_ns'] for e in events if e['kind']=='decode_submitted')
    mask = (ticks > submitted + 120_000_000) & (ticks < submitted + 350_000_000)
    assert mask.sum() >= 10
    left = z['ticks/command/left_arm'][:][mask]
    right = z['ticks/command/right_arm'][:][mask]
    assert left[-1, 0] > left[0, 0] + .03
    np.testing.assert_allclose(right, 0, atol=0, rtol=0)
