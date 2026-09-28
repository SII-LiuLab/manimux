from __future__ import annotations

import unittest
from pathlib import Path
import time

import numpy as np

from manimux.embodiments.asyncsim import AsyncSimRobot, AsyncSimSensor
from manimux.cli import load_config
from manimux.policies import build_policy_model
from manimux.policies.capabilities import PolicyCapabilities
from manimux.policies.xpolicylab.client import XPolicyLabWsPolicyModel
from manimux.runtime.asyncsim import AsyncSimRuntime
from manimux.types import ActionChunk, InferenceResponse, RobotCommand


class FakeAsyncSim:
    def __init__(self, clock_ns=lambda: 1_000_000_000) -> None:
        self.episode_id = None
        self.commands = []
        self.streams = []
        self.closed = False
        self.sim_ts = 0.0
        self.clock_ns = clock_ns

    def connect(self) -> None:
        self.closed = False

    def reset(self, *, seed=None):
        self.episode_id = "episode-fake"
        self.sim_ts = 0.0
        return {"episode_id": self.episode_id, "seed": seed}

    def subscribe(self, streams):
        self.streams = streams

    def read_snapshot(self, streams):
        assert streams == self.streams
        packets = {
            "proprio.joint_state": {
                "packet": {"episode_id": self.episode_id, "seq": 3,
                           "capture_ts": self.sim_ts, "payload": {0: {
                               "left_joints": [0.0, 0.0], "right_joints": [0.0, 0.0],
                           }}}
            },
            "camera.head.rgb": {
                "packet": {"episode_id": self.episode_id, "seq": 1,
                           "capture_ts": self.sim_ts, "payload": {0: np.zeros((2, 3, 3), dtype=np.uint8)}}
            },
        }
        return {"episode_id": self.episode_id, "sim_ts": self.sim_ts,
                "clock_anchor": {"sim_ts": self.sim_ts, "monotonic_ns": self.clock_ns()},
                "packets": {name: packets[name] for name in streams}}

    def health(self):
        return {"episode": {"sim_ts": self.sim_ts, "state": "running"},
                "canonical_group_dims": {"left_arm": 2, "right_arm": 2}}

    def submit_command(self, command):
        self.commands.append(command)
        return {"accepted": True, "reason": "accepted"}

    def result(self):
        return {"episode_id": self.episode_id, "result": {"action_count": len(self.commands)}}

    def close(self):
        self.closed = True


class FixedClock:
    def __init__(self):
        self.time_ns = 1_000_000_000

    def now_ns(self):
        return self.time_ns

    def sleep_until_ns(self, target_ns):
        self.time_ns = target_ns


class FakeWorker:
    def __init__(self, clock, capabilities=None):
        self.clock = clock
        self.capabilities = capabilities or PolicyCapabilities()
        self.is_alive = False
        self.submitted = []
        self._pending = None

    def start(self):
        self.is_alive = True

    def submit_latest(self, request):
        self.submitted.append(request)
        self._pending = request

    def poll(self):
        request, self._pending = self._pending, None
        if request is None:
            return None
        actions = {name: np.ones((4, 2)) for name in request.observation.state.groups}
        return InferenceResponse(
            request.session_id, request.request_seq, self.clock.now_ns(), 1.0,
            {"format": "joint", "actions": actions,
             "action_semantics": "absolute_joint_position"},
            observation_time_ns=request.observation_time_ns,
        )

    def close(self):
        self.is_alive = False


class StaleWorker(FakeWorker):
    def poll(self):
        response = super().poll()
        if response is not None:
            response.session_id = "another-session"
        return response


def config():
    source = Path(__file__).resolve().parents[2] / "manimux/configs/experiments/asyncsim/fake_joint.yaml"
    return load_config(source)


class AsyncSimRuntimeTests(unittest.TestCase):
    def test_act_aggregation_uses_manimux_strategy(self):
        backend = FakeAsyncSim()
        clock = FixedClock()
        configured = config()
        configured["inference"]["algorithm"] = "act_temporal_ensemble"
        configured["policy"]["adapter"]["type"] = "manimux.policy_adapter.joint:JointAdapter"
        runtime = AsyncSimRuntime(configured, client=backend, clock=clock, worker=FakeWorker(clock))
        result = runtime.run_policy(max_steps=4)
        self.assertEqual(runtime.strategy.name, "act_temporal_ensemble")
        self.assertGreaterEqual(result["accepted_plans"], 1, result)
        self.assertTrue(all(item["action_space"] == "joint_position" for item in backend.commands))

    def test_smooth_executor_filters_canonical_command(self):
        backend = FakeAsyncSim()
        configured = config()
        configured["executor"]["type"] = "smooth"
        runtime = AsyncSimRuntime(configured, client=backend, clock=FixedClock())
        runtime.start()
        runtime.step()
        chunk = ActionChunk("plan-smooth", 0, runtime.clock.now_ns(), runtime.clock.now_ns(),
                            "joint_position", 20_000_000, {
                                "left_arm": np.ones((3, 2)),
                                "right_arm": np.ones((3, 2)),
                            })
        _, result, _ = runtime.step(chunk)
        self.assertTrue(result.accepted)
        self.assertGreater(backend.commands[-1]["groups"]["left_arm"][0], 0.0)
        self.assertLess(backend.commands[-1]["groups"]["left_arm"][0], 1.0)
        runtime.close()

    def test_xpolicylab_template_loads_client_without_network(self):
        source = Path(__file__).resolve().parents[2] / (
            "manimux/configs/experiments/asyncsim/xpolicylab_joint_template.yaml"
        )
        configured = load_config(source)
        self.assertEqual(configured["robot"]["group_dims"], {"left_arm": 7, "right_arm": 7})
        self.assertEqual(configured["robot"]["options"]["state_stream"], "proprio.canonical_joint_state")
        backend = build_policy_model(configured["policy"])
        self.assertIsInstance(backend, XPolicyLabWsPolicyModel)
        backend.close()

    def test_stale_policy_response_is_rejected(self):
        backend = FakeAsyncSim()
        clock = FixedClock()
        worker = StaleWorker(clock)
        runtime = AsyncSimRuntime(config(), client=backend, clock=clock, worker=worker)
        result = runtime.run_policy(max_steps=3)
        self.assertEqual(result["accepted_plans"], 0)
        self.assertIn("stale_session", result["rejection_reasons"])

    def test_policy_worker_adapter_and_timeline_pipeline(self):
        backend = FakeAsyncSim()
        clock = FixedClock()
        worker = FakeWorker(clock)
        configured = config()
        configured["policy"]["adapter"]["type"] = "manimux.policy_adapter.joint:JointAdapter"
        runtime = AsyncSimRuntime(configured, client=backend, clock=clock, worker=worker)
        result = runtime.run_policy(max_steps=3)
        self.assertGreaterEqual(result["accepted_plans"], 1, result)
        self.assertTrue(worker.submitted)
        self.assertEqual(worker.submitted[0].observation.frames["head"].sequence, 1)
        self.assertTrue(any(command["plan_id"].startswith("joint-") for command in backend.commands))
        self.assertEqual(len(backend.commands), 3)
        self.assertEqual(len(result["command_acks"]), 3)
        self.assertEqual(result["asyncsim_result"]["result"]["action_count"], 3)
        self.assertTrue(backend.closed)

    def test_real_fake_policy_worker_process(self):
        backend = FakeAsyncSim(time.monotonic_ns)
        configured = config()
        configured["policy"]["timeout_s"] = 3.0
        configured["inference"]["max_plan_age_s"] = 3.0
        runtime = AsyncSimRuntime(configured, client=backend)
        result = runtime.run_policy(max_steps=20)
        self.assertGreaterEqual(result["accepted_plans"], 1, result)
        self.assertTrue(any(item["plan_id"].startswith("plan-") for item in backend.commands))

    def test_policy_backend_identity_mismatch_aborts(self):
        backend = FakeAsyncSim()
        clock = FixedClock()
        configured = config()
        configured["policy"]["expected_backend"] = {"server": "expected"}
        worker = FakeWorker(clock, PolicyCapabilities(backend_metadata={"server": "other"}))
        runtime = AsyncSimRuntime(configured, client=backend, clock=clock, worker=worker)
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            runtime.run_policy(max_steps=2)
        self.assertFalse(worker.submitted)
        self.assertTrue(backend.closed)

    def test_fake_backend_uses_one_snapshot_and_one_command_per_tick(self):
        backend = FakeAsyncSim()
        runtime = AsyncSimRuntime(config(), client=backend, clock=FixedClock())
        runtime.start(seed=3)
        observation, result, ack = runtime.step()
        self.assertIsNone(result)
        self.assertTrue(ack["accepted"])
        self.assertEqual(observation.state.sequence, 3)
        self.assertEqual(observation.frames["head"].data.shape, (2, 3, 3))
        self.assertEqual(backend.commands[0]["plan_id"], "hold")
        self.assertEqual(backend.streams, ["proprio.joint_state", "camera.head.rgb"])
        chunk = ActionChunk("plan-1", 0, runtime.clock.now_ns(), runtime.clock.now_ns(),
                            "joint_position", 20_000_000, {
                                "left_arm": np.ones((3, 2)),
                                "right_arm": np.ones((3, 2)),
                            })
        _, result, _ = runtime.step(chunk)
        self.assertTrue(result.accepted)
        self.assertEqual(backend.commands[-1]["groups"]["left_arm"], [1.0, 1.0])
        self.assertEqual([item["command_seq"] for item in backend.commands], [1, 2])
        runtime.close()
        self.assertTrue(backend.closed)
        runtime.start(seed=3)
        runtime.step()
        self.assertEqual(backend.commands[-1]["command_seq"], 1)
        runtime.close()

    def test_sensor_rejects_non_uint8_and_robot_rejects_wrong_layout(self):
        backend = FakeAsyncSim()
        backend.reset()
        backend.subscribe(["proprio.joint_state", "camera.head.rgb"])
        snapshot = backend.read_snapshot(["proprio.joint_state", "camera.head.rgb"])
        sensor = AsyncSimSensor(backend, {"head": "camera.head.rgb"})
        sensor.start()
        snapshot["packets"]["camera.head.rgb"]["packet"]["payload"][0] = np.ones((2, 2, 3))
        with self.assertRaises(ValueError):
            sensor.use_snapshot(snapshot)
        robot = AsyncSimRobot(backend, {"left_arm": 2}, {"left_arm": "left_joints"})
        robot.use_snapshot(snapshot)
        with self.assertRaises(ValueError):
            robot.send_command(RobotCommand({"left_arm": [1.0]}, 1_000_000_000, "bad"))


if __name__ == "__main__":
    unittest.main()
