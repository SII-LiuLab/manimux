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
from manimux.runtime.dvac import DvacInferenceRequest
from manimux.runtime.paint import PaintInferenceRequest
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.types import ActionChunk, InferenceResponse, RobotCommand


class FakeAsyncSim:
    def __init__(self, clock_ns=lambda: 1_000_000_000) -> None:
        self.episode_id = None
        self.commands = []
        self.streams = []
        self.closed = False
        self.sim_ts = 0.0
        self.clock_ns = clock_ns
        self.instruction = "Pick up the key and insert it."

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
            "language.instruction": {
                "packet": {"episode_id": self.episode_id, "seq": 1,
                           "capture_ts": self.sim_ts, "payload": {0: self.instruction}}
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
        return {"accepted": True, "reason": "accepted", "plan_id": command["plan_id"],
                "command_seq": command["command_seq"]}

    def result(self):
        return {"episode_id": self.episode_id, "sim_ts": self.sim_ts, "state": "running",
                "result": {"action_count": len(self.commands), "success_rate": 0.0, "score": 0.0}}

    def metrics(self):
        return {"episode_id": self.episode_id, "sim_ts": self.sim_ts,
                "runtime": {"ticks": len(self.commands)}}

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


class DelayedWorker(FakeWorker):
    def __init__(self, clock):
        super().__init__(clock)
        self._wait = 0

    def submit_latest(self, request):
        super().submit_latest(request)
        self._wait = 2

    def poll(self):
        if self._pending is not None and self._wait:
            self._wait -= 1
            return None
        return super().poll()


class SamplingWorker(FakeWorker):
    def __init__(self, clock, mode):
        super().__init__(clock, PolicyCapabilities(sampling_modes=frozenset({mode})))
        self.mode = mode

    def poll(self):
        response = super().poll()
        if response is not None and self.mode == "dvac":
            response.raw_action["dvac"] = {"execution_steps": 2}
        return response


def config():
    source = Path(__file__).resolve().parents[2] / "manimux/configs/experiments/asyncsim/fake_joint.yaml"
    return load_config(source)


class AsyncSimRuntimeTests(unittest.TestCase):
    def test_terminal_episode_rejection_preserves_result(self):
        class TerminatingAsyncSim(FakeAsyncSim):
            state = "running"

            def health(self):
                health = super().health()
                health["episode"]["state"] = self.state
                return health

            def submit_command(self, command):
                if self.commands:
                    self.state = "failed"
                    return {"accepted": False, "reason": "episode_not_accepting_commands"}
                return super().submit_command(command)

            def result(self):
                result = super().result()
                result["state"] = self.state
                return result

        backend = TerminatingAsyncSim()
        clock = FixedClock()
        configured = config()
        configured["policy"]["adapter"]["type"] = "manimux.policy_adapter.joint:JointAdapter"
        run = AsyncSimRuntime(configured, client=backend, clock=clock, worker=FakeWorker(clock)).run_policy(max_steps=4)
        self.assertEqual(run["asyncsim_result"]["state"], "failed")
        self.assertEqual(len(run["command_acks"]), 1)
        self.assertTrue(any(event["kind"] == "command_rejected" for event in run["audit"]["manimux_events"]))

    def test_delayed_initial_camera_waits_before_inference(self):
        class DelayedCamera(FakeAsyncSim):
            reads = 0

            def read_snapshot(self, streams):
                self.reads += 1
                snapshot = super().read_snapshot(streams)
                if self.reads <= 2:
                    snapshot["packets"]["camera.head.rgb"]["packet"] = None
                return snapshot

        backend = DelayedCamera()
        clock = FixedClock()
        worker = FakeWorker(clock)
        configured = config()
        configured["policy"]["adapter"]["type"] = "manimux.policy_adapter.joint:JointAdapter"
        result = AsyncSimRuntime(configured, client=backend, clock=clock, worker=worker).run_policy(max_steps=2)
        self.assertGreaterEqual(backend.reads, 4)
        self.assertGreaterEqual(result["accepted_plans"], 1)
        self.assertEqual(len(worker.submitted[0].observation.frames), 1)

    def test_slow_simulation_retains_chunk_with_wall_clock_deadlines(self):
        class SlowClock(FixedClock):
            def sleep_until_ns(self, target_ns):
                self.time_ns = max(self.time_ns, target_ns)

        class SlowAsyncSim(FakeAsyncSim):
            def __init__(self, clock):
                super().__init__(clock_ns=clock.now_ns)
                self.clock = clock

            def read_snapshot(self, streams):
                self.clock.time_ns += 200_000_000
                self.sim_ts += 0.02
                return super().read_snapshot(streams)

        clock = SlowClock()
        backend = SlowAsyncSim(clock)
        worker = FakeWorker(clock)
        configured = config()
        configured["run"]["simulation_time_timeline"] = True
        configured["inference"]["inference_schedule"] = "single_inflight"
        configured["inference"]["refill_threshold_s"] = 0.02
        configured["policy"]["adapter"]["type"] = "manimux.policy_adapter.joint:JointAdapter"
        run = AsyncSimRuntime(configured, client=backend, clock=clock, worker=worker).run_policy(max_steps=8)
        planned = [item["plan_id"] for item in backend.commands if item["plan_id"].startswith("joint-")]
        self.assertGreater(len(planned), len(set(planned)), planned)
        self.assertGreater(run["accepted_plans"], 1)
        self.assertTrue(all(request.deadline_ns > request.observation_time_ns for request in worker.submitted))
        events = run["audit"]["manimux_events"]
        overlap = False
        for index, event in enumerate(events):
            if event["kind"] != "inference_submitted" or event["request_seq"] <= 1:
                continue
            response_index = next((offset for offset in range(index + 1, len(events))
                                   if events[offset]["kind"] == "inference_response"
                                   and events[offset]["request_seq"] == event["request_seq"]), None)
            if response_index is None:
                continue
            overlap |= any(item["kind"] == "command_ack" and item["plan_id"].startswith("joint-")
                           for item in events[index + 1:response_index])
        self.assertTrue(overlap)

    def test_specialized_strategies_use_their_requests_and_canonical_timeline(self):
        for mode, request_type in (
            ("rtc", RtcInferenceRequest),
            ("paint", PaintInferenceRequest),
            ("dvac", DvacInferenceRequest),
        ):
            with self.subTest(mode=mode):
                configured = config()
                configured["inference"]["algorithm"] = mode
                configured["policy"]["adapter"]["type"] = "manimux.policy_adapter.joint:JointAdapter"
                configured["inference"]["rtc"].update(
                    initial_delay_policy_steps=1, min_execute_policy_steps=1,
                )
                configured["inference"]["paint"].update(
                    initial_delay_policy_steps=1, execution_policy_steps=1,
                )
                backend = FakeAsyncSim()
                clock = FixedClock()
                worker = SamplingWorker(clock, mode)
                result = AsyncSimRuntime(configured, client=backend, clock=clock, worker=worker).run_policy(max_steps=8)
                self.assertGreaterEqual(result["accepted_plans"], 1, result)
                if mode == "paint":
                    self.assertNotIsInstance(worker.submitted[0], PaintInferenceRequest)
                    self.assertTrue(any(isinstance(item, request_type) for item in worker.submitted[1:]))
                else:
                    self.assertIsInstance(worker.submitted[0], request_type)
                self.assertTrue(all(item["action_space"] == "joint_position" for item in backend.commands))
                if mode == "dvac":
                    self.assertTrue(any(item["plan_id"].startswith("joint-") for item in backend.commands))

    def test_sampling_mode_mismatch_fails_before_request(self):
        configured = config()
        configured["inference"]["algorithm"] = "rtc"
        worker = FakeWorker(FixedClock())
        runtime = AsyncSimRuntime(configured, client=FakeAsyncSim(), clock=worker.clock, worker=worker)
        with self.assertRaisesRegex(RuntimeError, "sampling modes"):
            runtime.run_policy(max_steps=2)
        self.assertFalse(worker.submitted)

    def test_process_decoding_remains_explicitly_unsupported(self):
        configured = config()
        configured["policy"]["action_decoding"] = "process"
        with self.assertRaisesRegex(ValueError, "inline"):
            AsyncSimRuntime(configured, client=FakeAsyncSim())

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

    def test_skip_elapsed_steps_reuses_one_chunk_for_multiple_commands(self):
        backend = FakeAsyncSim()
        clock = FixedClock()
        configured = config()
        configured["run"]["simulation_time_timeline"] = True
        configured["inference"]["action_start_mode"] = "skip_elapsed_steps"
        runtime = AsyncSimRuntime(configured, client=backend, clock=clock)
        runtime.start()
        try:
            runtime.observe()
            backend.sim_ts = 0.04
            observation = runtime.observe()
            groups = {name: np.repeat(np.arange(4)[:, None], 2, axis=1)
                      for name in ("left_arm", "right_arm")}
            chunk = ActionChunk(
                "plan-trimmed", 0, clock.now_ns(), clock.now_ns(),
                "joint_position", 20_000_000, groups,
            )
            result, _ = runtime.execute(observation, chunk)
            self.assertTrue(result.accepted)
            self.assertEqual(result.trimmed_steps, 2)
            self.assertEqual(backend.commands[-1]["groups"]["left_arm"], [2.0, 2.0])

            backend.sim_ts = 0.06
            observation = runtime.observe()
            runtime.execute(observation)
            self.assertEqual(backend.commands[-1]["groups"]["left_arm"], [3.0, 3.0])
            self.assertEqual([cmd["plan_id"] for cmd in backend.commands], ["plan-trimmed"] * 2)
        finally:
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

    def test_robodojo_pi05_experiments_match_official_joint_contract(self):
        base = Path(__file__).resolve().parents[2] / "manimux/configs/experiments/asyncsim"
        for task, steps in (("insert_key", 300), ("match_and_pick_from_conveyor", 700)):
            with self.subTest(task=task):
                configured = load_config(base / f"robodojo_pi05_{task}.yaml")
                self.assertEqual(configured["run"]["task"], task)
                self.assertEqual(configured["run"]["instruction_stream"], "language.instruction")
                self.assertEqual(configured["run"]["max_control_steps"], steps)
                self.assertEqual(configured["robot"]["control_hz"], 250)
                self.assertEqual(configured["policy"]["action_dt_s"], 0.004)
                self.assertEqual(configured["policy"]["horizon_policy_steps"], 50)
                self.assertEqual(configured["inference"]["inference_schedule"], "single_inflight")
                self.assertEqual(configured["inference"]["action_start_mode"], "skip_elapsed_steps")
                self.assertEqual(configured["policy_server"]["checkpoint_num"], 59999)
                self.assertEqual(configured["policy_server"]["task_name"], task)
                self.assertEqual(configured["policy"]["expected_backend"]["model"]["task_name"], task)
                self.assertEqual(configured["policy"]["expected_backend"]["model"]["repo_id"], "arx_x5_sim")
                backend = build_policy_model(configured["policy"])
                self.assertIsInstance(backend, XPolicyLabWsPolicyModel)
                backend.close()

    def test_instruction_stream_and_inflight_action_replacement(self):
        backend = FakeAsyncSim()
        clock = FixedClock()
        configured = config()
        configured["run"]["instruction_stream"] = "language.instruction"
        configured["policy"]["adapter"]["type"] = "manimux.policy_adapter.joint:JointAdapter"
        configured["inference"].update(inference_schedule="single_inflight", refill_threshold_s=0.08)
        worker = DelayedWorker(clock)
        result = AsyncSimRuntime(configured, client=backend, clock=clock, worker=worker).run_policy(max_steps=12)
        events = result["audit"]["manimux_events"]
        self.assertIn("language.instruction", backend.streams)
        self.assertEqual(worker.submitted[0].instruction, backend.instruction)
        commits = [item for item in events if item["kind"] == "plan_committed"]
        self.assertGreaterEqual(len(commits), 2)
        first, second = commits[:2]
        submitted = next(index for index, item in enumerate(events)
                         if item["kind"] == "inference_submitted" and item["request_seq"] == second["request_seq"])
        second_commit = events.index(second)
        self.assertTrue(any(item["kind"] == "command_ack" and item["plan_id"] == first["plan_id"]
                            for item in events[submitted:second_commit]))
        self.assertTrue(any(item["kind"] == "command_ack" and item["plan_id"] == second["plan_id"]
                            for item in events[second_commit:]))
        self.assertIn("issued_sim_ts", result["command_acks"][0])

    def test_missing_instruction_does_not_fall_back_to_task_id(self):
        configured = config()
        configured["run"]["instruction_stream"] = "language.instruction"
        backend = FakeAsyncSim()
        backend.instruction = ""
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            AsyncSimRuntime(configured, client=backend, clock=FixedClock()).run_policy(max_steps=2)

    def test_model_wait_does_not_consume_robodojo_action_limit(self):
        configured = config()
        configured["run"]["skip_unplanned_commands"] = True
        configured["policy"]["adapter"]["type"] = "manimux.policy_adapter.joint:JointAdapter"
        configured["inference"].update(inference_schedule="single_inflight", refill_threshold_s=0.08)
        backend = FakeAsyncSim()
        clock = FixedClock()
        result = AsyncSimRuntime(configured, client=backend, clock=clock, worker=DelayedWorker(clock)).run_policy(max_steps=3)
        self.assertEqual(result["steps"], 3)
        self.assertGreater(result["loop_ticks"], result["steps"])
        self.assertEqual(len(backend.commands), 3)
        self.assertTrue(all(command["plan_id"].startswith("joint-") for command in backend.commands))
        self.assertEqual(result["asyncsim_result"]["result"]["action_count"], 3)

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
        self.assertEqual(result["asyncsim_metrics"]["episode_id"], result["episode_id"])
        self.assertTrue(any(item["kind"] == "plan_committed" for item in result["audit"]["manimux_events"]))
        self.assertEqual(len(result["audit"]["actions"]), 3)
        self.assertEqual(len(result["packets"]), 2)
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

    def test_pi05_station_resolves_read_only_robodojo_checkpoint(self):
        base = Path(__file__).resolve().parents[2]
        source = base / "manimux/configs/experiments/asyncsim/robodojo_pi05_insert_key.yaml"
        local = base / "manimux/configs/local/asyncsim_pi05.example.yaml"
        configured = load_config(source, local=local)
        expected = configured["policy"]["expected_backend"]["model"]
        self.assertEqual(expected["model_root"], configured["policy_server"]["model_path"])
        self.assertEqual(expected["norm_stats_path"], configured["policy_server"]["norm_stats_path"])
        self.assertTrue(Path(expected["model_root"]).joinpath("params").is_dir())
        self.assertTrue(Path(expected["norm_stats_path"]).joinpath("norm_stats.json").is_file())
        from manimux.servers.asyncsim_pi05 import resolve_server

        self.assertEqual(resolve_server(source, local)["model_path"], expected["model_root"])

    def test_wrong_pi05_checkpoint_aborts_before_command(self):
        source = Path(__file__).resolve().parents[2] / (
            "manimux/configs/experiments/asyncsim/robodojo_pi05_insert_key.yaml"
        )
        local = source.parents[2] / "local/asyncsim_pi05.example.yaml"
        configured = load_config(source, local=local)
        clock = FixedClock()
        expected = configured["policy"]["expected_backend"]
        reported = {"server": expected["server"], "model": {
            **expected["model"], "model_root": "/another-checkpoint/59999",
        }}
        worker = FakeWorker(clock, PolicyCapabilities(backend_metadata=reported))
        backend = FakeAsyncSim()
        backend.health = lambda: {
            "episode": {"sim_ts": backend.sim_ts, "state": "running"},
            "canonical_group_dims": {"left_arm": 7, "right_arm": 7},
        }
        with self.assertRaisesRegex(RuntimeError, "model_root"):
            AsyncSimRuntime(configured, client=backend, clock=clock, worker=worker).run_policy(max_steps=2)
        self.assertFalse(worker.submitted)
        self.assertFalse(backend.commands)

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
