from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from manimux.embodiments.asyncsim import AsyncSimRobot, AsyncSimSensor
from manimux.cli import load_config
from manimux.runtime.asyncsim import AsyncSimRuntime
from manimux.types import ActionChunk, RobotCommand


class FakeAsyncSim:
    def __init__(self) -> None:
        self.episode_id = None
        self.commands = []
        self.streams = []
        self.closed = False
        self.sim_ts = 0.0

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
                "clock_anchor": {"sim_ts": self.sim_ts, "monotonic_ns": 1_000_000_000},
                "packets": {name: packets[name] for name in streams}}

    def health(self):
        return {"episode": {"sim_ts": self.sim_ts}}

    def submit_command(self, command):
        self.commands.append(command)
        return {"accepted": True, "reason": "accepted"}

    def close(self):
        self.closed = True


class FixedClock:
    def __init__(self):
        self.time_ns = 1_000_000_000

    def now_ns(self):
        return self.time_ns

    def sleep_until_ns(self, target_ns):
        self.time_ns = target_ns


def config():
    source = Path(__file__).resolve().parents[2] / "manimux/configs/experiments/asyncsim/fake_joint.yaml"
    return load_config(source)


class AsyncSimRuntimeTests(unittest.TestCase):
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
