from __future__ import annotations

import numpy as np
import pytest

import manimux.kinematics
from manimux.config import load_config
from manimux.integrations.sapolicy_yam.policy_plugin import (
    ARM_JOINTS,
    GROUP_DIM,
    WIRE_ACTION_DIM,
    SAPolicyYamAdapter,
    _pose_to_wire_endpose,
)
from manimux.types import (
    ActionContext,
    InferenceRequest,
    ObservationSnapshot,
    RobotState,
    SensorFrame,
)

CONFIG = "configs/sapolicy/yam/infra/manimux-xpl.yaml"


class _FakeKinematics:
    num_arm_joints = 6

    def __init__(self) -> None:
        self.fail_x: list[float] = []
        self.ik_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fk(self, joints, gripper):
        del gripper
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = float(np.asarray(joints)[0])
        pose[1, 3] = float(np.asarray(joints)[1])
        return pose

    def clip_arm_joints(self, joints):
        return np.asarray(joints, dtype=np.float64).reshape(-1).copy()

    def ik(self, target, seed, gripper):
        del gripper
        target = np.asarray(target, dtype=np.float64)
        seed = np.asarray(seed, dtype=np.float64)
        self.ik_calls.append((target.copy(), seed.copy()))
        solved = seed.copy()
        solved[0] = target[0, 3]
        solved[1] = target[1, 3]
        converged = not any(
            np.isclose(target[0, 3], value, atol=1e-8) for value in self.fail_x
        )
        return converged, solved


def _build_adapter(monkeypatch):
    config = load_config(CONFIG)
    fake = _FakeKinematics()
    monkeypatch.setattr(
        manimux.kinematics,
        "build_kinematics",
        lambda name, **options: fake,
    )
    adapter = SAPolicyYamAdapter(config.robot, config.policy)
    return adapter, fake


def _snapshot(now_ns: int, *, left_seed: float = 0.7, right_seed: float = -0.7):
    return ObservationSnapshot(
        state=RobotState(
            groups={
                "left_arm": np.array([left_seed, 0.2, 0.0, 0.0, 0.0, 0.0, 0.5]),
                "right_arm": np.array([right_seed, -0.2, 0.0, 0.0, 0.0, 0.0, 0.5]),
            },
            monotonic_ns=now_ns,
            sequence=1,
        ),
        frames={
            name: SensorFrame(
                name=name,
                data=np.zeros((8, 8, 3), dtype=np.uint8),
                capture_monotonic_ns=now_ns,
                sequence=1,
            )
            for name in ("front_camera", "left_camera", "right_camera")
        },
    )


def _wire_actions(adapter: SAPolicyYamAdapter) -> np.ndarray:
    packed = np.zeros((16, 14), dtype=np.float64)
    for step in range(16):
        packed[step, 0] = step * 0.01
        packed[step, 1] = 0.2
        packed[step, 6] = 0.5
        packed[step, 7] = -0.2 - step * 0.01
        packed[step, 8] = -0.2
        packed[step, 13] = 0.5
    wire = np.empty((len(packed), WIRE_ACTION_DIM), dtype=np.float64)
    for step, row in enumerate(packed):
        packed_offset = 0
        wire_offset = 0
        for group in adapter._group_order:
            state = row[packed_offset : packed_offset + GROUP_DIM]
            local_pose = adapter._kinematics.fk(state[:ARM_JOINTS], float(state[-1]))
            model_pose = adapter._model_from_kinematics[group] @ local_pose
            wire[step, wire_offset : wire_offset + 7] = _pose_to_wire_endpose(model_pose)
            wire[step, wire_offset + 7] = float(state[-1])
            packed_offset += GROUP_DIM
            wire_offset += 8
    return wire


def _anchor(adapter: SAPolicyYamAdapter, seq: int, snapshot: ObservationSnapshot) -> None:
    adapter.prepare_request(
        InferenceRequest(
            session_id="session",
            request_seq=seq,
            observation_time_ns=snapshot.state.monotonic_ns,
            deadline_ns=snapshot.state.monotonic_ns + 1_000_000_000,
            observation=snapshot,
        )
    )


def test_sapolicy_seeds_ik_from_measured_state(monkeypatch) -> None:
    adapter, fake = _build_adapter(monkeypatch)
    observation_ns = 1_000_000_000
    snapshot = _snapshot(observation_ns)
    _anchor(adapter, 1, snapshot)
    actions = _wire_actions(adapter)
    fake.ik_calls.clear()

    chunk = adapter.decode_action(
        actions,
        ActionContext(
            request_seq=1,
            observation_time_ns=observation_ns,
            created_time_ns=observation_ns + 100,
            execution_time_ns=observation_ns + 3 * adapter._action_dt_ns,
            measured_state=_snapshot(
                observation_ns + 3 * adapter._action_dt_ns,
                left_seed=0.8,
                right_seed=-0.8,
            ).state,
        ),
    )

    assert chunk.horizon_steps == 16
    assert chunk.source_offset_steps == 0
    assert fake.ik_calls[0][1][0] == pytest.approx(0.8)
    assert fake.ik_calls[16][1][0] == pytest.approx(-0.8)


def test_sapolicy_holds_last_good_when_ik_fails(monkeypatch) -> None:
    adapter, fake = _build_adapter(monkeypatch)
    observation_ns = 2_000_000_000
    snapshot = _snapshot(observation_ns)
    _anchor(adapter, 2, snapshot)
    actions = _wire_actions(adapter)
    fake.ik_calls.clear()
    fake.fail_x = [0.06]

    chunk = adapter.decode_action(
        actions,
        ActionContext(
            request_seq=2,
            observation_time_ns=observation_ns,
            created_time_ns=observation_ns + 100,
            measured_state=snapshot.state,
        ),
    )

    assert chunk.horizon_steps == 16
    assert chunk.groups["left_arm"].shape == (16, 7)
    assert chunk.groups["right_arm"].shape == (16, 7)
    np.testing.assert_allclose(chunk.groups["left_arm"][6, 0], chunk.groups["left_arm"][5, 0])
    assert chunk.groups["left_arm"][7, 0] == pytest.approx(0.07)
