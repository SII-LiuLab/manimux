"""DP deployment preserves absolute poses, camera history and checkpoint timing."""

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.cli import load_config
from manimux.policy_adapter.dp.yam import DPYamAdapter, HistorySnapshot, HistoryStrategy
from manimux.types import InferenceRequest, RobotState, SensorFrame
from XPolicyLab.policy.DP.eef_codec import pack_eef, unpack_eef

EXPERIMENT = "manimux/configs/experiments/put_bottles/yam_dp_serial_eef_step100000.yaml"


def test_eef_wire_roundtrip():
    dims = dict(arm_dim=[6, 6], ee_dim=[1, 1])
    row = np.array([0.1, 0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.2, 0.1, 0.4, -0.3, 0.2, -0.1, 0.8])
    wire = unpack_eef(row[None], dims)[0]
    assert wire["left_ee_pose"].shape == (7,)
    np.testing.assert_allclose(pack_eef({"state": wire}, dims), row, atol=1e-7)


def test_history_and_fk_request():
    c = load_config(EXPERIMENT)
    adapter = DPYamAdapter(c["robot"], c["policy"])
    samples = []
    for i in range(3):
        q = np.array([0.0, 1.0, 1.0, -0.5, 0.0, 0.0, 0.5])
        state = RobotState(dict(left_arm=q, right_arm=q), 1_000_000_000 + i * 33_333_333, i)
        frames = {
            n: SensorFrame(n, np.full((240, 320, 3), i, dtype=np.uint8), state.monotonic_ns, i)
            for n in ("front_camera", "left_camera", "right_camera")
        }
        from manimux.types import ObservationSnapshot

        samples.append(ObservationSnapshot(state, frames))
    snapshot = HistorySnapshot(samples[-1].state, samples[-1].frames, tuple(samples))
    r = adapter.prepare_request(
        InferenceRequest("test", 0, snapshot.state.monotonic_ns, 2_000_000_000, snapshot)
    )
    assert len(r.observation.frames) == 9
    assert len(r.xpolicylab_additional_info["dp_history_states"]) == 3
    for i in range(3):
        assert r.observation.frames[f"front_camera_t{i}"].data[0, 0, 0] == i
    pose = r.xpolicylab_additional_info["dp_history_states"][0]["left_ee_pose"]
    expected = adapter.kin.fk(q[:6], q[-1])
    np.testing.assert_allclose(pose[:3], expected[:3, 3])
    np.testing.assert_allclose(
        Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix(), expected[:3, :3], atol=1e-9
    )


def test_history_waits_for_measured_samples():
    c = load_config(EXPERIMENT)
    strategy = HistoryStrategy(c)

    class Delegate:
        def build_submission(self, **kwargs):
            return kwargs["snapshot"]

    strategy.delegate = Delegate()
    from manimux.types import ObservationSnapshot

    for i in range(3):
        ns = 1_000_000_000 + i * 33_333_333
        frames = {
            n: SensorFrame(n, np.zeros((2, 2, 3), np.uint8), ns, i) for n in strategy.history.names
        }
        snapshot = ObservationSnapshot(RobotState({"left_arm": np.zeros(7)}, ns, i), frames)
        result = strategy.build_submission(snapshot=snapshot, now_ns=ns)
        if i < 2:
            assert result is None
    assert len(result.samples) == 3
    # Polling the same camera frame cannot fabricate another history sample.
    strategy.build_submission(snapshot=snapshot, now_ns=ns + 10_000_000)
    assert len(strategy.history.samples) == 3
