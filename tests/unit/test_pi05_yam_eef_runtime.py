"""Exercise the same grouped kinematics injection as EdgeRuntime, without hardware."""

import numpy as np
import pytest

from manimux.cli import load_config
from manimux.embodiments.robot.base import RobotModel
from manimux.policy_adapter import build_policy_adapter
from manimux.types import ActionContext, InferenceRequest, ObservationSnapshot, RobotState


@pytest.mark.parametrize("injected", [True, False])
def test_grouped_kinematics_request_and_decode(injected):
    c = load_config("manimux/configs/experiments/put_bottles/pi05/yam_pi05_manimux_eef_step30000.yaml")
    kin = RobotModel.from_config(c["robot"]["config"]).kinematics
    a = build_policy_adapter(c["robot"], c["policy"], kinematics=kin if injected else None)
    if injected:
        assert a.kin is kin
    q = np.array([0.15, 0.47, 0.8, -1.0, -0.33, 0.27, 0.5])
    state = RobotState({"left_arm": q.copy(), "right_arm": q.copy()}, 1, 0)
    request = a.prepare_request(InferenceRequest("test", 0, 1, 100, ObservationSnapshot(state, {})))
    raw = []
    for _ in range(50):
        row = dict(request.xpolicylab_state)
        for side in ("left", "right"):
            row[side + "_ee_joint_state"] = np.array([0.5])
        raw.append(row)
    chunk = a.decode_action(raw, ActionContext(0, 1, 1, measured_state=state))
    assert chunk.horizon_steps == c["policy"]["adapter"]["decode_policy_steps"] == 20
    for group, values in chunk.groups.items():
        assert values.shape == (20, 7)
        np.testing.assert_allclose(values[:, 6], 0.5)
        target = kin.fk({group: q})[group]
        for solution in values:
            np.testing.assert_allclose(kin.fk({group: solution})[group], target, atol=2e-4)
