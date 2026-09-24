"""Pi05 auxiliary EEF deployment uses observation TCP deltas and retains grippers."""

import numpy as np
import pytest
from openpi.policies import yam_policy
from scipy.spatial.transform import Rotation

from XPolicyLab.policy.Pi_05.model import Model, _resolve_train_config
from XPolicyLab.policy.Pi_05.yam_eef import absolute_eef_actions


def test_training_delta_inverse_and_grippers():
    current = np.tile(
        np.r_[[0.2, 0.1, 0.3], Rotation.from_euler("xyz", [0.2, -0.3, 1.0]).as_matrix().ravel()], 2
    )
    targets = np.tile(current, (5, 1))
    targets[:, 0] += 0.04
    targets[:, 12] -= 0.03
    for i in (0, 12):
        targets[:, i + 3 : i + 12] = (
            (
                Rotation.from_matrix(current[i + 3 : i + 12].reshape(3, 3))
                * Rotation.from_rotvec([0.1, 0.2, -0.1])
            )
            .as_matrix()
            .ravel()
        )
    delta = yam_policy.relative_ee_actions(current, targets)
    raw = np.zeros((5, 26))
    raw[:, 6] = 0.2
    raw[:, 13] = 0.8
    raw[:, 14:] = delta
    anchors = {}
    for side, i in [("left", 0), ("right", 12)]:
        q = Rotation.from_matrix(current[i + 3 : i + 12].reshape(3, 3)).as_quat()
        anchors[side + "_ee_pose"] = np.r_[current[i : i + 3], q[3], q[:3]]
    rows = absolute_eef_actions(raw, anchors, dict(arm_dim=[6, 6], ee_dim=[1, 1]))
    for row in rows:
        for side, i, g in [("left", 0, 0.2), ("right", 12, 0.8)]:
            p = row[side + "_ee_pose"]
            np.testing.assert_allclose(p[:3], targets[0, i : i + 3], atol=1e-7)
            np.testing.assert_allclose(
                Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix(),
                targets[0, i + 3 : i + 12].reshape(3, 3),
                atol=1e-7,
            )
            np.testing.assert_allclose(row[side + "_ee_joint_state"], [g])
            assert side + "_arm_joint_state" not in row


def test_opt_in_preserves_joint_default():
    joint = _resolve_train_config(dict(train_config_name="pi05_yam_joint_ee", action_type="joint"))
    eef = _resolve_train_config(dict(train_config_name="pi05_yam_joint_ee", action_type="ee"))
    assert not joint.data.expose_eef_actions and eef.data.expose_eef_actions
    for c, width in [(joint, 14), (eef, 26)]:
        d = c.data.create(c.assets_dirs, c.model)
        transform = next(
            t for t in d.data_transforms.outputs if isinstance(t, yam_policy.YamOutputs)
        )
        assert transform({"actions": np.zeros((50, 32))})["actions"].shape == (50, width)


def test_sampling_modes_reject_unimplemented_eef_guidance():
    m = Model.__new__(Model)
    m.yam_eef = True
    assert m.sampling_modes() == ["default"]
    for method in ("rtc", "paint", "aac", "autohorizon", "dvac"):
        with pytest.raises(ValueError, match="default sampling"):
            getattr(m, "get_action_" + method)({})
    m.yam_eef = False
    assert "rtc" in m.sampling_modes()


def test_runtime_prefix_and_failure_diagnostics():
    from manimux.kinematics import IKResult, RobotKinematics
    from manimux.policy_adapter.pi05.yam_eef import Pi05YamEefAdapter
    from manimux.types import ActionContext, InferenceRequest, ObservationSnapshot, RobotState

    class Kin:
        calls = 0
        fail = False

        def fk(self, q):
            return np.eye(4)

        def ik(self, target, seed, *, fixed_coordinates):
            self.calls += 1
            if self.fail:
                return IKResult(False, reason="no_solution")
            return IKResult(True, np.r_[seed[:6], fixed_coordinates["gripper"]])

    kin = Kin()
    adapter = Pi05YamEefAdapter(
        {"group_dims": {"left_arm": 7, "right_arm": 7}},
        {"action_dt_s": 1 / 30, "horizon_policy_steps": 50, "adapter": {"decode_policy_steps": 12}},
        kinematics=RobotKinematics({"left_arm": kin, "right_arm": kin}),
    )
    state = RobotState({side + "_arm": np.zeros(7) for side in ("left", "right")}, 1, 0)
    request = adapter.prepare_request(
        InferenceRequest("test", 0, 1, 100, ObservationSnapshot(state, {}))
    )
    assert set(request.xpolicylab_state) == {"left_ee_pose", "right_ee_pose"}
    row = {
        side + key: val
        for side in ("left", "right")
        for key, val in [
            ("_ee_pose", np.array([0, 0, 0, 1, 0, 0, 0])),
            ("_ee_joint_state", np.array([0.4])),
        ]
    }
    context = ActionContext(0, 1, 1, measured_state=state)
    chunk = adapter.decode_action([row] * 50, context)
    assert kin.calls == 24 and chunk.horizon_steps == 12
    kin.fail = True
    with pytest.raises(ValueError, match="no_solution"):
        adapter.decode_action([row] * 50, context)
