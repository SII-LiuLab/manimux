"""RoboDojo's gripper contract is applied at the ManiMux action boundary."""

import unittest

import numpy as np

from manimux.policy_adapter.robodojo_joint import RoboDojoJointAdapter
from manimux.types import ActionContext


class RoboDojoJointAdapterTest(unittest.TestCase):
    def setUp(self):
        self.robot = {"group_dims": {"left_arm": 7, "right_arm": 7}}
        self.policy = {
            "adapter": {
                "group_layouts": {
                    "left_arm": {"arm_dofs": 6, "gripper_dofs": 1},
                    "right_arm": {"arm_dofs": 6, "gripper_dofs": 1},
                },
            },
            "action_dt_s": 0.004,
            "horizon_policy_steps": 2,
        }

    def test_grippers_clip_without_changing_arm_or_source(self):
        adapter = RoboDojoJointAdapter(self.robot, self.policy)
        left = np.array([[0.2] * 6 + [-0.4], [0.3] * 6 + [1.2]])
        right = np.array([[0.4] * 6 + [1.1], [0.5] * 6 + [-0.1]])
        raw = {"format": "joint", "actions": {"left_arm": left, "right_arm": right}}
        chunk = adapter.decode_action(raw, ActionContext(1, 100, 200))
        np.testing.assert_array_equal(chunk.groups["left_arm"][:, -1], [0.0, 1.0])
        np.testing.assert_array_equal(chunk.groups["right_arm"][:, -1], [1.0, 0.0])
        np.testing.assert_array_equal(chunk.groups["left_arm"][:, :6], left[:, :6])
        np.testing.assert_array_equal(chunk.groups["right_arm"][:, :6], right[:, :6])
        np.testing.assert_array_equal(left[:, -1], [-0.4, 1.2])

    def test_nonfinite_action_rejected_before_clipping(self):
        adapter = RoboDojoJointAdapter(self.robot, self.policy)
        raw = {"format": "joint", "actions": {
            "left_arm": np.array([[0] * 6 + [float("nan")]] * 2),
            "right_arm": np.zeros((2, 7)),
        }}
        with self.assertRaisesRegex(ValueError, "invalid action matrix"):
            adapter.decode_action(raw, ActionContext(1, 100, 200))


if __name__ == "__main__":
    unittest.main()
