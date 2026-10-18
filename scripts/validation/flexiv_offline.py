"""Validate a Flexiv deployment's loader, geometry and action decoding without devices."""

import argparse
import json

import numpy as np

from manimux.cli import load_config
from manimux.embodiments.robot.base import RobotModel
from manimux.kinematics.poses import matrix_pose
from manimux.policies.xpolicylab.codec import _layouts_from_options, decode_policy_actions
from manimux.policy_adapter.joint import JointAdapter
from manimux.policy_adapter.pose import PoseAdapter
from manimux.types import ActionContext, RobotState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="manimux/configs/examples/flexiv_joint.yaml")
    parser.add_argument("--local", default=None)
    args = parser.parse_args()
    config = load_config(args.config, local=args.local)
    model = RobotModel.from_config(config["robot"]["config"])
    policy = config["policy"]
    kind = policy["options"]["action_format"]
    if kind not in {"joint", "pose"}:
        raise ValueError("this probe requires a joint or pose template")
    adapter_type = JointAdapter if kind == "joint" else PoseAdapter
    adapter = adapter_type(config["robot"], policy, kinematics=model.kinematics)
    layouts = _layouts_from_options(policy["adapter"], config["robot"]["group_dims"])
    q = np.array([0.2, 0.5, -0.3, 1.2, 0.4, 0.8, 1.5])
    state = RobotState({name: q.copy() for name in model.groups}, 1, 1)
    step = {}
    for layout in layouts:
        if layout.arm_dofs != 7 or layout.gripper_dofs:
            raise ValueError("this probe expects bare seven-axis arms")
        key = layout.arm_key if kind == "joint" else f"{layout.prefix}_ee_pose"
        step[key] = q.tolist() if kind == "joint" else matrix_pose(
            model.kinematics.models[layout.group].fk(q)
        )
    raw = decode_policy_actions(
        [step] * policy["horizon_policy_steps"], layouts=layouts, format=kind
    )
    chunk = adapter.decode_action(
        raw, ActionContext(1, 1, 1, observation_state=state, ik_seed=state)
    )
    for values in chunk.groups.values():
        np.testing.assert_allclose(values[0], q, atol=1e-4)
    print(json.dumps({
        "validation": "offline only; no SDK, model server, camera or hardware connection",
        "robot": model.name,
        "format": kind,
        "groups": {name: list(values.shape) for name, values in chunk.groups.items()},
        "execute": config["robot"]["options"].get("execute", False),
        "camera_map": policy["adapter"].get("camera_map", {}),
    }, indent=2))


if __name__ == "__main__":
    main()
