"""Config instantiation + start-position helpers used by the eval launcher.

Trimmed from the upstream ``gello.utils.launch_utils`` to only the two helpers
the MolmoAct eval path needs: ``instantiate_from_dict`` (build a robot from a
``_target_`` config block) and ``move_to_start_position`` (interpolate the
arm(s) to ``agent.start_joints`` between rollouts). The teleop launch manager
and its Dynamixel/ZMQ dependencies are intentionally left out.
"""

import importlib
import time
from typing import Any

import numpy as np


def instantiate_from_dict(cfg):
    """Recursively instantiate objects from a ``_target_``-style config dict."""
    if isinstance(cfg, dict) and "_target_" in cfg:
        module_path, class_name = cfg["_target_"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        kwargs = {k: v for k, v in cfg.items() if k != "_target_"}
        return cls(**{k: instantiate_from_dict(v) for k, v in kwargs.items()})
    elif isinstance(cfg, dict):
        return {k: instantiate_from_dict(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [instantiate_from_dict(v) for v in cfg]
    else:
        return cfg


def move_to_start_position(
    env,
    bimanual: bool = False,
    left_cfg: dict[str, Any] | None = None,
    right_cfg: dict[str, Any] | None = None,
):
    """Interpolate the robot to ``agent.start_joints`` if specified in config."""
    if bimanual:
        if right_cfg is None:
            return
        left_start = left_cfg["agent"].get("start_joints")
        right_start = right_cfg["agent"].get("start_joints")
        if left_start is None or right_start is None:
            return
        reset_joints = np.concatenate([np.array(left_start), np.array(right_start)])
    else:
        if "start_joints" not in left_cfg["agent"] or left_cfg["agent"]["start_joints"] is None:
            return
        reset_joints = np.array(left_cfg["agent"]["start_joints"])

    # Parking only needs robot state. Do not make safe shutdown depend on the
    # camera server still being reachable.
    curr_joints = env.get_robot_state()["joint_positions"]
    if reset_joints.shape != curr_joints.shape:
        print("Warning: Mismatch in joint shapes, skipping move_to_start_position.")
        return

    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = max(2, min(int(np.ceil(max_delta / 0.01)) + 1, 100))

    print(f"Moving robot to start position: {reset_joints}")
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step_command_only(jnt, reset=True)
        time.sleep(0.001)


def move_to_zero_home(env, bimanual: bool = False, time_interval_s: float = 5.0):
    """Move each YAM follower to its calibrated zero-joint home via i2rt."""
    home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    robot = env.robot()
    arms = [robot._robot_l, robot._robot_r] if bimanual else [robot]

    print(f"Moving robot to zero home: {home}")
    for arm in arms:
        native_robot = getattr(arm, "robot", None)
        move_joints = getattr(native_robot, "move_joints", None)
        if not callable(move_joints):
            raise RuntimeError(
                f"{arm.__class__.__name__} does not expose i2rt move_joints; "
                "cannot safely return to zero home."
            )
        move_joints(home, time_interval_s=time_interval_s)
