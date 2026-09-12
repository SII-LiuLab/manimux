"""Derive end-effector poses from recorded YAM joint state/action arrays."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

from ..config import StationConfig
from .schema import action_gripper_key, action_joint_key, gripper_pos_key, joint_pos_key

EE_SITE = "grasp_site"


def _arm_name(robot_type: str) -> str:
    return robot_type.replace("yam_lead_", "").replace("yam_", "") or "left"


class ForwardKinematics:
    """i2rt FK adapter for one configured YAM arm and gripper model."""

    def __init__(self, arm_type: str, gripper_type: str, num_arm_joints: int) -> None:
        self.arm_type = ArmType.from_string_name(arm_type)
        self.gripper_type = GripperType.from_string_name(gripper_type)
        self.num_arm_joints = num_arm_joints
        self.xml_path = combine_arm_and_gripper_xml(self.arm_type, self.gripper_type)
        try:
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
            self.kinematics = Kinematics(self.xml_path, EE_SITE)
        finally:
            # Both consumers have loaded independent in-memory models. i2rt's
            # combiner uses delete=False, so unlink or every episode leaks XMLs.
            Path(self.xml_path).unlink(missing_ok=True)

    def robot_state_to_qpos(self, joints: np.ndarray, gripper: np.ndarray | float) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64).reshape(-1)
        grip = float(np.asarray(gripper, dtype=np.float64).reshape(-1)[0])
        if joints.shape != (self.num_arm_joints,):
            raise ValueError(
                f"expected {self.num_arm_joints} arm joints, got shape {joints.shape}"
            )
        if not np.all(np.isfinite(joints)) or not np.isfinite(grip):
            raise ValueError("joint/gripper values must be finite")
        if not -1e-6 <= grip <= 1.0 + 1e-6:
            raise ValueError(f"normalized gripper must be in [0, 1], got {grip}")
        grip = float(np.clip(grip, 0.0, 1.0))

        robot_state = np.concatenate((joints, [grip]))
        if robot_state.size > self.model.nq:
            raise ValueError(
                f"robot state has {robot_state.size} values but model nq is {self.model.nq}"
            )
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[: robot_state.size] = robot_state

        # Same convention as i2rt's MujocoControlInterface._robot_cmd_to_qpos.
        for joint_id in range(self.model.njnt):
            address = int(self.model.jnt_qposadr[joint_id])
            if address >= robot_state.size:
                continue
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE:
                lower, upper = self.model.jnt_range[joint_id]
                qpos[address] = lower + qpos[address] * (upper - lower)
        for equality_id in range(self.model.neq):
            if self.model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            joint1 = int(self.model.eq_obj1id[equality_id])
            joint2 = int(self.model.eq_obj2id[equality_id])
            address1 = int(self.model.jnt_qposadr[joint1])
            address2 = int(self.model.jnt_qposadr[joint2])
            coefficients = self.model.eq_data[equality_id, :5]
            qpos[address2] = np.polyval(coefficients[::-1], qpos[address1])
        return qpos

    def batch(self, joints: np.ndarray, grippers: np.ndarray) -> dict[str, np.ndarray]:
        joints = np.asarray(joints, dtype=np.float64)
        grippers = np.asarray(grippers, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[1] != self.num_arm_joints:
            raise ValueError(
                f"joints must be (N,{self.num_arm_joints}), got {joints.shape}"
            )
        if grippers.shape != (len(joints), 1):
            raise ValueError(f"grippers must be (N,1), got {grippers.shape}")

        transforms = np.empty((len(joints), 4, 4), dtype=np.float64)
        for frame, (joint, gripper) in enumerate(zip(joints, grippers, strict=True)):
            qpos = self.robot_state_to_qpos(joint, gripper)
            transforms[frame] = self.kinematics.fk(qpos, site_name=EE_SITE)

        rotm = transforms[:, :3, :3]
        quat_xyzw = np.empty((len(joints), 4), dtype=np.float64)
        for frame, rotation in enumerate(rotm):
            quat_wxyz = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat_wxyz, rotation.reshape(9))
            quat_xyzw[frame] = quat_wxyz[[1, 2, 3, 0]]
            # q and -q encode the same rotation. Pick the sign closest to the
            # previous frame so a continuous trajectory does not contain fake jumps.
            if frame and np.dot(quat_xyzw[frame - 1], quat_xyzw[frame]) < 0.0:
                quat_xyzw[frame] *= -1.0

        # Fixed-axis XYZ / roll-pitch-yaw convention:
        # R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Values are [roll, pitch, yaw].
        pitch = np.arcsin(np.clip(-rotm[:, 2, 0], -1.0, 1.0))
        roll = np.arctan2(rotm[:, 2, 1], rotm[:, 2, 2])
        yaw = np.arctan2(rotm[:, 1, 0], rotm[:, 0, 0])
        euler_xyz = np.stack((roll, pitch, yaw), axis=1)

        return {
            "ee_pos": transforms[:, :3, 3].copy(),
            "ee_rotm": rotm.reshape(len(joints), 9).copy(),
            "ee_quat": quat_xyzw,
            "ee_euler_xyz": euler_xyz,
            "ee_transform": transforms,
        }


def add_eepose_buffers(
    buffers: dict[str, np.ndarray], station: StationConfig, arm_names: list[str]
) -> tuple[dict[str, np.ndarray], dict]:
    """Add observed and action EE representations to an episode buffer."""
    robots = {_arm_name(robot.type): robot for robot in station.robot.robots}
    models: dict[str, dict] = {}
    out = dict(buffers)

    for arm in arm_names:
        robot = robots.get(arm)
        if robot is None:
            raise ValueError(f"no robot configuration found for recorded arm {arm!r}")
        fk = ForwardKinematics(
            station.robot.arm_type,
            robot.gripper,
            station.robot.num_arm_joints,
        )
        models[arm] = {
            "arm_type": station.robot.arm_type,
            "gripper_type": robot.gripper,
            "site": EE_SITE,
        }
        for prefix, joint_key, gripper_key in (
            (arm, joint_pos_key(arm), gripper_pos_key(arm)),
            (f"action-{arm}", action_joint_key(arm), action_gripper_key(arm)),
        ):
            pose = fk.batch(out[joint_key], out[gripper_key])
            for suffix, values in pose.items():
                out[f"{prefix}-{suffix}"] = values

    metadata = {
        "enabled": True,
        "derived_from": "recorded joint/gripper arrays using official i2rt forward kinematics",
        "reference_frame": (
            "each arm's own MuJoCo model world frame, coincident with that arm base; "
            "left/right absolute positions are not in a shared calibrated bimanual frame"
        ),
        "transform_direction": "T_base_ee (maps EE-frame coordinates into the arm base frame)",
        "models": models,
        "fields": {
            "*-ee_pos": {"shape": ["N", 3], "components": ["x", "y", "z"], "unit": "m"},
            "*-ee_rotm": {
                "shape": ["N", 9],
                "components": ["r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22"],
                "layout": "row-major 3x3 rotation matrix",
            },
            "*-ee_quat": {
                "shape": ["N", 4],
                "components": ["qx", "qy", "qz", "qw"],
                "order": "xyzw",
            },
            "*-ee_euler_xyz": {
                "shape": ["N", 3],
                "components": ["roll_x", "pitch_y", "yaw_z"],
                "unit": "rad",
                "convention": "fixed-axis XYZ; R = Rz(yaw) @ Ry(pitch) @ Rx(roll)",
            },
            "*-ee_transform": {
                "shape": ["N", 4, 4],
                "description": "homogeneous transform T_base_ee",
            },
        },
        "prefixes": {
            "<arm>-*": "observed follower state",
            "action-<arm>-*": "commanded action target",
        },
    }
    return out, metadata
