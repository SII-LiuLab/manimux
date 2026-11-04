"""Ablation-only independent implementation of RoboDojo's X5 IK/interpolation.

Only robot data and the cuRobo library are shared with RoboDojo. No RoboDojo
planner, IK, interpolation or execution function is imported here.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_pose(matrix):
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return np.concatenate([matrix[:3, 3], quaternion])


def relative_target(target, base_pose):
    # Retain the official sequence of transforms and float32 conversion.
    origin = pose_matrix(base_pose)
    inverse = np.eye(4)
    inverse[:3, :3] = origin[:3, :3].T
    inverse[:3, 3] = -origin[:3, :3].T @ origin[:3, 3]
    relative = pose_matrix([0, 0, 0, 1, 0, 0, 0]) @ (inverse @ pose_matrix(target))
    return np.asarray(matrix_pose(relative), dtype=np.float32)


def solver_config(data, active_joints, table_height):
    kinematics = deepcopy(data["robot_cfg"]["kinematics"])
    kinematics["tool_frames"] = list(kinematics.get("tool_frames", [kinematics.get("ee_link")]))
    kinematics["format_version"] = 2.0
    cspace = deepcopy(kinematics.get("cspace", {}))
    if "default_joint_position" not in cspace and "retract_config" in cspace:
        cspace["default_joint_position"] = cspace["retract_config"]
    allowed_cspace = {"joint_names", "default_joint_position", "cspace_distance_weight",
        "null_space_weight", "null_space_maximum_distance", "max_acceleration", "max_jerk",
        "velocity_scale", "acceleration_scale", "jerk_scale", "position_limit_clip"}
    cspace = {key: value for key, value in cspace.items() if key in allowed_cspace}
    cspace.setdefault("joint_names", list(active_joints))
    cspace.setdefault("default_joint_position", [0.0] * len(cspace["joint_names"]))
    kinematics["cspace"] = cspace
    allowed = {"format_version", "base_link", "tool_frames", "collision_link_names",
        "collision_spheres", "collision_sphere_buffer", "extra_collision_spheres",
        "self_collision_ignore", "self_collision_buffer", "asset_root_path", "mesh_link_names",
        "lock_joints", "extra_links", "cspace", "urdf_path", "use_global_cumul", "grasp_contact_link_names"}
    robot = {"robot_cfg": {"kinematics": {key: value for key, value in kinematics.items() if key in allowed}},
             "load_dynamics": bool(data.get("load_dynamics", False))}
    scene = {"cuboid": {"table": {"dims": [3.0, 3.0, 0.05],
        "pose": [0.0, 0.0, float(table_height) - 0.025, 1.0, 0.0, 0.0, 0.0]}}}
    return robot, scene


class RoboDojoCompatibleIK:
    """Own cuRobo solver instance with native model, seeds, costs and tolerances."""

    def __init__(self, yml_path, active_joints, table_height):
        import torch
        import curobo.runtime as runtime
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import DeviceCfg

        runtime.cuda_graph_reset = True
        self.torch = torch
        self.active_joints = list(active_joints)
        with open(yml_path) as source:
            data = yaml.safe_load(source)
        self.robot, self.scene = solver_config(data, active_joints, table_height)
        self.frame_bias = data["planner"]["frame_bias"]
        self.device = DeviceCfg()
        config = InverseKinematicsCfg.create(robot=self.robot, scene_model=self.scene,
            device_cfg=self.device, num_seeds=32, seed_solver_num_seeds=32,
            position_tolerance=0.001, orientation_tolerance=0.02, self_collision_check=True,
            use_cuda_graph=True, max_batch_size=1, multi_env=False, max_goalset=1)
        self.solver = InverseKinematics(config)
        self.names = list(self.solver.joint_names)
        cspace = self.robot["robot_cfg"]["kinematics"]["cspace"]
        retract = dict(zip(cspace["joint_names"], cspace["default_joint_position"]))
        self.retract = np.asarray([retract.get(name, 0.0) for name in self.names], dtype=np.float32)
        self.indices = [self.names.index(name) for name in self.active_joints]
        # Match the standalone IK warmup in RoboDojo, including sampler history.
        from curobo.types import GoalToolPose, JointState
        state = self.solver.default_joint_state.clone().unsqueeze(0)
        state = JointState.from_position(state.position, joint_names=self.names)
        kinematics = self.solver.compute_kinematics(state)
        goal = GoalToolPose.from_poses(kinematics.tool_poses.to_dict(),
            ordered_tool_frames=list(self.solver.tool_frames), num_goalset=1)
        self.solver.solve_pose(goal_tool_poses=goal, current_state=state, return_seeds=1)

    def solve(self, joints, target, base_pose):
        import transforms3d as t3d
        from curobo.types import GoalToolPose, JointState, Pose

        target = relative_target(target, base_pose)
        # The planner-origin frame is identity, but keep the same quat roundtrip.
        rotation = t3d.quaternions.quat2mat(np.array([1, 0, 0, 0], dtype=np.float32))
        position = rotation.T @ (target[:3] - np.zeros(3, dtype=np.float32))
        quaternion = t3d.quaternions.mat2quat(rotation.T @ t3d.quaternions.quat2mat(target[3:]))
        for index in range(3):
            position[index] += self.frame_bias[index]
        full = self.retract.copy()
        full[self.indices] = np.asarray(joints, dtype=np.float32)
        current = JointState.from_position(self.torch.as_tensor(full, dtype=self.torch.float32,
            device=self.device.device).reshape(1, -1), joint_names=self.names)
        goal = GoalToolPose.from_poses({self.solver.tool_frames[0]: Pose(
            position=self.torch.as_tensor(position, dtype=self.torch.float32, device=self.device.device).reshape(1, 3),
            quaternion=self.torch.as_tensor(quaternion, dtype=self.torch.float32, device=self.device.device).reshape(1, 4))},
            ordered_tool_frames=list(self.solver.tool_frames), num_goalset=1)
        result = self.solver.solve_pose(goal_tool_poses=goal, current_state=current, return_seeds=1)
        if result.success is None or not bool(result.success.any().item()):
            return {"status": "Fail"}
        values = np.asarray(result.js_solution.position.detach().cpu())
        while values.ndim > 1:
            values = values[0]
        names = list(getattr(result.js_solution, "joint_names", None) or [])
        if len(names) == len(values):
            values = values[[names.index(name) for name in self.active_joints]]
        elif len(values) == len(self.names):
            values = values[self.indices]
        elif len(values) != len(self.active_joints):
            raise ValueError("unexpected compatible IK output dimension")
        return {"status": "Success", "joint_value": values}


class RoboDojoCompatibleDecoder:
    """Decode one EEF target using current measured state, never a predicted seed."""

    def __init__(self, robots, ticks, *, solvers=None):
        self.robots, self.ticks = robots, ticks
        self.solvers = {} if solvers is None else solvers
        for robot in robots:
            name = robot["robot_name"]
            if name not in self.solvers:
                self.solvers[name] = RoboDojoCompatibleIK(robot["yml_path"], robot["joint_names"],
                    0.74 - robot["base_pose"][2])

    def decode(self, action, measured):
        controls, statuses = {}, {}
        for robot in self.robots:
            prefix, arm, grip = robot["prefix"], robot["arm"], robot["grip"]
            result = self.solvers[robot["robot_name"]].solve(measured[arm],
                action[prefix + "_ee_pose"], robot["base_pose"])
            statuses[prefix] = result["status"]
            if result["status"] == "Success":
                controls[arm] = {"position": result["joint_value"]}
            value = deepcopy(action[grip][0])
            value = np.clip(value, 0, 1)
            scale = robot["scale"]
            if robot["sign"] == 1:
                value = value * (scale[1] - scale[0]) + scale[0]
            else:
                value = (1 - value) * (scale[1] - scale[0]) + scale[0]
            controls[grip] = {"position": [value, value * robot["mimic"][1] + robot["mimic"][2]]}
        return self.interpolate(controls, measured), statuses

    def interpolate(self, controls, measured):
        if self.ticks <= 0:
            return [deepcopy(controls)]
        sequence = [deepcopy(controls) for _ in range(self.ticks)]
        count = int(np.floor(self.ticks * 0.8))
        for robot in self.robots:
            arm, grip = robot["arm"], robot["grip"]
            if arm in controls:
                current, target = np.array(measured[arm]), np.array(controls[arm]["position"])
                for index in range(count):
                    alpha = (index + 1) / (count + 1)
                    sequence[index][arm]["position"] = ((1 - alpha) * current + alpha * target).tolist()
                for index in range(count, self.ticks):
                    sequence[index][arm]["position"] = target.tolist()
            if grip in controls:
                current, target = measured[grip][0], controls[grip]["position"][0]
                for index in range(self.ticks):
                    alpha = (index + 1) / (count + 1)
                    value = (1 - alpha) * current + alpha * target if index < count else target
                    value = np.clip(value, robot["scale"][0], robot["scale"][1])
                    sequence[index][grip]["position"] = [value,
                        value * robot["mimic"][1] + robot["mimic"][2]]
        return sequence


def eef_targets(adapter, raw, context, robots):
    """Preserve ManiMux's FK/observation anchor; change only IK/interpolation."""
    from manimux.policy_adapter.xr1.arx_x5 import EEF_REFRAME, GROUPS, SLICES

    anchor = adapter._anchors.pop(context.request_seq, None)
    actions = np.asarray(raw, dtype=float)
    if anchor is None or actions.ndim != 2 or actions.shape[1] != 60 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("invalid XR1 observation anchor or packed EEF chunk")
    result = [dict() for _ in actions]
    for name, robot in zip(GROUPS, robots):
        pose = adapter.kinematics.models[name].fk(adapter._geometry_state(anchor[name]))
        rotation = pose[:3, :3] @ EEF_REFRAME
        base = pose_matrix(robot["base_pose"])
        pos_slice, rot_slice, grip_index = SLICES[name]
        for item, row in zip(result, actions):
            target = np.eye(4)
            target[:3, 3] = pose[:3, 3] + rotation @ row[pos_slice]
            target[:3, :3] = rotation @ Rotation.from_rotvec(row[rot_slice]).as_matrix() @ EEF_REFRAME.T
            output = matrix_pose(base @ target)
            if output[3] < 0:
                output[3:] *= -1
            item[robot["prefix"] + "_ee_pose"] = output.astype(np.float32)
            item[robot["grip"]] = np.array([anchor[name][-1] + row[grip_index]], dtype=np.float32)
    return result
