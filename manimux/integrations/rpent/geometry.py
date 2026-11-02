"""Adapt ManiMux's YAM model to RPent's existing calibration and path checks."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.embodiments.arm.yam.kinematics import YamManipulatorKinematics
from manimux.kinematics.robot import RobotKinematics


class YamPlanningKinematics:
    """The same TCP, gripper mapping and MuJoCo model used by ManiMux execution."""

    def __init__(self, kinematics: YamManipulatorKinematics, timeout_s=1.0) -> None:
        if not isinstance(kinematics, YamManipulatorKinematics):
            raise TypeError("RPent YAM planning requires the integrated YAM arm/gripper model")
        self.timeout_s = float(timeout_s)
        if not np.isfinite(self.timeout_s) or not 0 < self.timeout_s <= 30:
            raise ValueError("planning_timeout_s must be in (0, 30]")
        self.kinematics = kinematics
        self.model = kinematics.solver.model

    def fk(self, joints, gripper):
        return self.kinematics.fk(np.r_[joints, gripper])

    def solve(self, target, seed, gripper):
        result = self.kinematics.ik_deadline(
            target,
            np.r_[seed, gripper],
            fixed_coordinates={"gripper": gripper},
            deadline_ns=time.monotonic_ns() + int(self.timeout_s * 1e9),
        )
        joints = np.asarray(result.joints[:6] if result.converged else seed)
        actual = self.fk(joints, gripper)
        return SimpleNamespace(
            success=result.converged,
            q_target=joints,
            reason=result.reason,
            position_error=float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
            rotation_error=float(
                Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).magnitude()
            ),
        )


def build_geometry(kinematics: RobotKinematics, options: dict):
    from robots.yam.geometry import YamGeometry

    if set(kinematics.models) != {"left_arm", "right_arm"}:
        raise ValueError("RPent YAM geometry requires left_arm and right_arm")
    adapters = {
        side: YamPlanningKinematics(
            kinematics.models[f"{side}_arm"],
            options.get("planning_timeout_s", 1.0),
        )
        for side in ("left", "right")
    }
    geometry = YamGeometry(options, kinematics=adapters)
    for side in ("left", "right"):
        configured = adapters[side].kinematics.solver.joint_position_limits()
        index = 0 if side == "left" else 1
        # RPent and the controller must share the stricter interval, not relax it.
        geometry.lower[index] = np.maximum(geometry.lower[index], configured[0])
        geometry.upper[index] = np.minimum(geometry.upper[index], configured[1])
    if np.any(geometry.lower >= geometry.upper):
        raise ValueError("RPent and ManiMux YAM joint limits have no shared interval")
    if geometry.calibration.world_from_right_base is None:
        raise ValueError("RPent primitives require calibrated left/right base transforms")
    geometry.calibration.cam2world_cv("top")
    if set(geometry.calibration.wrist_grasp_from_camera) != {"left", "right"}:
        raise ValueError("RPent primitives require both wrist hand-eye calibrations")
    if not geometry.collision_guard_enabled or not geometry.table_guard_configured:
        raise ValueError("RPent primitives require collision_guard and a configured table")
    # Validate the exact model layout before serving; construction performs no I/O to hardware.
    geometry._collision_guard = build_collision_guard(geometry)
    return geometry


def build_path_validator(kinematics: RobotKinematics, options: dict):
    """Reuse calibrated whole-YAM collision checks for the unified tool runtime."""
    geometry = build_geometry(kinematics, options)

    def validate(groups, reference):
        previous = np.r_[reference["left_arm"], reference["right_arm"]]
        for left, right in zip(groups["left_arm"], groups["right_arm"], strict=True):
            target = np.r_[left, right]
            result = geometry.check_qpos_transition(previous, target)
            if not result["ok"]:
                raise ValueError(f"YAM path rejected: {result}")
            previous = target

    return validate


def build_collision_guard(geometry):
    from robots.yam.geometry import _ModelCollisionGuard

    class ManiMuxYamGuard(_ModelCollisionGuard):
        """Retain upstream collision checks with an explicit linear_4310 model contract."""

        def _validate_model(self, model):
            bodies = [
                "base",
                "link1",
                "link2",
                "link3",
                "link4",
                "link5",
                "link6",
                "gripper",
                "tip_left",
                "tip_right",
            ]
            if model.nbody == 10:
                return super()._validate_model(model)
            if (
                model.nbody != 11
                or model.njnt != 8
                or model.nq != 8
                or model.nv != 8
                or {model.body(i).name for i in range(1, model.nbody)} != set(bodies)
                or {model.joint(i).name for i in range(model.njnt)}
                != {f"joint{i}" for i in range(1, 9)}
            ):
                raise ValueError("collision_guard requires a supported complete YAM model")
            for index, name in enumerate(bodies):
                body = model.body(name)
                parent = "world" if index == 0 else "gripper" if index >= 8 else bodies[index - 1]
                geoms = np.flatnonzero(model.geom_bodyid == body.id)
                if (
                    model.body(int(body.parentid[0])).name != parent
                    or len(geoms) != (0 if name == "link6" else 1)
                    or np.any(model.geom_type[geoms] != self._mj.mjtGeom.mjGEOM_MESH)
                    or not np.isfinite(model.geom_rbound[geoms]).all()
                    or np.any(model.geom_rbound[geoms] <= 0)
                ):
                    raise ValueError(f"collision_guard unsupported linear YAM body {name}")
            expected = {
                frozenset(pair)
                for pair in (
                    ("gripper", "tip_left"),
                    ("gripper", "tip_right"),
                    ("tip_left", "tip_right"),
                )
            }
            excluded = {
                frozenset((model.body(int(v) >> 16).name, model.body(int(v) & 65535).name))
                for v in model.exclude_signature
            }
            if excluded != expected:
                raise ValueError("collision_guard unsupported linear YAM exclusions")
            for i in range(1, 9):
                joint = model.joint(f"joint{i}")
                body = f"link{i}" if i <= 6 else "tip_left" if i == 7 else "tip_right"
                kind = self._mj.mjtJoint.mjJNT_HINGE if i <= 6 else self._mj.mjtJoint.mjJNT_SLIDE
                if (
                    model.body(int(joint.bodyid[0])).name != body
                    or joint.type[0] != kind
                    or not joint.limited[0]
                    or not np.isfinite(joint.range).all()
                    or joint.range[0] >= joint.range[1]
                ):
                    raise ValueError(f"collision_guard invalid linear YAM joint{i}")

    return ManiMuxYamGuard(geometry, geometry.collision_clearance_m)
