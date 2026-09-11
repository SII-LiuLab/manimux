"""YAM arm kinematics on the bundled i2rt model assets.

One implementation serves both consumers: the viewer renders end-effector trails
with :meth:`fk`, and pose-space policy adapters (XR-1 and friends) additionally
solve :meth:`ik`. Forward kinematics needs nothing but ``mujoco`` so it works in
the viewer's slim environment; the IK solver is imported lazily because ``mink``
only lives in the robot environment.

The joint -> ``qpos`` mapping matches i2rt's ``MujocoControlInterface`` and the
recorder that produced the datasets' ``ee_pos`` / ``ee_rotm``: the gripper is a
normalized [0, 1] fraction of the slide joint's stroke, and joint equality
constraints drive the mimic finger.
"""

from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

from manimux.kinematics.base import FloatArray

EE_SITE = "grasp_site"
DEFAULT_ASSETS_ROOT = Path(__file__).resolve().parents[1] / "assets"


def combined_model_xml(assets_root: Path, arm: str, gripper: str) -> str:
    """Splice the gripper body onto the arm's kinematic chain, meshes resolved."""
    models = assets_root / "i2rt/robot_models"
    arm_path = models / f"arm/{arm}/{arm}.xml"
    gripper_path = models / f"gripper/{gripper}/{gripper}.xml"
    for path in (arm_path, gripper_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"YAM model asset not found: {path}. "
                "Pass assets_root with the i2rt model checkout path."
            )

    arm_root = ET.parse(arm_path).getroot()
    gripper_root = ET.parse(gripper_path).getroot()
    arm_asset = arm_root.find("asset")
    if arm_asset is None:
        raise RuntimeError(f"asset section is missing from {arm_path}")
    for mesh in arm_asset.findall("mesh"):
        filename = Path(mesh.get("file", "")).name
        mesh.set("file", str((arm_path.parent / "assets" / filename).resolve()))
    gripper_asset = gripper_root.find("asset")
    if gripper_asset is not None:
        for mesh in gripper_asset.findall("mesh"):
            copy = deepcopy(mesh)
            filename = Path(mesh.get("file", "")).name
            copy.set("file", str((gripper_path.parent / "assets" / filename).resolve()))
            arm_asset.append(copy)
    compiler = arm_root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    body = arm_root.find("worldbody/body")
    if body is None:
        raise RuntimeError(f"root body is missing from {arm_path}")
    while body.find("body") is not None:
        child = body.find("body")
        assert child is not None
        body = child
    gripper_body = gripper_root.find("worldbody/body[@name='gripper']")
    if gripper_body is None:
        raise RuntimeError(f"gripper body is missing from {gripper_path}")
    body.append(deepcopy(gripper_body))
    for tag in ("equality", "contact"):
        element = gripper_root.find(tag)
        if element is not None:
            arm_root.append(deepcopy(element))
    return ET.tostring(arm_root, encoding="unicode")


class YamKinematics:
    """FK (always) and IK (when ``mink`` is installed) for one YAM arm."""

    def __init__(
        self,
        arm_type: str = "yam",
        gripper_type: str = "linear_4310",
        num_arm_joints: int = 6,
        *,
        assets_root: Path | str | None = None,
        pos_threshold: float = 1e-4,
        ori_threshold: float = 1e-4,
        max_iters: int = 200,
        recovery_ori_threshold: float | None = None,
        recovery_max_joint_delta: float = 0.35,
    ) -> None:
        self._num_arm_joints = int(num_arm_joints)
        self._pos_threshold = float(pos_threshold)
        self._ori_threshold = float(ori_threshold)
        self._max_iters = int(max_iters)
        if recovery_ori_threshold is not None and not 0 < recovery_ori_threshold <= np.pi / 18:
            raise ValueError("recovery_ori_threshold must be in (0, 10 degrees]")
        if recovery_max_joint_delta <= 0:
            raise ValueError("recovery_max_joint_delta must be positive")
        self._recovery_ori_threshold = recovery_ori_threshold
        self._recovery_max_joint_delta = float(recovery_max_joint_delta)
        self._assets_root = Path(assets_root) if assets_root else DEFAULT_ASSETS_ROOT
        self._arm_type = arm_type
        self._gripper_type = gripper_type

        self._xml = combined_model_xml(self._assets_root, arm_type, gripper_type)
        self.model = mujoco.MjModel.from_xml_string(self._xml)
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        if self.site_id < 0:
            raise RuntimeError(f"{EE_SITE} is missing from the {arm_type} model")

        # mujoco's MjData and mink's Configuration are both single-writer.
        self._lock = threading.Lock()
        self._solver: object | None = None
        self._limits: list | None = None
        self._arm_limits = self.joint_position_limits()

    @property
    def num_arm_joints(self) -> int:
        return self._num_arm_joints

    @property
    def state_dim(self) -> int:
        return self._num_arm_joints + 1

    def robot_state_to_qpos(self, joints: FloatArray, gripper: float) -> FloatArray:
        """Map [arm joints, normalized gripper] onto the MuJoCo qpos vector."""
        joints = np.asarray(joints, dtype=np.float64).reshape(-1)
        grip = float(np.asarray(gripper, dtype=np.float64).reshape(-1)[0])
        if joints.shape != (self._num_arm_joints,):
            raise ValueError(
                f"expected {self._num_arm_joints} arm joints, got shape {joints.shape}"
            )
        if not np.isfinite(joints).all() or not np.isfinite(grip):
            raise ValueError("joint/gripper values must be finite")

        width = self.state_dim
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[: self._num_arm_joints] = joints
        # The gripper is a normalized fraction of the slide joint's stroke.
        qpos[self._num_arm_joints] = np.clip(grip, 0.0, 1.0)

        for joint_id in range(self.model.njnt):
            address = int(self.model.jnt_qposadr[joint_id])
            if address < width and self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE:
                lower, upper = self.model.jnt_range[joint_id]
                qpos[address] = lower + qpos[address] * (upper - lower)
        for equality_id in range(self.model.neq):
            if self.model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            joint_1 = int(self.model.eq_obj1id[equality_id])
            joint_2 = int(self.model.eq_obj2id[equality_id])
            address_1 = int(self.model.jnt_qposadr[joint_1])
            address_2 = int(self.model.jnt_qposadr[joint_2])
            coefficients = self.model.eq_data[equality_id, :5][::-1]
            qpos[address_2] = np.polyval(coefficients, qpos[address_1])
        return qpos

    def fk(self, joints: FloatArray, gripper: float) -> FloatArray:
        """Joint positions -> 4x4 grasp-site transform in the arm base frame."""
        qpos = self.robot_state_to_qpos(joints, gripper)
        with self._lock:
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
            transform[:3, 3] = self.data.site_xpos[self.site_id]
        return transform

    def pose(self, configuration: FloatArray) -> FloatArray:
        """FK from one packed ``[joints..., gripper]`` vector (viewer entry point)."""
        values = np.asarray(configuration, dtype=np.float64).reshape(-1)
        if values.size != self.state_dim:
            raise ValueError(f"expected {self.state_dim} YAM values, got {values.size}")
        return self.fk(values[: self._num_arm_joints], float(values[-1]))

    def joint_position_limits(self) -> tuple[FloatArray, FloatArray]:
        """Return lower/upper limits for the canonical arm-joint vector."""
        if hasattr(self, "_arm_limits"):
            return self._arm_limits[0].copy(), self._arm_limits[1].copy()
        lower = np.full(self._num_arm_joints, -np.inf, dtype=np.float64)
        upper = np.full(self._num_arm_joints, np.inf, dtype=np.float64)
        for qpos_index in range(self._num_arm_joints):
            matches = np.flatnonzero(self.model.jnt_qposadr == qpos_index)
            if matches.size != 1:
                raise RuntimeError(
                    f"YAM qpos {qpos_index} does not map to exactly one joint"
                )
            joint_id = int(matches[0])
            if bool(self.model.jnt_limited[joint_id]):
                lower[qpos_index], upper[qpos_index] = self.model.jnt_range[joint_id]
        return lower, upper

    def clip_arm_joints(self, joints: FloatArray) -> FloatArray:
        """Project arm joints onto the model's finite position limits."""
        values = np.asarray(joints, dtype=np.float64).reshape(-1)
        if values.shape != (self._num_arm_joints,) or not np.isfinite(values).all():
            raise ValueError(
                f"expected {self._num_arm_joints} finite arm joints, got {values.shape}"
            )
        lower, upper = self.joint_position_limits()
        return np.clip(values, lower, upper)

    def joint_limit_margins(self, joints: FloatArray) -> FloatArray:
        """Signed distance from each arm joint to its nearest position limit."""
        values = np.asarray(joints, dtype=np.float64).reshape(-1)
        if values.shape != (self._num_arm_joints,) or not np.isfinite(values).all():
            raise ValueError(
                f"expected {self._num_arm_joints} finite arm joints, got {values.shape}"
            )
        lower, upper = self.joint_position_limits()
        return np.minimum(values - lower, upper - values)

    def pose_error(
        self,
        target_pose: FloatArray,
        joints: FloatArray,
        gripper: float,
    ) -> tuple[float, float]:
        """Return translational and rotational FK residuals for IK diagnostics."""
        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError(f"target_pose must be a finite 4x4 transform, got {target.shape}")
        actual = self.fk(joints, gripper)
        position_error_m = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        relative_rotation = actual[:3, :3].T @ target[:3, :3]
        cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
        orientation_error_rad = float(np.arccos(cosine))
        return position_error_m, orientation_error_rad

    def _joint_limits(self) -> list:
        import mink

        if self._limits is None:
            self._limits = [mink.ConfigurationLimit(self.model)]
        return self._limits

    def _get_solver(self) -> object:
        if self._solver is None:
            import tempfile

            from i2rt.robots.kinematics import Kinematics

            # i2rt's solver loads from a path; hand it the same spliced model.
            with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as handle:
                handle.write(self._xml)
                xml_path = handle.name
            try:
                self._solver = Kinematics(xml_path, EE_SITE)
            finally:
                Path(xml_path).unlink(missing_ok=True)
        return self._solver

    def ik(
        self,
        target_pose: FloatArray,
        init_joints: FloatArray,
        gripper: float,
    ) -> tuple[bool, FloatArray]:
        """4x4 grasp-site target -> (converged, joint positions).

        ``init_joints`` seeds the differential solver, so passing the measured
        joints keeps the solution on the branch the arm is already on.
        """
        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError(f"target_pose must be a finite 4x4 transform, got {target.shape}")
        init_qpos = self.robot_state_to_qpos(init_joints, gripper)
        with self._lock:
            solver = self._get_solver()
            converged, qpos = solver.ik(  # type: ignore[attr-defined]
                target,
                site_name=EE_SITE,
                init_q=init_qpos,
                # Without limits the differential solver happily integrates past
                # the joint stops while chasing an unreachable target.
                limits=self._joint_limits(),
                pos_threshold=self._pos_threshold,
                ori_threshold=self._ori_threshold,
                max_iters=self._max_iters,
            )
        joints = np.asarray(qpos, dtype=np.float64)[: self._num_arm_joints].copy()
        # mink's ConfigurationLimit still leaks ~1 mrad past a stop; clip so the
        # commanded joints stay inside the same range SafetyGuard enforces.
        return bool(converged), self.clip_arm_joints(joints)

    def ik_bounded(
        self, target_pose: FloatArray, init_joints: FloatArray, gripper: float,
        *, deadline_ns: int,
    ) -> tuple[bool, FloatArray, dict[str, object]]:
        accepted, joints, info = self._ik_bounded_task(
            target_pose, init_joints, gripper, deadline_ns=deadline_ns,
            orientation_cost=1.0, orientation_threshold=self._ori_threshold,
        )
        if accepted or self._recovery_ori_threshold is None or time.monotonic_ns() >= deadline_ns:
            return accepted, joints, info
        strict_info = dict(info)
        # Keep the position tolerance and physical joint bounds. Only soften
        # orientation, and reject branch changes relative to this waypoint's seed.
        for cost in (0.1, 0.02):
            if time.monotonic_ns() >= deadline_ns:
                break
            ok, candidate, recovery = self._ik_bounded_task(
                target_pose, joints, gripper, deadline_ns=deadline_ns,
                orientation_cost=cost, orientation_threshold=self._recovery_ori_threshold,
            )
            delta = float(np.max(np.abs(candidate - np.asarray(init_joints))))
            info = {**recovery, "strict_result": strict_info, "recovery_used": True,
                    "recovery_max_joint_delta_rad": delta}
            if ok and delta <= self._recovery_max_joint_delta:
                info["reason"] = "recovered_orientation"
                return True, candidate, info
            if ok:
                info["reason"] = "recovery_joint_delta_exceeded"
        # An unconstrained recovery can leave the seed's allowed neighborhood
        # even when a valid nearby solution exists. Search inside that region
        # before rejecting the waypoint, keeping successful existing solves intact.
        seed = self.clip_arm_joints(init_joints)
        lower, upper = self.joint_position_limits()
        bounds = (np.maximum(lower, seed - self._recovery_max_joint_delta),
                  np.minimum(upper, seed + self._recovery_max_joint_delta))
        for cost in (0.1, 0.02):
            if time.monotonic_ns() >= deadline_ns:
                break
            ok, candidate, recovery = self._ik_bounded_task(
                target_pose, seed, gripper, deadline_ns=deadline_ns,
                orientation_cost=cost, orientation_threshold=self._recovery_ori_threshold,
                joint_bounds=bounds,
            )
            delta = float(np.max(np.abs(candidate - np.asarray(init_joints))))
            info = {**recovery, "strict_result": strict_info, "recovery_used": True,
                    "recovery_method": "seed_box",
                    "recovery_max_joint_delta_rad": delta}
            if ok and delta <= self._recovery_max_joint_delta:
                info["reason"] = "recovered_orientation"
                return True, candidate, info
        return False, joints, info

    def _ik_bounded_task(
        self, target_pose: FloatArray, init_joints: FloatArray, gripper: float,
        *, deadline_ns: int, orientation_cost: float, orientation_threshold: float,
        joint_bounds: tuple[FloatArray, FloatArray] | None = None,
    ) -> tuple[bool, FloatArray, dict[str, object]]:
        """Same Mink objective/limits, with a cooperative deadline and stall exit.

        A single native QP cannot be preempted here; the decoder process client
        separately bounds how long publication waits for that process.
        """
        import mink

        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("target_pose must be a finite 4x4 transform")
        qpos = self.robot_state_to_qpos(self.clip_arm_joints(init_joints), gripper)
        reason, iterations = "max_iters", 0
        with self._lock:
            if not hasattr(self, "_bounded_configuration"):
                self._bounded_configuration = mink.Configuration(self.model)
            configuration = self._bounded_configuration
            configuration.update(qpos)
            task = mink.FrameTask(
                frame_name=EE_SITE, frame_type="site", position_cost=1.0,
                orientation_cost=orientation_cost, lm_damping=1.0,
            )
            task.set_target(mink.SE3.from_matrix(target))
            limits = self._joint_limits()
            if joint_bounds is not None:
                local_limit = mink.ConfigurationLimit(self.model)
                local_limit.lower[:self._num_arm_joints] = joint_bounds[0]
                local_limit.upper[:self._num_arm_joints] = joint_bounds[1]
                limits = [local_limit]
            best, stale = float("inf"), 0
            for iteration in range(self._max_iters + 1):
                error = task.compute_error(configuration)
                pos, rot = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
                # Like the original i2rt solver, take at least one update. Using
                # the convergence tolerance as a seed deadband quantizes small
                # moving targets into repeated holds followed by larger jumps.
                if iteration > 0 and pos <= self._pos_threshold and rot <= orientation_threshold:
                    reason = "converged"
                    break
                if time.monotonic_ns() >= deadline_ns:
                    reason = "budget_exceeded"
                    break
                if iteration == self._max_iters:
                    break
                score = max(pos / self._pos_threshold, rot / orientation_threshold)
                if score < best * (1.0 - 1e-4):
                    best, stale = score, 0
                else:
                    stale += 1
                if stale >= 12:
                    reason = "stagnation"
                    break
                velocity = mink.solve_ik(configuration, [task], .01, "quadprog",
                                         damping=1e-4, limits=limits)
                configuration.integrate_inplace(velocity, .01)
                iterations += 1
            joints = self.clip_arm_joints(configuration.q[:self._num_arm_joints])
            if joint_bounds is not None:
                joints = np.clip(joints, *joint_bounds)
        # Validate the actual clipped command, not the solver's unclipped state.
        pos, rot = self.pose_error(target, joints, gripper)
        converged = pos <= self._pos_threshold and rot <= orientation_threshold
        if converged:
            reason = "converged"
        elif reason == "converged":
            reason = "clipped_residual"
        return converged, joints, {
            "reason": reason, "iterations": iterations,
            "position_error_m": pos, "rotation_error_rad": rot,
            "joint_limit_margin_rad": float(np.min(self.joint_limit_margins(joints))),
        }
