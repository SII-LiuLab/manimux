"""Prepare agent motion tools with shared kinematics; never own hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.kinematics.motion import JointPathPlanner, PoseMotionLimits, pose_error
from manimux.plugins import load_plugin
from manimux.types import ObservationSnapshot


@dataclass
class ToolMotion:
    groups: dict[str, np.ndarray]
    group: str
    target: np.ndarray
    gripper: float | None
    dt_ns: int
    observation_time_ns: int
    contact_since: float | None = None


class MotionTools:
    """Group-based TCP tools in each configured model's named base frame.

    A path is planned from the last command, while relative goals use measured
    feedback. Workspace bounds and optional whole-robot checks apply before
    admission. This planner has no device connection or execution loop.
    """

    def __init__(self, kinematics, options: dict[str, Any], control_hz: float):
        self.models = kinematics.models
        self.options = options
        self.dt_ns = int(1e9 / control_hz)
        self.max_waypoints = int(options.get("max_waypoints", 2000))
        self.settle_timeout_s = float(options.get("settle_timeout_s", 5.0))
        self.position_tolerance_m = float(options.get("position_tolerance_m", 0.002))
        self.rotation_tolerance_rad = float(options.get("rotation_tolerance_rad", 0.02))
        self.gripper_tolerance = float(options.get("gripper_tolerance", 0.03))
        values = [
            self.settle_timeout_s,
            self.position_tolerance_m,
            self.rotation_tolerance_rad,
            self.gripper_tolerance,
        ]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("motion arrival tolerances/timeouts must be finite and positive")
        groups = options.get("groups", {})
        if set(groups) != set(self.models):
            raise ValueError("motion.groups must configure every kinematic group")
        self.planners, self.grippers = {}, {}
        for name, model in self.models.items():
            spec = groups[name]
            grippers = [i for i, c in enumerate(model.coordinates) if c.unit == "normalized"]
            if (
                len(grippers) != 1
                or model.coordinates[grippers[0]].name != spec["gripper_coordinate"]
            ):
                raise ValueError(f"{name}: declare the single normalized gripper coordinate")
            if spec.get("gripper_convention") != "zero_closed_one_open":
                raise ValueError(f"{name}: gripper_convention must be zero_closed_one_open")
            self.grippers[name] = grippers[0]
            self.planners[name] = JointPathPlanner(
                model,
                control_dt_s=self.dt_ns / 1e9,
                workspace_min_m=spec.get("workspace_min_m"),
                workspace_max_m=spec.get("workspace_max_m"),
                limits=PoseMotionLimits(**options.get("limits", {})),
                max_waypoints=self.max_waypoints,
            )
        self.validator = None
        guard = options.get("path_validator")
        if guard is not None:
            factory = load_plugin(guard["factory"], group="manimux.path_validators", builtins={})
            self.validator = factory(kinematics, guard.get("options", {}))

    def describe(self) -> dict[str, Any]:
        return {
            "tools": ["move_to", "move_delta", "rotate", "set_gripper"],
            "groups": {
                name: {
                    "base_frame": model.base_frame,
                    "tcp_frame": model.tcp_frame,
                    "gripper": "0=closed, 1=open",
                }
                for name, model in self.models.items()
            },
            "position_units": "metres",
            "rotation_units": "radians",
            "quaternion_order": "wxyz",
        }

    @staticmethod
    def _vector(value, size, field):
        vector = np.asarray(value, dtype=float)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"{field}: expected {size} finite values; got shape {vector.shape}")
        return vector

    def plan(
        self,
        tool: str,
        arguments: dict[str, Any],
        snapshot: ObservationSnapshot,
        reference: dict[str, np.ndarray],
    ) -> ToolMotion:
        args = dict(arguments)
        group = args.pop("group")
        if group not in self.models:
            raise ValueError(f"unknown motion group {group!r}; expected {list(self.models)}")
        model, planner = self.models[group], self.planners[group]
        index = self.grippers[group]
        seed = reference[group].copy()
        target = model.fk(snapshot.state.groups[group])
        opening = None
        if tool == "move_to":
            target[:3, 3] = self._vector(args.pop("xyz"), 3, "xyz")
            quaternion = args.pop("quaternion_wxyz", None)
            if quaternion is not None:
                q = self._vector(quaternion, 4, "quaternion_wxyz")
                if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-5):
                    raise ValueError("quaternion_wxyz must have unit norm")
                target[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        elif tool == "move_delta":
            target[:3, 3] += self._vector(args.pop("delta_xyz"), 3, "delta_xyz")
        elif tool == "rotate":
            delta = self._vector(args.pop("delta_rpy"), 3, "delta_rpy")
            target[:3, :3] = (
                Rotation.from_euler("xyz", delta) * Rotation.from_matrix(target[:3, :3])
            ).as_matrix()
        elif tool == "set_gripper":
            opening = args.pop("opening")
            if (
                isinstance(opening, bool)
                or not isinstance(opening, int | float)
                or not np.isfinite(opening)
                or not 0 <= opening <= 1
            ):
                raise ValueError("opening: expected finite value in [0,1]; no clipping is applied")
            target = model.fk(seed)
        else:
            raise ValueError(f"unsupported motion tool {tool!r}")
        if args:
            raise ValueError(f"unknown {tool} arguments: {sorted(args)}")
        if opening is None:
            path = planner.plan(
                target, seed, fixed_coordinates={model.coordinates[index].name: float(seed[index])}
            )
        else:
            seed[index] = opening
            path = seed[None]
            planner.validate(path, reference[group])
        groups = {name: np.tile(q, (len(path), 1)) for name, q in reference.items()}
        groups[group] = path
        if self.validator is not None:
            self.validator(groups, reference)
        return ToolMotion(groups, group, target, opening, self.dt_ns, snapshot.state.monotonic_ns)

    def arrival(self, motion: ToolMotion, snapshot: ObservationSnapshot) -> dict[str, Any]:
        name = motion.group
        q = snapshot.state.groups[name]
        pe, re = pose_error(motion.target, self.models[name].fk(q))
        ge = None if motion.gripper is None else abs(float(q[self.grippers[name]]) - motion.gripper)
        feedback = snapshot.state.grippers.get(name)
        closing = (
            motion.gripper == 0.0
            and feedback is not None
            and feedback.contact == "closing"
            and feedback.target_opening == 0.0
        )
        if not closing:
            motion.contact_since = None
        elif motion.contact_since is None:
            motion.contact_since = feedback.timestamp
        contact = bool(closing and feedback.timestamp - motion.contact_since >= 0.1)
        return {
            "reached": pe <= self.position_tolerance_m
            and re <= self.rotation_tolerance_rad
            and (ge is None or ge <= self.gripper_tolerance or contact),
            "closing_contact_confirmed": contact,
            "position_error_m": pe,
            "rotation_error_rad": re,
            "gripper_error": ge,
        }

    def validate_policy_path(self, groups, reference):
        """Apply the configured TCP workspace and whole-robot guard to VLA paths.

        Policy timing and speed limits remain owned by its adapter/executor; the
        slower point-to-point tool speed is not a new model action conversion.
        """
        for name, rows in groups.items():
            planner = self.planners[name]
            for index, q in enumerate(rows):
                xyz = self.models[name].fk(q)[:3, 3]
                if np.any(xyz < planner.lower) or np.any(xyz > planner.upper):
                    raise ValueError(
                        f"{name}.tcp: row {index} outside configured workspace; "
                        f"actual={xyz.tolist()}"
                    )
        if self.validator is not None:
            self.validator(groups, reference)
