"""Canonical TCP poses to joints through the configured offline robot geometry."""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from manimux.embodiments.layout import group_layouts
from manimux.kinematics.poses import matrix_pose, pose_matrix
from manimux.policies.actions import action_groups
from manimux.policies.base import action_interval
from manimux.policy_adapter.gripper_mapping import clip_aperture
from manimux.policy_adapter.kinematics import KinematicAdapter
from manimux.policy_adapter.pose_output import pose_chunk
from manimux.types import ActionChunk, InferenceRequest


@dataclass(slots=True)
class PoseRequest(InferenceRequest):
    model_state: dict = field(default_factory=dict)


class PoseAdapter(KinematicAdapter):
    """Absolute or explicitly anchored delta poses with the shared IK failure policy.

    Base deltas translate in the configured arm base and left-multiply rotation.
    Tool deltas right-multiply the observation TCP transform. Observation deltas
    share one anchor across the chunk. Per-step feedback deltas retain one row:
    future measured poses are unavailable when decoding a complete chunk.
    """

    supports_context_only_decode = True

    supports_step_ik = True

    def __init__(self, robot, policy, *, kinematics=None):
        super().__init__(robot, policy, kinematics=kinematics)
        options = policy["adapter"]
        self.gripper_output = options.get("gripper_output", "reject")
        if self.gripper_output not in {"reject", "clip"}:
            raise ValueError("Pose gripper_output must be reject or clip")
        self.layouts = group_layouts(robot["group_dims"], options)
        if any(layout["gripper_dofs"] not in (0, 1) for layout in self.layouts.values()):
            raise ValueError("PoseAdapter supports zero or one normalized gripper per arm")
        self.prefixes = options["group_prefixes"]
        parallel = options.get("parallel_ik", False)
        if type(parallel) is not bool:
            raise ValueError("parallel_ik must be a boolean")
        if parallel and policy.get("action_decoding", "inline") != "process":
            raise ValueError("parallel_ik requires process action decoding")
        self.decode_partitions = tuple(self.layouts) if parallel else ()
        expected = {""} if len(self.layouts) == 1 else {"left", "right"}
        if set(self.prefixes) != set(self.layouts) or set(self.prefixes.values()) != expected:
            raise ValueError(
                "group_prefixes must map one group to an empty prefix, or two groups to left/right"
            )
        self.semantics = options["action_semantics"]
        if self.semantics not in {
            "absolute_per_arm_base_xyz_wxyz",
            "delta_observation_base_xyz_wxyz",
            "delta_observation_tool_xyz_wxyz",
            "delta_step_base_xyz_wxyz",
        }:
            raise ValueError("Unsupported pose action_semantics")
        self.supports_context_only_decode = self.semantics != "delta_step_base_xyz_wxyz"
        self.decode_seed_source = (
            "observation_state"
            if self.semantics.startswith("delta_observation_")
            else "execution_reference"
        )
        # The paired service declares pose semantics in its handshake metadata.
        if (policy.get("expected_backend") or {}).get("model", {}).get(
            "action_semantics"
        ) != self.semantics:
            raise ValueError("expected_backend.model.action_semantics must match the pose adapter")
        self.horizon = policy["horizon_policy_steps"]
        self.decode_steps = options.get("decode_policy_steps", self.horizon)
        if type(self.decode_steps) is not int or not 0 < self.decode_steps <= self.horizon:
            raise ValueError("Invalid pose decode_policy_steps")
        if self.semantics == "delta_step_base_xyz_wxyz" and self.decode_steps != 1:
            raise ValueError(
                "Per-step feedback deltas require decode_policy_steps=1 and replanning"
            )
        if (
            self.semantics == "delta_step_base_xyz_wxyz"
            and policy.get("action_decoding", "inline") != "inline"
        ):
            raise ValueError("Per-step feedback deltas require inline decoding from measured state")
        self.dt_ns = round(action_interval(policy) * 1e9)
        self.required_cameras = tuple(options.get("camera_map", {}).values())
        for group, dim in robot["group_dims"].items():
            coordinates = self.kinematics.models[group].coordinates
            has_gripper = self.layouts[group]["gripper_dofs"] == 1
            if len(coordinates) != dim or (has_gripper and coordinates[-1].unit != "normalized"):
                raise ValueError(
                    f"{group} geometry must match the declared arm/tool coordinate layout"
                )
        warmup = options.get("ik_warmup_joints")
        self._warmup_joints = {}
        if warmup is not None:
            if not isinstance(warmup, Mapping) or set(warmup) != set(self.layouts):
                raise ValueError("ik_warmup_joints must provide every robot group")
            for group, values in warmup.items():
                q = np.array(values, dtype=float, copy=True)
                if (
                    q.shape != (robot["group_dims"][group],)
                    or not np.isfinite(q).all()
                    or (self.layouts[group]["gripper_dofs"] == 1 and not 0 <= q[-1] <= 1)
                ):
                    raise ValueError(f"Invalid ik_warmup_joints for {group}")
                self._warmup_joints[group] = q
            if policy.get("action_decoding", "inline") == "inline":
                self.warmup_decode(None)

    def warmup_decode(self, partition):
        """Initialize offline IK before control starts; never send a robot command."""
        groups = (partition,) if partition is not None else tuple(self.layouts)
        for group in groups:
            if group not in self.layouts:
                raise ValueError(f"Unknown pose decode partition: {group}")
            if group not in self._warmup_joints:
                continue
            model = self.kinematics.models[group]
            seed = self._warmup_joints[group]
            result = model.ik(
                model.fk(seed),
                seed.copy(),
                fixed_coordinates=(
                    {model.coordinates[-1].name: float(seed[-1])}
                    if self.layouts[group]["gripper_dofs"]
                    else {}
                ),
            )
            if not result.converged:
                raise ValueError(f"Pose IK warmup failed: group={group} reason={result.reason}")

    def build_observation(self, snapshot):
        missing = set(self.required_cameras) - set(snapshot.frames)
        if missing:
            raise ValueError(f"Pose adapter is missing cameras: {sorted(missing)}")
        return snapshot

    def prepare_request(self, request):
        if type(request) is not InferenceRequest:
            raise ValueError("PoseAdapter supports default inference requests only")
        poses = self.kinematics.fk(request.observation.state.groups)
        state = {}
        for group, prefix in self.prefixes.items():
            key = f"{prefix}_ee_pose" if prefix else "ee_pose"
            state[key] = matrix_pose(poses[group])
        return PoseRequest(
            request.session_id,
            request.request_seq,
            request.observation_time_ns,
            request.deadline_ns,
            request.observation,
            request.instruction,
            state,
        )

    def decode_action(self, raw, context):
        return self._decode_action(raw, context, tuple(self.layouts))

    def decode_action_partition(self, raw, context, partition):
        if partition not in self.decode_partitions:
            raise ValueError(f"Unknown pose decode partition: {partition}")
        return self._decode_action(raw, context, (partition,))

    def _decode_action(self, raw, context, selected_groups):
        actions = action_groups(
            raw,
            {group: 7 + layout["gripper_dofs"] for group, layout in self.layouts.items()},
            format="pose",
        )
        if raw.get("action_semantics", self.semantics) != self.semantics:
            raise ValueError("Pose action semantics mismatch")
        if any(len(rows) != self.horizon for rows in actions.values()):
            raise ValueError("Pose action horizon mismatch")
        gripper_metadata = {}
        raw_actions = actions
        if self.gripper_output == "clip":
            apertures, gripper_metadata = clip_aperture(
                {
                    name: rows[:, 7]
                    for name, rows in actions.items()
                    if self.layouts[name]["gripper_dofs"]
                }
            )
            actions = {name: rows.copy() for name, rows in actions.items()}
            for name, values in apertures.items():
                actions[name][:, 7] = values
        if context.ik_seed is None:
            raise ValueError("Pose IK requires a decode seed")
        if self.semantics.startswith("delta_observation_") and context.observation_state is None:
            raise ValueError("Observation-relative poses require the request observation state")
        groups, diagnostics = {}, {}
        for group in selected_groups:
            rows = actions[group]
            model = self.kinematics.models[group]
            seed = np.array(context.ik_seed.groups[group], dtype=float, copy=True)
            anchor = model.fk(
                context.observation_state.groups[group]
                if self.semantics.startswith("delta_observation_")
                else seed
            )
            sequence = self.ik_sequence(group, seed)
            solved = []
            for row in rows[: self.decode_steps]:
                target = pose_matrix(row[:7])
                fixed = {}
                if self.layouts[group]["gripper_dofs"]:
                    aperture = float(row[7])
                    if not 0 <= aperture <= 1:
                        raise ValueError("Pose gripper must be a normalized opening in [0, 1]")
                    fixed[model.coordinates[-1].name] = aperture
                if self.semantics == "delta_observation_tool_xyz_wxyz":
                    target = anchor @ target
                elif self.semantics in {
                    "delta_observation_base_xyz_wxyz",
                    "delta_step_base_xyz_wxyz",
                }:
                    target[:3, 3] += anchor[:3, 3]
                    target[:3, :3] = target[:3, :3] @ anchor[:3, :3]
                if self.decode_pose_only:
                    if not self.layouts[group]["gripper_dofs"]:
                        raise ValueError("Pose-only decoding requires a normalized gripper")
                    solved.append(np.r_[matrix_pose(target), aperture])
                    continue
                result = model.ik(
                    target,
                    sequence.current,
                    fixed_coordinates=fixed,
                )
                seed = sequence.resolve(result.converged, result.joints, reason=result.reason)
                solved.append(seed)
            diagnostics[group] = sequence.metadata()
            groups[group] = np.asarray(solved)
            if len(solved) == 1:
                # A timeline segment needs two endpoints. Hold the one decoded
                # command for one action interval; do not consume another delta.
                groups[group] = np.repeat(groups[group], 2, axis=0)
        if self.decode_pose_only:
            return pose_chunk(context, groups, self.dt_ns, metadata=gripper_metadata)
        return ActionChunk(
            f"pose-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            context.request_seq,
            context.observation_time_ns,
            context.created_time_ns,
            "joint_position",
            self.dt_ns,
            groups,
            metadata={
                **gripper_metadata,
                "ik": diagnostics,
                "action_semantics": self.semantics,
                "decoded_policy_steps": self.decode_steps,
                "terminal_hold_endpoint": self.decode_steps == 1,
                "raw_model_eef": {
                    group: raw_actions[group][: self.decode_steps].tolist()
                    for group in selected_groups
                },
            },
        )
