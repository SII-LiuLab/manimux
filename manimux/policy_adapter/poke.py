"""Calibrated Poke flange targets through the configured robot FK and IK."""

import uuid
from dataclasses import dataclass, field, fields

import numpy as np

from manimux.kinematics.poses import matrix_pose, pose_matrix
from manimux.policies.base import action_interval
from manimux.policies.poke.codec import pose_rows, require_default_sampling
from manimux.policy_adapter.kinematics import KinematicAdapter
from manimux.policy_adapter.pose_output import pose_chunk
from manimux.types import ActionChunk, InferenceRequest


@dataclass(slots=True)
class PokeRequest(InferenceRequest):
    model_state: list = field(default_factory=list)


class PokePoseAdapter(KinematicAdapter):
    """One controlled arm; other groups hold the runtime's selected reference.

    All transform poses use ManiMux's xyz/wxyz convention. The API uses xyzw.
    The orientation alignment is rotation-only, keeping the control point at
    the physical flange, as in RoboAPI's Home/plane calibration.
    """

    supports_step_ik = True
    supports_context_only_decode = True
    supports_multi_inflight = False

    def __init__(self, robot, policy, *, kinematics=None):
        super().__init__(robot, policy, kinematics=kinematics)
        options = policy["adapter"]
        self.group = options["controlled_group"]
        self.camera = options["camera"]
        if self.group not in robot["group_dims"]:
            raise ValueError("Poke controlled_group is not in the robot assembly")
        model = self.kinematics.models[self.group]
        if model.coordinates[-1].unit != "normalized":
            raise ValueError("Poke requires a trailing normalized gripper coordinate")
        transforms = {}
        for name in ("model_from_base", "flange_from_tcp", "flange_from_model_tool"):
            if options.get(name) is None:
                raise ValueError(
                    f"Poke adapter requires calibrated {name}; no implicit identity transform"
                )
            transforms[name] = pose_matrix(options[name])
        self.model_from_base = transforms["model_from_base"]
        self.flange_from_tcp = transforms["flange_from_tcp"]
        self.flange_from_model_tool = transforms["flange_from_model_tool"]
        if not np.allclose(self.flange_from_model_tool[:3, 3], 0, atol=1e-12):
            raise ValueError(
                "flange_from_model_tool must be rotation-only; Poke controls the flange centre"
            )
        self.base_from_model = np.linalg.inv(self.model_from_base)
        self.tcp_from_model_tool = np.linalg.inv(self.flange_from_tcp) @ self.flange_from_model_tool
        self.model_tool_from_tcp = np.linalg.inv(self.tcp_from_model_tool)
        self.dt_ns = round(action_interval(policy) * 1e9)
        self.max_rows = policy["horizon_policy_steps"]

    def build_observation(self, snapshot):
        if self.camera not in snapshot.frames:
            raise ValueError(f"Poke wrist camera {self.camera!r} is missing")
        return snapshot

    def prepare_request(self, request):
        require_default_sampling(request)
        self.build_observation(request.observation)
        q = request.observation.state.groups[self.group]
        if not 0 <= q[-1] <= 1:
            raise ValueError("Poke observed gripper must be in [0, 1]")
        base_tcp = self.kinematics.models[self.group].fk(q)
        pose = matrix_pose(self.model_from_base @ base_tcp @ self.tcp_from_model_tool)
        state = np.r_[pose[:3], pose[4:7], pose[3], q[-1]]
        return PokeRequest(
            **{item.name: getattr(request, item.name) for item in fields(InferenceRequest)},
            model_state=state.tolist(),
        )

    def decode_action(self, raw, context):
        if (
            not isinstance(raw, dict)
            or raw.get("format") != "native"
            or raw.get("action_semantics") != "absolute_model_flange_xyz_xyzw"
        ):
            raise ValueError("Poke action representation/semantics mismatch")
        actions = pose_rows(raw["actions"], label="Poke adapter output")
        if len(actions) > self.max_rows:
            raise ValueError(
                f"Poke delivered {len(actions)} rows above configured capacity {self.max_rows}"
            )
        if np.any((actions[:, 7] < 0) | (actions[:, 7] > 1)):
            raise ValueError("Poke adapter requires codec-normalized gripper values")
        if context.ik_seed is None:
            raise ValueError("Poke IK requires the runtime-selected seed")
        groups = {}
        for group, model in self.kinematics.models.items():
            seed = np.array(context.ik_seed.groups[group], dtype=float, copy=True)
            if group != self.group:
                row = (
                    np.r_[matrix_pose(model.fk(seed)), seed[-1]] if self.decode_pose_only else seed
                )
                groups[group] = np.repeat(row[None], max(2, len(actions)), axis=0)
                continue
            solved = []
            for step, row in enumerate(actions):
                model_pose = pose_matrix(row[[0, 1, 2, 6, 3, 4, 5]])
                target = self.base_from_model @ model_pose @ self.model_tool_from_tcp
                if self.decode_pose_only:
                    solved.append(np.r_[matrix_pose(target), row[7]])
                    continue
                result = model.ik(
                    target, seed, fixed_coordinates={model.coordinates[-1].name: float(row[7])}
                )
                if not result.converged:
                    log = raw.get("sampling_stats", {}).get("poke_api", {}).get("response_log")
                    raise ValueError(
                        f"Poke IK failed: group={group} row={step} reason={result.reason}; "
                        f"response_log={log}"
                    )
                seed = np.asarray(result.joints).copy()
                solved.append(seed)
            groups[group] = np.asarray(solved)
            if len(solved) == 1:
                groups[group] = np.repeat(groups[group], 2, axis=0)
        metadata = {
            "controlled_group": self.group,
            "raw_model_steps": len(actions),
            "held_groups": [g for g in groups if g != self.group],
            "raw_model_eef": {self.group: actions.tolist()},
            "raw_model_eef_frame": "poke_model_flange_xyz_xyzw",
            "terminal_hold_endpoint": len(actions) == 1,
        }
        if self.decode_pose_only:
            return pose_chunk(context, groups, self.dt_ns, metadata=metadata)
        return ActionChunk(
            f"poke-{context.request_seq}-{uuid.uuid4().hex[:8]}",
            context.request_seq,
            context.observation_time_ns,
            context.created_time_ns,
            "joint_position",
            self.dt_ns,
            groups,
            metadata=metadata,
        )
