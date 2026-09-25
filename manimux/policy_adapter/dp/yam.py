"""Absolute per-arm-base DP poses and measured three-frame YAM history."""

import uuid
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.policies.actions import action_groups
from manimux.policy_adapter.kinematics import KinematicAdapter
from manimux.policy_adapter.umi_dp.history import MeasuredHistory
from manimux.types import ActionChunk, InferenceRequest, ObservationSnapshot, SensorFrame


@dataclass(slots=True)
class HistorySnapshot(ObservationSnapshot):
    samples: tuple = ()


@dataclass(slots=True)
class DPRequest(InferenceRequest):
    model_info: dict = field(default_factory=dict)


class HistoryStrategy:
    """Supply camera/state samples at 30 Hz to the existing serial scheduler."""

    discard_plans_while_paused = True

    def __init__(self, config):
        from manimux.runtime.inference import build_inference_strategy

        delegate = deepcopy(config)
        delegate["inference"]["strategy"] = None
        self.delegate = build_inference_strategy(delegate)
        timing = config["inference"]["history"]
        self.history = MeasuredHistory(
            camera_names=config["policy"]["adapter"]["physical_cameras"],
            period_s=timing["period_s"],
            tolerance_s=timing["period_tolerance_s"],
            state_tolerance_s=timing["state_tolerance_s"],
            camera_skew_s=timing["camera_skew_s"],
            max_age_s=timing["max_age_s"],
        )

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def reset(self):
        self.history.reset()
        self.delegate.reset()

    def build_submission(self, **kwargs):
        self.history.observe(kwargs["snapshot"])
        samples = list(self.history.samples)
        if len(samples) < 3:
            return None
        newest = samples[-1]
        if kwargs["now_ns"] - newest.state.monotonic_ns > self.history.max_age_ns:
            return None
        selected = []
        for back in (2, 1, 0):
            target = newest.state.monotonic_ns - back * self.history.period_ns
            sample = min(samples, key=lambda s: abs(s.state.monotonic_ns - target))
            if abs(sample.state.monotonic_ns - target) > self.history.tolerance_ns:
                return None
            selected.append(sample)
        for before, after in zip(selected, selected[1:], strict=False):
            for name in self.history.names:
                delta = (
                    after.frames[name].capture_monotonic_ns
                    - before.frames[name].capture_monotonic_ns
                )
                if abs(delta - self.history.period_ns) > self.history.tolerance_ns:
                    return None
        window = HistorySnapshot(newest.state, newest.frames, tuple(selected))
        return self.delegate.build_submission(**{**kwargs, "snapshot": window})


class DPYamAdapter(KinematicAdapter):
    def __init__(self, robot, policy, *, kinematics=None):
        super().__init__(robot, policy, kinematics=kinematics)

        self.groups = ("left_arm", "right_arm")
        self.dt = round(policy["action_dt_s"] * 1e9)
        self.horizon = policy["horizon_policy_steps"]
        self.camera_map = policy["adapter"]["camera_map"]
        self.anchors = {}
        if robot["group_dims"] != dict(left_arm=7, right_arm=7):
            raise ValueError("DP YAM requires two 6+1 joint groups")

    def prepare_request(self, request):
        import cv2

        samples = request.observation.samples
        if len(samples) != 3:
            raise ValueError("DP requires measured three-frame history")
        frames, states = {}, []
        for i, sample in enumerate(samples):
            state = {}
            for side, group in zip(("left", "right"), self.groups, strict=True):
                q = sample.state.groups[group]
                pose = self._fk(group, q[:6], float(q[6]))
                quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
                state[f"{side}_ee_pose"] = np.r_[pose[:3, 3], quat[3], quat[:3]]
                state[f"{side}_ee_joint_state"] = np.asarray(q[6:7])
            states.append(state)
            for model_name, source in self.camera_map.items():
                if not model_name.endswith(f"_t{i}"):
                    continue
                physical = source.rsplit("_t", 1)[0]
                frame = sample.frames[physical]
                frames[source] = SensorFrame(
                    source,
                    cv2.resize(frame.data, (320, 240), interpolation=cv2.INTER_AREA),
                    frame.capture_monotonic_ns,
                    frame.sequence,
                )
        self.anchors[request.request_seq] = request.observation.state
        while len(self.anchors) > 8:
            self.anchors.pop(next(iter(self.anchors)))
        return DPRequest(
            request.session_id,
            request.request_seq,
            request.observation_time_ns,
            request.deadline_ns,
            ObservationSnapshot(request.observation.state, frames),
            request.instruction,
            {"dp_history_states": states},
        )

    def decode_action(self, raw, context):
        actions = action_groups(raw, {g: 8 for g in self.groups}, format="pose")
        if any(len(rows) != self.horizon for rows in actions.values()):
            raise ValueError("DP action horizon mismatch")
        anchor = self.anchors.pop(context.request_seq, None)
        seed_state = context.measured_state if context.measured_state is not None else anchor
        if seed_state is None:
            raise ValueError("DP IK requires a measured seed")
        groups = {}
        for group in self.groups:
            seed = np.asarray(seed_state.groups[group][:6]).copy()
            rows = []
            for action in actions[group]:
                pose = np.asarray(action[:7], dtype=float)
                grip = np.asarray(action[7:], dtype=float)
                if (
                    pose.shape != (7,)
                    or grip.shape != (1,)
                    or not np.isfinite(np.r_[pose, grip]).all()
                ):
                    raise ValueError("DP returned an invalid EEF action")
                transform = np.eye(4)
                transform[:3, 3] = pose[:3]
                transform[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                aperture = float(np.clip(grip[0], 0, 1))
                ok, joints = self._ik(group, transform, seed, aperture)
                if not ok:
                    raise ValueError(f"DP IK failed for {group} at step {len(rows)}")
                seed = joints
                rows.append(np.r_[joints, aperture])
            groups[group] = np.asarray(rows)
        return ActionChunk(
            f"dp-{uuid.uuid4().hex[:8]}",
            context.request_seq,
            context.observation_time_ns,
            context.created_time_ns,
            "joint_position",
            self.dt,
            groups,
        )
