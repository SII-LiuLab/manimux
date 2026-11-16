"""InternW0-Delta on a joint-controlled robot: absolute joints plus anchor/recent memory."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from manimux.policy_adapter.internw0.memory import MemorySnapshot
from manimux.policy_adapter.joint import JointAdapter
from manimux.types import InferenceRequest, ObservationSnapshot


@dataclass(slots=True)
class InternW0Request(InferenceRequest):
    model_info: dict = field(default_factory=dict)
    sampling_request: InferenceRequest | None = None


class InternW0Adapter(JointAdapter):
    """JointAdapter that also ships the anchor and recent frames with each request.

    ``camera_map`` lists the live cameras plus ``<physical>@anchor`` and ``<physical>@recent``
    entries; only the live cameras are required of the sensor snapshot.
    """

    def __init__(self, robot, policy, *, kinematics=None):
        super().__init__(robot, policy, kinematics=kinematics)
        physical = tuple(policy["adapter"]["physical_cameras"])
        self._physical = physical
        self._required_cameras = physical
        self._group_order = tuple(robot["group_dims"])
        self._camera_map_keys = dict(policy["adapter"]["camera_map"])
        for tag in ("anchor", "recent"):
            expected = {f"{name}@{tag}" for name in physical}
            if not expected <= set(self._camera_map_keys.values()):
                raise ValueError(f"camera_map must map the {tag} frames {sorted(expected)}")

    def _vector(self, state) -> list[float]:
        return np.concatenate([state.groups[name] for name in self._group_order]).tolist()

    def prepare_request(self, request):
        source = request
        request = super().prepare_request(request)
        snapshot = request.observation
        if not isinstance(snapshot, MemorySnapshot) or snapshot.anchor is None:
            raise ValueError("InternW0 requires the memory inference strategy")
        frames = dict(snapshot.frames)
        for tag in ("anchor", "recent"):
            history = getattr(snapshot, tag)
            for name in self._physical:
                frames[f"{name}@{tag}"] = history.frames[name]
        info = {
            "internw0": {
                "episode_step": int(snapshot.episode_step),
                "recent_is_real": bool(snapshot.recent_is_real),
                "anchor_state": self._vector(snapshot.anchor.state),
                "recent_state": self._vector(snapshot.recent.state),
            }
        }
        return InternW0Request(
            request.session_id,
            request.request_seq,
            request.observation_time_ns,
            request.deadline_ns,
            ObservationSnapshot(snapshot.state, frames),
            request.instruction,
            model_info=info,
            sampling_request=source,
        )


__all__ = ["InternW0Adapter", "InternW0Request"]
