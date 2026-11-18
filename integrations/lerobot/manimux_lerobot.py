"""Local LeRobot policy binding, discovered inside the ManiMux policy worker."""

from pathlib import Path

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from manimux.policies.capabilities import PolicyCapabilities
from manimux.policy_adapter.base import require_request_type
from manimux.types import InferenceRequest


class SmolVLAWorker:
    def __init__(self, config):
        options = config["options"]
        if options["action_semantics"] != "absolute_joint_position":
            raise ValueError("SmolVLA integration requires an absolute joint checkpoint")
        checkpoint = Path(options["checkpoint"]).expanduser().resolve(strict=True)
        device = config["device"]
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(device).eval()
        model = self.policy.config
        self.group = options["group"]
        self.cameras = dict(config["adapter"]["camera_map"])
        self.scale = np.asarray(options["model_units_per_robot_unit"], dtype=np.float32)
        self.offset = np.asarray(options["model_zero"], dtype=np.float32)
        width = tuple(model.input_features["observation.state"].shape)
        if (
            len(width) != 1
            or tuple(model.output_features["action"].shape) != width
            or self.scale.shape != width
            or self.offset.shape != width
            or not np.isfinite(self.scale).all()
            or np.any(self.scale == 0)
            or not np.isfinite(self.offset).all()
        ):
            raise ValueError("state, absolute action and joint calibration dimensions must match")
        visuals = model.image_features
        if (
            not self.cameras
            or set(self.cameras) != set(visuals)
            or len(set(self.cameras.values())) != len(self.cameras)
            or set(model.input_features) != {"observation.state", *visuals}
            or model.n_obs_steps != 1
            or model.use_delta_joint_actions_aloha
        ):
            raise ValueError(
                "require one current state and a mapped RGB stream per checkpoint camera"
            )
        self.shapes = {key: tuple(feature.shape) for key, feature in visuals.items()}
        self.steps = config["horizon_policy_steps"]
        if not 2 <= self.steps <= model.chunk_size:
            raise ValueError("configured horizon must fit the checkpoint action chunk")
        self.pre, self.post = make_pre_post_processors(
            model,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.identity = {
            "server": "lerobot_local",
            "model": {
                "type": "smolvla",
                "checkpoint": checkpoint.name,
                "chunk_size": model.chunk_size,
            },
        }
        self.session = None

    def capabilities(self):
        return PolicyCapabilities(backend_metadata=self.identity)

    def reset(self, session_id):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()
        self.session = session_id

    @torch.inference_mode()
    def infer(self, request):
        require_request_type(request, (InferenceRequest,), "LeRobot SmolVLA")
        if self.session is None or request.session_id != self.session:
            raise ValueError("request belongs to a different session")
        groups = request.observation.state.groups
        if set(groups) != {self.group}:
            raise ValueError("checkpoint requires exactly its configured joint group")
        state = np.asarray(groups[self.group], dtype=np.float32)
        if state.shape != self.scale.shape or not np.isfinite(state).all():
            raise ValueError("invalid measured state")
        batch = {
            "observation.state": torch.from_numpy(state * self.scale + self.offset),
            "task": request.instruction,
        }
        for key, source in self.cameras.items():
            rgb = request.observation.frames[source].data
            shape = self.shapes[key]
            if rgb.dtype != np.uint8 or rgb.shape != (shape[1], shape[2], shape[0]):
                raise ValueError(f"{source} requires RGB size {shape}")
            batch[key] = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
        predicted = self.policy.predict_action_chunk(self.pre(batch))
        if (
            predicted.ndim != 3
            or predicted.shape[0] != 1
            or predicted.shape[1] < self.steps
            or predicted.shape[2] != self.scale.size
        ):
            raise ValueError("invalid checkpoint action chunk")
        actions = torch.cat([self.post(row) for row in predicted[:, : self.steps].unbind(1)])
        joints = (actions.cpu().numpy() - self.offset) / self.scale
        if not np.isfinite(joints).all():
            raise ValueError("non-finite checkpoint action")
        return {
            "format": "joint",
            "action_semantics": "absolute_joint_position",
            "actions": {self.group: joints},
        }

    def close(self):
        self.session = None
