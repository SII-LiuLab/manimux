"""LeRobot SmolVLA backend for single-group absolute joint actions."""

from collections.abc import Callable
from functools import partial
from hashlib import file_digest
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from manimux.policies.capabilities import PolicyCapabilities
from manimux.policies.response_log import capture_response
from manimux.policy_adapter.base import require_request_type
from manimux.types import InferenceRequest


class SmolVLAPolicyModel:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        capture_prediction: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        options = config["options"]
        if capture_prediction is None and options.get("response_log_dir") is not None:
            capture_prediction = partial(capture_response, stage="smolvla_prediction")
        self._capture_prediction = capture_prediction
        if options["action_semantics"] != "absolute_joint_position":
            raise ValueError("SmolVLA integration requires an absolute joint checkpoint")
        checkpoint = Path(options["checkpoint"]).expanduser().resolve(strict=True)
        expected_model = (config.get("expected_backend") or {}).get("model") or {}
        if (
            expected_model.get("checkpoint") == "pretrained_model"
            and "checkpoint_path" not in expected_model
            and "checkpoint_sha256" not in expected_model
        ):
            raise ValueError(
                "pretrained_model is ambiguous in policy.expected_backend.model. "
                "Specify checkpoint_path or checkpoint_sha256."
            )
        model_identity: dict[str, object] = {
            "type": "smolvla",
            "checkpoint": checkpoint.name,
            "checkpoint_path": str(checkpoint),
        }
        if "checkpoint_sha256" in expected_model:
            with (checkpoint / "model.safetensors").open("rb") as weights:
                model_identity["checkpoint_sha256"] = file_digest(weights, "sha256").hexdigest()
        device = config["device"]
        checkpoint_config = PreTrainedConfig.from_pretrained(checkpoint)
        # LeRobot places the weights on config.device during loading.
        checkpoint_config.device = device
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=checkpoint_config)
        model = self.policy.config
        self.group = options["group"]
        self.camera_map = dict(config["adapter"]["camera_map"])
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
            not self.camera_map
            or set(self.camera_map) != set(visuals)
            or len(set(self.camera_map.values())) != len(self.camera_map)
            or set(model.input_features) != {"observation.state", *visuals}
            or model.n_obs_steps != 1
            or model.use_delta_joint_actions_aloha
        ):
            raise ValueError(
                "require one current state and a mapped RGB stream per checkpoint camera"
            )
        self.image_shapes: dict[str, tuple[int, int, int]] = {}
        for key, camera in self.camera_map.items():
            shape = visuals[key].shape
            if len(shape) != 3 or shape[0] != 3 or min(shape) <= 0:
                raise ValueError(f"SmolVLA feature {key!r} requires RGB (C, H, W), got {shape}")
            self.image_shapes[camera] = (shape[1], shape[2], shape[0])
        self.horizon_steps = config["horizon_policy_steps"]
        if not 2 <= self.horizon_steps <= model.chunk_size:
            raise ValueError("configured horizon must fit the checkpoint action chunk")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            model,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.identity: dict[str, object] = {
            "server": "lerobot_local",
            "model": {**model_identity, "chunk_size": model.chunk_size},
        }
        self.session_id: str | None = None

    def capabilities(self) -> PolicyCapabilities:
        return PolicyCapabilities(
            backend_metadata=self.identity,
            required_robot_group_dims={self.group: self.scale.size},
            required_camera_shapes=self.image_shapes,
        )

    def reset(self, session_id: str) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.session_id = session_id

    @torch.inference_mode()
    def infer(self, request: InferenceRequest) -> dict[str, object]:
        require_request_type(request, (InferenceRequest,), "LeRobot SmolVLA")
        if self.session_id is None or request.session_id != self.session_id:
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
        for key, source in self.camera_map.items():
            rgb = request.observation.frames[source].data
            shape = self.image_shapes[source]
            if rgb.dtype != np.uint8 or rgb.shape != shape:
                raise ValueError(f"{source} requires RGB size {shape}")
            batch[key] = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
        predicted = self.policy.predict_action_chunk(self.preprocessor(batch))
        if self._capture_prediction is not None:
            payload: object = predicted
            if isinstance(predicted, torch.Tensor):
                payload = {
                    "dtype": str(predicted.dtype),
                    "shape": list(predicted.shape),
                    "values": predicted.tolist(),
                }
            self._capture_prediction({"backend": self.identity, "action_chunk": payload})
        if not isinstance(predicted, torch.Tensor):
            raise ValueError(
                f"SmolVLA action chunk expected a tensor, got {type(predicted).__name__}"
            )
        if (
            predicted.ndim != 3
            or predicted.shape[0] != 1
            or predicted.shape[1] < self.horizon_steps
            or predicted.shape[2] != self.scale.size
        ):
            raise ValueError(
                f"SmolVLA action chunk expected shape (1, at least {self.horizon_steps}, "
                f"{self.scale.size}), got {tuple(predicted.shape)}"
            )
        actions = torch.cat(
            [self.postprocessor(row) for row in predicted[:, : self.horizon_steps].unbind(1)]
        )
        joints = (actions.cpu().numpy() - self.offset) / self.scale
        invalid = ~np.isfinite(joints)
        if invalid.any():
            indices = np.argwhere(invalid)
            raise ValueError(
                f"SmolVLA action group {self.group!r} has {len(indices)} non-finite values "
                f"after conversion at (step, coordinate) indices {indices[:8].tolist()}"
            )
        return {
            "format": "joint",
            "action_semantics": "absolute_joint_position",
            "actions": {self.group: joints},
        }

    def close(self) -> None:
        self.session_id = None
