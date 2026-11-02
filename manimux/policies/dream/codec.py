"""Dream-1's named camera and action fields at the hosted API boundary."""

import base64
import io

import numpy as np
from PIL import Image

CAMERAS = frozenset({"overhead", "wrist_l", "wrist_r"})


def require_default_sampling(request):
    """Refuse sampler extensions instead of silently dropping their conditions."""
    for name in (
        "action_condition",
        "condition_weights",
        "paint_action_prefix",
        "paint_delay_steps",
        "aac_num_samples",
        "bid_num_samples",
    ):
        if getattr(request, name, None) is not None:
            raise ValueError(f"Dream API does not support {name}")
    if getattr(request, "autohorizon", False):
        raise ValueError("Dream API does not support AutoHorizon")


class DreamCodec:
    def __init__(self, config):
        adapter, options = config["adapter"], config["options"]
        self.cameras = dict(adapter["camera_map"])
        self.prefixes = dict(adapter["group_prefixes"])
        if set(self.cameras) != CAMERAS or len(set(self.cameras.values())) != 3:
            raise ValueError("Dream-1 requires distinct overhead, wrist_l and wrist_r cameras")
        if len(self.prefixes) != 2 or set(self.prefixes.values()) != {"left", "right"}:
            raise ValueError("Dream-1 requires one left and one right arm group")
        self.horizon = config["horizon_policy_steps"]
        if self.horizon != 48:
            raise ValueError("Dream-1 beta returns 48 action rows")
        self.size = tuple(options.get("image_size", [320, 176]))
        if len(self.size) != 2 or any(type(v) is not int or v <= 0 for v in self.size):
            raise ValueError("image_size must be [width, height], both positive integers")
        self.quality = options.get("jpeg_quality", 85)
        if type(self.quality) is not int or not 1 <= self.quality <= 95:
            raise ValueError("jpeg_quality must be an integer in [1, 95]")

    def encode(self, request):
        require_default_sampling(request)
        if set(request.observation.state.groups) != set(self.prefixes):
            raise ValueError("Dream observation groups differ from the configured groups")
        state = getattr(request, "model_state", None)
        if not isinstance(state, dict):
            raise ValueError("Dream API requires DreamYamJointAdapter.prepare_request")
        expected = {
            f"state.{side}_{part}": width
            for side in ("left", "right")
            for part, width in (
                ("arm", 6),
                ("gripper", 1),
                ("grip", 1),
                ("eef_xyz", 3),
                ("eef_rot6", 6),
            )
        }
        if set(state) != set(expected):
            raise ValueError("Dream model state is missing required named fields")
        for key, width in expected.items():
            values = np.asarray(state[key], dtype=float)
            if values.shape != (width,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid Dream state: {key}")
        frames = {}
        for role, name in self.cameras.items():
            frame = request.observation.frames[name]
            image = Image.fromarray(frame.data).resize(self.size, Image.Resampling.BILINEAR)
            stream = io.BytesIO()
            image.save(stream, format="JPEG", quality=self.quality)
            frames[f"video.{role}"] = base64.b64encode(stream.getvalue()).decode("ascii")
        return frames, {**state, "prompt": request.instruction}

    def decode(self, action):
        # The SDK smooths q_*_chunk against its own execution assumptions. per_key
        # retains the model answer; ManiMux alone owns handoff and latency trimming.
        parts = action.get("per_key") if isinstance(action, dict) else None
        if not isinstance(parts, dict):
            raise ValueError("Dream response has no per_key actions")
        groups = {}
        clipping = {}
        for group, side in self.prefixes.items():
            arm = np.asarray(parts.get(f"action.{side}_arm"), dtype=float)
            grip = np.asarray(parts.get(f"action.{side}_grip"), dtype=float)
            if grip.shape == (self.horizon,):
                grip = grip[:, None]
            if arm.shape != (self.horizon, 6) or grip.shape != (self.horizon, 1):
                raise ValueError(
                    f"Dream {side} action shape: arm={arm.shape}, grip={grip.shape}; "
                    f"expected arm=({self.horizon}, 6), grip=({self.horizon}, 1)"
                )
            for name, values in ((f"action.{side}_arm", arm), (f"action.{side}_grip", grip)):
                invalid = np.argwhere(~np.isfinite(values))
                if invalid.size:
                    index = tuple(invalid[0])
                    raise ValueError(
                        f"Dream {name} non-finite: count={len(invalid)}, "
                        f"first_index={tuple(map(int, index))}, value={values[index]}"
                    )
            invalid = np.argwhere((grip < 0) | (grip > 1))
            if invalid.size:
                clipping[f"action.{side}_grip"] = {
                    "count": len(invalid),
                    "raw_min": float(grip.min()),
                    "raw_max": float(grip.max()),
                    "indices": invalid[:, 0].tolist(),
                }
            # Match dreamlabs 1.0.5 robots.adapter._arm_command: normalized
            # gripper commands saturate at physical closed/open endpoints.
            # Keep model joints unchanged and report every saturated grip row.
            groups[group] = np.concatenate([arm, np.clip(grip, 0.0, 1.0)], axis=1)
        return {
            "format": "joint",
            "action_semantics": "absolute_joint_position",
            "actions": groups,
            "sampling_stats": {"dream_api": {"gripper_clipping": clipping}},
        }
