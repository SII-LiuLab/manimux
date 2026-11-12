"""Native EEF encoding to canonical poses; anchoring belongs to the robot adapter."""

import numpy as np
from scipy.spatial.transform import Rotation

SEMANTICS = {
    "absolute_per_arm_base_xyz_wxyz",
    "delta_observation_base_xyz_wxyz",
    "delta_observation_tool_xyz_wxyz",
    "delta_step_base_xyz_wxyz",
}


class EefCodec:
    def __init__(self, options, gripper_dofs):
        self.options = dict(options)
        self.rotation = options.get("rotation")
        if self.rotation not in {"axis_angle", "quaternion_wxyz"}:
            raise ValueError("eef.rotation must explicitly select axis_angle or quaternion_wxyz")
        self.semantics = options.get("semantics")
        if self.semantics not in SEMANTICS:
            raise ValueError(f"eef.semantics must be one of {sorted(SEMANTICS)}")
        self.arm_width = 6 if self.rotation == "axis_angle" else 7
        self.tool_widths = tuple(gripper_dofs)
        self.native_dim = len(self.tool_widths) * self.arm_width + sum(self.tool_widths)
        self.translation_scale = self._scale("translation_scale")
        self.rotation_scale = self._scale("rotation_scale")
        self.clip = options.get("clip_input")
        if self.clip is not None and (not np.isfinite(self.clip) or self.clip <= 0):
            raise ValueError("eef.clip_input must be positive and finite")
        if self.rotation == "quaternion_wxyz" and (
            self.clip is not None or not np.array_equal(self.rotation_scale, np.ones(3))
        ):
            raise ValueError("quaternion output cannot use rotation scaling or input clipping")
        self.threshold = options.get("gripper_threshold")
        if self.threshold is not None and not np.isfinite(self.threshold):
            raise ValueError("eef.gripper_threshold must be finite")

    def _scale(self, name):
        value = np.asarray(self.options.get(name, [1, 1, 1]), dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f"eef.{name} must contain three positive finite scales")
        return value

    def convert(self, actions):
        parts, offset = [], 0
        for width in self.tool_widths:
            arm = np.array(actions[:, offset : offset + self.arm_width], dtype=float, copy=True)
            tool = actions[:, offset + self.arm_width : offset + self.arm_width + width].copy()
            if self.clip is not None:
                arm = np.clip(arm, -self.clip, self.clip)
            xyz = arm[:, :3] * self.translation_scale
            if self.rotation == "axis_angle":
                quat = Rotation.from_rotvec(arm[:, 3:6] * self.rotation_scale).as_quat()[
                    :, [3, 0, 1, 2]
                ]
            else:
                quat = arm[:, 3:7]
                if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-4):
                    raise ValueError("StarVLA EEF quaternion must be unit length")
            if self.threshold is not None:
                tool = (tool > self.threshold).astype(float)
            parts.append(np.concatenate((xyz, quat, tool), axis=-1))
            offset += self.arm_width + width
        return np.concatenate(parts, axis=-1).astype(np.float32)
