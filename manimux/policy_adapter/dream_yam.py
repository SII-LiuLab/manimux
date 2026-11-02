"""YAM joint actions with Dream's training-reference TCP state on requests."""

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import yourdfpy

from manimux.policies.dream.codec import require_default_sampling
from manimux.policy_adapter.joint import JointAdapter
from manimux.types import InferenceRequest


@dataclass(slots=True)
class DreamRequest(InferenceRequest):
    model_state: dict = field(default_factory=dict)


class DreamYamJointAdapter(JointAdapter):
    """The reference URDF encodes model inputs, never robot IK or execution."""

    supports_gripper_mapping = False
    supports_multi_inflight = False
    supports_candidate_decoding = False

    def __init__(self, robot, policy, *, kinematics=None):
        super().__init__(robot, policy, kinematics=kinematics)
        if kinematics is None:
            from manimux.embodiments.robot.base import RobotModel

            kinematics = RobotModel.from_config(robot["config"]).kinematics
        from manimux.embodiments.arm.yam.kinematics import YamManipulatorKinematics

        if set(kinematics.models) != set(self._dimensions) or any(
            not isinstance(model, YamManipulatorKinematics) for model in kinematics.models.values()
        ):
            raise ValueError("Dream native joint output requires the configured YAM robot")
        self.prefixes = policy["adapter"]["group_prefixes"]
        if (
            len(self._dimensions) != 2
            or set(self.prefixes) != set(self._dimensions)
            or set(self.prefixes.values()) != {"left", "right"}
            or any(width != 7 for width in self._dimensions.values())
        ):
            raise ValueError("Dream YAM requires two groups, each six joints and one gripper")
        self.reference = yourdfpy.URDF.load(
            Path(__file__).parent / "assets/dream_yam.urdf",
            load_meshes=False,
            load_collision_meshes=False,
        )
        self.tip = np.eye(4)
        self.tip[2, 3] = 0.1347  # dreamlabs 1.0.5: YAM_ARM.tools["tcp"], link_6 axes.

    def prepare_request(self, request):
        require_default_sampling(request)
        if set(request.observation.state.groups) != set(self.prefixes):
            raise ValueError("Dream YAM observation groups differ from configuration")
        state = {}
        for group, side in self.prefixes.items():
            values = np.asarray(request.observation.state.groups[group], dtype=float)
            if values.shape != (7,) or not np.isfinite(values).all() or not 0 <= values[-1] <= 1:
                raise ValueError(f"Invalid YAM joint/gripper state for {group}")
            self.reference.update_cfg({f"joint{i + 1}": values[i] for i in range(6)})
            pose = self.reference.get_transform("link_6") @ self.tip
            state.update(
                {
                    f"state.{side}_arm": values[:6].tolist(),
                    f"state.{side}_gripper": [float(values[-1])],
                    f"state.{side}_grip": [float(values[-1])],
                    f"state.{side}_eef_xyz": pose[:3, 3].tolist(),
                    f"state.{side}_eef_rot6": np.r_[pose[:3, 0], pose[:3, 1]].tolist(),
                }
            )
        return DreamRequest(
            **{item.name: getattr(request, item.name) for item in fields(InferenceRequest)},
            model_state=state,
        )
