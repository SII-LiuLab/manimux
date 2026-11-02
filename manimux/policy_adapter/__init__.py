"""Load adapters from experiment configuration without legacy aliases or model-specific registries."""

from manimux.kinematics.failure import DEFAULT_FAILURE_POLICY, validate_failure_policy
from manimux.plugins import load_plugin
from manimux.policy_adapter.base import PolicyAdapter


def build_policy_adapter(
    robot: dict, policy: dict, *, kinematics=None, motion_limits=None, ik_options=None
) -> PolicyAdapter:
    if "ik_failure_policy" in policy["adapter"]:
        raise ValueError("Move policy.adapter.ik_failure_policy to execution.ik.failure_policy")
    failure_policy = validate_failure_policy(
        (ik_options or {}).get("failure_policy", DEFAULT_FAILURE_POLICY)
    )
    adapter_class = load_plugin(
        policy["adapter"]["type"], group="manimux.policy_adapter", builtins={}
    )
    if (
        policy["adapter"].get("gripper_mapping", {}).get("mode") == "curve"
        and not getattr(adapter_class, "supports_gripper_mapping", False)
    ):
        raise ValueError(
            f"{adapter_class.__name__} does not support executor.smooth.gripper.mode=curve"
        )
    options = {"kinematics": kinematics}
    if getattr(adapter_class, "uses_motion_limits", False):
        options["motion_limits"] = motion_limits
    adapter = adapter_class(robot, policy, **options)
    adapter.ik_failure_policy = failure_policy
    return adapter
