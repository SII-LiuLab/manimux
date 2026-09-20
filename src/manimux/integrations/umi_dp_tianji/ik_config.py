"""Bind embodiment QP rates to the shared executor motion profile."""

from manimux.embodiments.arm.tianji.kinematics import DifferentialIKConfig


def profile_parameters(config):
    motion = config["execution"]["motion_limits"]
    if motion is None or motion["arm"]["max_step_dt_s"] is None:
        raise ValueError("UMI differential IK requires shared motion_limits and max_step_dt_s")
    return {
        "max_velocity_rad_s": motion["arm"]["max_velocity"],
        "dt_max_s": motion["arm"]["max_step_dt_s"],
    }


def bind_diff_ik_profile(config):
    """Only the offline binding launcher/probes call this; runtime only validates."""
    options = config["policy"]["options"]
    if options.get("ik_backend", "analytic") != "diff":
        return
    parameters = options.setdefault("diff_ik", {})
    for key, value in profile_parameters(config).items():
        if key in parameters and parameters[key] != value:
            raise ValueError(f"diff_ik.{key} conflicts with the shared motion profile")
        parameters[key] = value
    validate_diff_ik_profile(config)


def validate_diff_ik_profile(config):
    options = config["policy"]["options"]
    if options.get("ik_backend", "analytic") != "diff":
        return
    parameters = DifferentialIKConfig.model_validate(options.get("diff_ik", {}))
    for key, value in profile_parameters(config).items():
        if getattr(parameters, key) != value:
            raise ValueError(f"diff_ik.{key} conflicts with the shared motion profile; rebind")
