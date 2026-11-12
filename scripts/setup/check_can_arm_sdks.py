"""Check installed ARX and standard PiPER SDKs without creating device sessions."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

from manimux.embodiments.arm.arx_x5 import ArxX5Arm
from manimux.embodiments.robot.base import RobotModel
from manimux.types import FloatArray


def pose_matrix(pose: FloatArray) -> FloatArray:
    """SDK pose vectors use metres followed by xyz Euler angles in radians."""
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def check_arx() -> dict[str, object]:
    """Inspect the official binding and run FK without constructing a device session."""
    import arx_x5_python
    import kinematic_solver

    driver = arx_x5_python.InterfacesPy
    for name in (
        "get_joint_positions",
        "set_joint_positions",
        "set_catch",
        "set_arm_status",
        "arx_x",
    ):
        if not callable(getattr(driver, name, None)):
            raise RuntimeError(f"official X5 binding is missing {name}")
    model = URDF.load(str(ArxX5Arm.model_path), load_meshes=False)
    names = [joint.name for joint in model.actuated_joints if joint.type == "revolute"]
    if names != [f"joint{i}" for i in range(1, 7)]:
        raise RuntimeError("packaged X5 URDF must declare joint1 through joint6 in order")
    model.update_cfg(dict(zip(names, np.zeros(6), strict=True)))
    origin = model.get_transform("link6", "base_link")[:3, 3].copy()
    solver = kinematic_solver.KinematicSolver()
    errors = []
    for values in (np.zeros(6), [0.0, 0.2, 0.3, 0.1, 0.1, -0.1], [0.3, 0.4, 0.2, 0.1, 0.2, 0.1]):
        joints = np.asarray(values, dtype=np.float64)
        actual = pose_matrix(np.asarray(solver.forward_kinematics(joints)))
        model.update_cfg(dict(zip(names, joints, strict=True)))
        expected = model.get_transform("link6", "base_link").copy()
        # The vendor solver reports position relative to the zero-pose endpoint.
        # This reference convention is distinct from the URDF's arm-base origin.
        expected[:3, 3] -= origin
        error = float(np.max(np.abs(actual - expected)))
        if not np.isfinite(actual).all() or error > 1e-8:
            raise RuntimeError(f"vendor X5 FK disagrees with its zero-reference URDF: {error}")
        errors.append(error)
    return {
        "binding": "arx_x5_python",
        "fk_error": max(errors),
        "zero_pose_origin_m": origin.tolist(),
    }


def check_piper() -> dict[str, object]:
    from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

    config = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.DEFAULT,
        interface="socketcan",
        channel="MANIMUX_OFFLINE_ONLY",
        auto_connect=False,
        enable_check_can=False,
    )
    for firmware in ("default", "v183", "v188", "v189"):
        driver = AgxArmFactory.load_class({**config, "firmeware_version": firmware})
        for name in (
            "connect",
            "disconnect",
            "init_effector",
            "get_context",
            "get_driver_states",
            "get_joints_enable_status_list",
            "enable",
            "set_speed_percent",
            "move_j",
        ):
            if not callable(getattr(driver, name, None)):
                raise RuntimeError(f"PiPER {firmware} driver is missing {name}")
    table = get_mdh("piper")
    joints = [0.0, 0.5, -0.5, 0.1, 0.2, 0.3]
    actual = np.asarray(fk_from_mdh(table, joints))
    expected = np.eye(4)
    for (d, a, alpha, offset), q in zip(table, joints, strict=True):
        x, z = np.eye(4), np.eye(4)
        x[:3, :3] = Rotation.from_euler("x", alpha).as_matrix()
        x[0, 3] = a
        z[:3, :3] = Rotation.from_euler("z", q + offset).as_matrix()
        z[2, 3] = d
        expected = expected @ x @ z
    actual_matrix = pose_matrix(actual)
    error = float(np.max(np.abs(actual_matrix - expected)))
    if config["robot"] != "piper" or not np.isfinite(actual).all() or error > 1e-10:
        raise RuntimeError(f"standard PiPER configuration or FK failed: {error}")
    return {
        "package": "pyAgxArm",
        "version": version("pyAgxArm"),
        "firmwares": ["default", "v183", "v188", "v189"],
        "fk_error": error,
    }


def check_models() -> dict[str, dict[str, int]]:
    """Load the real public assemblies without importing hardware bindings."""
    import manimux

    root = Path(manimux.__file__).parent
    result = {}
    for name in ("aloha_agilex", "arx_x5_dual", "piper_dual"):
        model = RobotModel.from_config(root / "configs/embodiment/robot" / f"{name}.yaml")
        for kin in model.kinematics.models.values():
            if kin.num_coordinates != 7 or not np.isfinite(kin.fk(np.r_[np.zeros(6), 0.5])).all():
                raise RuntimeError(f"invalid offline model: {name}")
        result[model.name] = {
            group: kin.num_coordinates for group, kin in model.kinematics.models.items()
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdk",
        choices=("arx", "piper", "both"),
        default="both",
        help="installed SDKs to check (default: both); missing dependencies fail the check",
    )
    parser.add_argument("--output", type=Path, help="also write the JSON report to this file")
    args = parser.parse_args()
    sdks = {}
    if args.sdk in {"arx", "both"}:
        sdks["arx"] = check_arx()
    if args.sdk in {"piper", "both"}:
        sdks["piper"] = check_piper()
    results = {
        "sdks": sdks,
        "assemblies": check_models(),
        "hardware_connected": False,
    }
    report = json.dumps(results, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
