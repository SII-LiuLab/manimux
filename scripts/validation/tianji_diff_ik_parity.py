#!/usr/bin/env python3
"""Compare the port with an explicit tianji-control checkout, without hardware.

The reference checkout is required only by this validation probe. Its actual
diff_ik, nullspace and kinematics modules are loaded unmodified. An isolated
ik_solver import supplies its literal BD67 table, avoiding SDK path side effects.
The vendored Marvin library is used only for the reference pose conversions.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    from manimux.cli import load_config
    from manimux.embodiments.arm.tianji.kinematics import (
        BD67_REAL,
        DH_TABLE_M6_40,
        DifferentialIKConfig,
        TianjiArmKinematics,
        TianjiDifferentialKinematics,
    )
    from manimux.embodiments.arm.tianji.sdk.marvin import fx_kine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    reference = args.reference.resolve()
    table_source = ast.parse((reference / "algos/ik_solver.py").read_text())
    table = next(
        ast.literal_eval(node.value)
        for node in table_source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "BD67_REAL" for target in node.targets
        )
    )
    np.testing.assert_array_equal(table, BD67_REAL)
    package = types.ModuleType("algos")
    package.__path__ = [str(reference / "algos")]
    sys.modules["algos"] = package
    stub = types.ModuleType("algos.ik_solver")
    stub.BD67_REAL = table
    sys.modules["algos.ik_solver"] = stub
    load_module("algos.nullspace", reference / "algos/nullspace.py")
    old = load_module("algos.diff_ik", reference / "algos/diff_ik.py")
    old_kin = load_module("algos.kinematics", reference / "algos/kinematics.py")
    converter = fx_kine.Marvin_Kine()
    profile = load_config(REPO / "configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml")
    motion = profile["execution"]["motion_limits"]["arm"]
    tuning_data = yaml.safe_load(
        (REPO / "configs/policy/umi_dp/adapter/tianji_diff.yaml").read_text()
    )
    tuning_data.update(
        max_velocity_rad_s=motion["max_velocity"], dt_max_s=motion["max_step_dt_s"]
    )
    tuning = DifferentialIKConfig(**tuning_data)
    starts = [
        [50, -40, -30, -100, -65, 0, 40],
        [155, -95, 140, -115, -100, 48, 48],  # active nullspace, positive J67 quadrant
        [-150, -95, 140, -115, -100, -48, 48],
        [-150, -95, 140, -115, -100, -48, -48],
        [155, -95, 140, -115, -100, 48, -48],
    ]
    report = {"sequences": [], "hardware_connected": False}
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    # Feed identical inputs (not independently integrated trajectories) to both
    # solvers; each still retains its own OSQP warm-start and dual state.
    for arm in ("left", "right"):
        overrides = {} if arm == "left" else {6: [-58, 58]}
        kin = TianjiArmKinematics(arm=arm, joint_limits_deg=overrides)
        lo, hi = (np.degrees(values) for values in kin.joint_position_limits())
        for sequence, initial in enumerate(starts):
            port = TianjiDifferentialKinematics(kin, tuning)
            legacy = old.DiffIKSolver(
                old_kin.ArmKinematics(DH_TABLE_M6_40),
                converter,
                lo.tolist(),
                hi.tolist(),
                [np.degrees(motion["max_velocity"])] * 7,
                limit_margin_deg=kin.limit_margin_deg,
                dt_max=motion["max_step_dt_s"],
                w_pos=tuning.w_pos,
                w_rot=tuning.w_rot,
                lam=tuning.lam,
                mu_nullspace=tuning.mu_nullspace,
                nullspace_activation_deg=tuning.nullspace_activation_deg,
            )
            previous = np.radians(initial)
            errors, rate_errors, pos_errors, rot_errors, times = [], [], [], [], []
            reasons = {}
            for step in range(args.steps):
                delta = 0.08 * np.sin(np.arange(7) + step * 0.13)
                target_joints = previous + np.radians(delta)
                target = kin.flange(target_joints)
                sdk_flange = target.copy()
                sdk_flange[:3, 3] *= 1000
                xyzabc = converter.mat4x4_to_xyzabc(pose_mat=sdk_flange)
                dt = (0.004, 0.008, 0.030, 0.0005)[step % 4]
                expected = legacy.solve(xyzabc, np.degrees(previous), dt)
                actual = port.ik(target, previous, duration_s=dt)
                expected_reason = expected.reason
                if expected.ok and expected.pos_err_mm > tuning.max_lag_mm:
                    expected_reason = "tracking_lag"
                if actual.reason != expected_reason:
                    raise AssertionError((arm, sequence, step, expected_reason, actual))
                reasons[actual.reason] = reasons.get(actual.reason, 0) + 1
                if expected.joints is not None:
                    actual_joints = (
                        actual.joints
                        if actual.joints is not None
                        else previous
                        + actual.diagnostics["qdot_rad_s"] * min(dt, tuning.dt_max_s)
                    )
                    errors.append(
                        float(np.max(np.abs(np.degrees(actual_joints) - expected.joints)))
                    )
                    rate_errors.append(
                        float(
                            np.max(
                                np.abs(
                                    np.degrees(actual.diagnostics["qdot_rad_s"])
                                    - expected.qdot_deg_s
                                )
                            )
                        )
                    )
                    pos_errors.append(
                        abs(actual.diagnostics["pos_err_mm"] - expected.pos_err_mm)
                    )
                    rot_errors.append(
                        abs(actual.diagnostics["rot_err_deg"] - expected.rot_err_deg)
                    )
                if actual.ok:
                    previous = actual.joints.copy()
                times.append(actual.diagnostics["solve_time_ms"])
            metrics = {
                "arm": arm,
                "sequence": sequence,
                "steps": args.steps,
                "max_joint_error_deg": max(errors, default=0),
                "max_velocity_error_deg_s": max(rate_errors, default=0),
                "max_position_residual_error_mm": max(pos_errors, default=0),
                "max_rotation_residual_error_deg": max(rot_errors, default=0),
                "median_solve_ms": float(np.median(times)),
                "reasons": reasons,
            }
            # SI/mm conversion roundoff perturbs OSQP's retained dual state;
            # compare numerically, not bitwise (also true of the old runner).
            if metrics["max_joint_error_deg"] > 1e-4:
                raise AssertionError(metrics)
            report["sequences"].append(metrics)
    files = ["algos/diff_ik.py", "algos/kinematics.py", "algos/nullspace.py", "algos/ik_solver.py"]
    report["reference_revision"] = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    report["reference_sha256"] = {
        name: hashlib.sha256((reference / name).read_bytes()).hexdigest() for name in files
    }
    report["max_velocity_rad_s"] = motion["max_velocity"]
    report["dt_max_s"] = motion["max_step_dt_s"]
    report["osqp_version"] = port._osqp.__version__
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
