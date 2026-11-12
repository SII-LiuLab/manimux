#!/usr/bin/env python3
# ruff: noqa: E501
"""Verify replay contracts and write baseline forensic evidence, offline only."""

import json
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import run as study

from manimux.runtime.execution import ExecutionPipeline
from manimux.runtime.executors.direct import DirectExecutor
from manimux.types import ActionHorizon, RobotState


def main():
    rows = [r for r in study.discover() if r["included"]]
    results = []
    forensics = []
    for row in rows:
        d = study.load(row)
        dest = study.OUT / "episodes" / row["id"]
        r = dict(np.load(dest / "baseline.npz"))
        model = study.make_model(d["c"])
        info = json.loads((dest / "identity.json").read_text())
        assert all(
            v["recorded"] == v["current"] and v["current"] for v in info["source_files"].values()
        )
        assert all(a["raw_to_committed_exact"] and a["clock_exact"] for a in info["plan_audit"])
        np.testing.assert_array_equal(r["command"], d["arrays"]["command"])
        active = np.isfinite(r["target"][:, 0, 0])
        np.testing.assert_array_equal(r["reference"][active], d["arrays"]["scheduled"][active])
        fraction_errors = [
            abs(r["fraction"][i, j] - event.get("pose_speed_limited", {}).get(g, 1.0))
            for i, event in d["events"].items()
            for j, g in enumerate(study.GROUPS)
        ]
        assert max(fraction_errors) < 1e-12
        e = d["c"]["executor"]
        dt = 1 / d["c"]["robot"]["control_hz"]
        safety = e["command_safety"]
        per_arm = {}
        for j, g in enumerate(study.GROUPS):
            previous = np.r_[r["command"][:1, j], r["command"][:-1, j]]
            desired_v = (r["reference"][:, j] - previous) / dt
            cap = e["motion_limits"]["arm"]["max_velocity"] * min(
                1, e["motion_limits"]["arm"]["max_step_dt_s"] / dt
            )
            shaped = np.max(abs(r["reference"][:, j, :7] - r["command"][:, j, :7]), axis=1) > 1e-9
            velocity_limited = active & (np.max(abs(desired_v[:, :7]), axis=1) > cap + 1e-9)
            holds = active & (r["reason"] != "ok")
            np.testing.assert_array_equal(r["command"][holds, j], previous[holds])
            kin = model.kinematics.models[g].arm
            lower, upper = kin.joint_position_limits()
            margin = np.radians(d["c"]["execution"]["ik"]["diff_ik"]["limit_margin_deg"])
            near_lower = (r["command"][:, j, :7] - (lower + margin)) * 180 / np.pi
            near_upper = ((upper - margin) - r["command"][:, j, :7]) * 180 / np.pi
            per_arm[g] = dict(
                ik_qdot_cap_rad_s=[e["motion_limits"]["arm"]["max_velocity"]] * 7,
                safety_velocity_rad_s=safety["max_velocity"][g][:7],
                physical_joint_bounds_deg=[np.degrees(lower).tolist(), np.degrees(upper).tolist()],
                ik_inner_bounds_deg=[
                    np.degrees(lower + margin).tolist(),
                    np.degrees(upper - margin).tolist(),
                ],
                command_safety_bounds_deg=[
                    np.degrees(safety["position_lower"][g][:7]).tolist(),
                    np.degrees(safety["position_upper"][g][:7]).tolist(),
                ],
                min_command_inner_margin_deg=np.minimum(near_lower, near_upper)
                .min(axis=0)
                .tolist(),
                executor_velocity_clipped_ticks=int(velocity_limited.sum()),
                executor_acceleration_changed_ticks=int(
                    (active & shaped & ~velocity_limited).sum()
                ),
                command_joint_velocity_max_rad_s=np.max(
                    abs(np.diff(r["command"][:, j, :7], axis=0) / dt), axis=0
                ).tolist(),
            )
        failures = np.flatnonzero(active & (r["reason"] != "ok"))
        first_failure = None
        if len(failures):
            first = int(failures[0])
            window = []
            for i in range(max(1, first - 6), min(len(active), first + 3)):
                window.append(
                    dict(
                        tick=i,
                        reason=str(r["reason"][i]),
                        previous_j4_deg=float(np.degrees(r["command"][i - 1, 0, 3])),
                        reference_j4_deg=float(np.degrees(r["reference"][i, 0, 3])),
                        command_j4_deg=float(np.degrees(r["command"][i, 0, 3])),
                        command_velocity_j4_rad_s=float(
                            (r["command"][i, 0, 3] - r["command"][i - 1, 0, 3]) / dt
                        ),
                        reference_velocity_j4_rad_s=float(
                            (r["reference"][i, 0, 3] - r["command"][i - 1, 0, 3]) / dt
                        ),
                    )
                )
            lower = model.kinematics.models["left_arm"].arm.joint_position_limits()[0][3]
            margin = np.radians(d["c"]["execution"]["ik"]["diff_ik"]["limit_margin_deg"])
            first_failure = dict(
                tick=first,
                required_recovery_velocity_rad_s=float(
                    (lower + margin - r["command"][first - 1, 0, 3]) / dt
                ),
                allowed_velocity_rad_s=e["motion_limits"]["arm"]["max_velocity"],
                window=window,
            )
        # State was read before the same tick's command dispatch; compare with the preceding command too.
        historical_previous = np.concatenate(
            [d["arrays"]["command"][:1], d["arrays"]["command"][:-1]]
        )
        a = study.poses(model, historical_previous)
        b = study.poses(model, d["arrays"]["state"])
        pos, rot = study.error(a, b)
        feedback = {
            g: dict(
                previous_command_to_state_mm=study.stats(pos[active, j]),
                previous_command_to_state_deg=study.stats(rot[active, j]),
            )
            for j, g in enumerate(study.GROUPS)
        }
        z = study.zarr.open(str(study.Path(row["path"]) / "data.zarr"), "r")
        sent = {}
        for g in study.GROUPS:
            valid = z[f"ticks/sent_command/{g}/valid"][:]
            sent[g] = dict(
                valid_ticks=int(valid.sum()),
                total_ticks=len(valid),
                note="A host dispatch snapshot, not a controller acknowledgement or timestamped measured response.",
            )
        forensics.append(
            dict(
                id=row["id"],
                task=row["task"],
                status=row["status"],
                stratum=info["stratum"],
                per_arm=per_arm,
                first_failure=first_failure,
                feedback=feedback,
                sent_command=sent,
            )
        )
        results.append(
            dict(
                episode=row["id"],
                exact_command=True,
                exact_reference=True,
                source_hashes_match=True,
                latency_skip_clock_verified=True,
                failed_ticks_hold_previous_command=True,
                max_pose_fraction_error=max(fraction_errors),
            )
        )

    d = study.load(rows[0])
    c = d["c"]
    model = study.make_model(c)
    active = np.flatnonzero(np.isfinite(d["target"][:, 0, 0]))
    i = int(active[0])
    now = int(d["times"][i])
    dt = 10_000_000
    prev = {g: d["arrays"]["command"][i - 1, j].copy() for j, g in enumerate(study.GROUPS)}
    state = RobotState(prev, now, 0)
    ref = ActionHorizon(
        start_time_ns=now,
        dt_ns=dt,
        plan_id="test",
        groups={g: d["target"][i, j][None] for j, g in enumerate(study.GROUPS)},
        action_space="absolute_tcp_pose",
    )
    executor = DirectExecutor(c["executor"]["motion_limits"], 0.01)
    executor.reset(state)
    pipe = ExecutionPipeline(
        executor,
        kinematics=model.kinematics,
        ik_options=c["execution"]["ik"],
        control_dt_ns=dt,
        motion_limits=c["executor"]["motion_limits"],
    )
    # Force a shared-budget overrun before either arm; reset must clear primal AND dual state.
    pipe.warmup(prev)
    clock = iter([0, 10, 10_000_000, 10_000_001, 10_000_002, 10_000_003])
    with patch("time.monotonic_ns", side_effect=lambda: next(clock)):
        result = pipe.step(now, state, ref, last_command=prev)
    assert result.ik_status["reason"] == "budget_exceeded"
    for g in study.GROUPS:
        np.testing.assert_array_equal(result.command.groups[g], prev[g])
        assert pipe.resolver.diff.solvers[g]._problem is None
        np.testing.assert_array_equal(executor._previous_velocity[g], np.zeros(8))
    # Force the actual empty-box condition without relaxing any bounds.
    pipe.resolver.budget_ns = 10**18
    bad = deepcopy(prev)
    lo = model.kinematics.models["left_arm"].arm.joint_position_limits()[0][3]
    bad["left_arm"][3] = lo + np.radians(c["execution"]["ik"]["diff_ik"]["limit_margin_deg"]) - 0.02
    executor.reset(RobotState(bad, now, 0))
    result = pipe.step(now, RobotState(bad, now, 0), ref, last_command=bad)
    assert result.ik_status["reason"] == "left_arm:qp_infeasible"
    for g in study.GROUPS:
        np.testing.assert_array_equal(result.command.groups[g], bad[g])
        assert pipe.resolver.diff.solvers[g]._problem is None
    study.save(
        study.OUT / "verification.json",
        dict(
            baselines=results,
            synthetic_budget_atomic_hold_reset=True,
            synthetic_empty_box_atomic_hold_reset=True,
            hardware_connected=False,
            model_service_started=False,
        ),
    )
    study.save(study.OUT / "baseline_forensics.json", forensics)
    from tcp_objective import TcpStepDifferentialIK

    from manimux.kinematics.tianji_diff import rotation_vector, step_diff_config

    tcp = TcpStepDifferentialIK(
        model.kinematics.models,
        step_diff_config(c["executor"]["motion_limits"], c["execution"]["ik"]["diff_ik"]),
        0.01,
    )
    jac_errors = []
    for g in study.GROUPS:
        solver = tcp.solvers[g]
        frame = solver.kinematics
        q = prev[g][:7]
        pose, jac = frame.flange_and_jacobian(q)
        np.testing.assert_allclose(pose, model.kinematics.models[g].fk(prev[g]), atol=1e-14)
        numeric = np.empty((6, 7))
        h = 1e-6
        for j in range(7):
            delta = np.zeros(7)
            delta[j] = h
            plus = frame.flange_and_jacobian(q + delta)[0]
            minus = frame.flange_and_jacobian(q - delta)[0]
            numeric[:3, j] = (plus[:3, 3] - minus[:3, 3]) / (2 * h)
            numeric[3:, j] = rotation_vector(plus[:3, :3] @ minus[:3, :3].T) / (2 * h)
        err = float(np.max(abs(numeric - jac)))
        assert err < 1e-7
        jac_errors.append(err)
        original = pipe.resolver.diff.solvers[g]
        np.testing.assert_array_equal(solver.lower, original.lower)
        np.testing.assert_array_equal(solver.upper, original.upper)
        assert (
            solver.margin == original.margin
            and solver.j67_margin == original.j67_margin
            and solver.vmax == original.vmax
        )
        result = tcp.solve(g, pose, prev[g], prev[g][-1])
        assert result.ok
        tcp.reset()
        assert solver._problem is None
    study.save(
        study.OUT / "tcp_verification.json",
        dict(
            tcp_jacobian_max_abs_errors=jac_errors,
            identical_joint_bounds_margins_velocity_boxes=True,
            identical_fixed_tool_fk=True,
            production_qp_reused=True,
            zero_error_target_solved=True,
            reset_clears_osqp=True,
        ),
    )
    print(
        "Verified all 11 baselines, source hashes, time trim/skip, atomic QP and budget hold/reset."
    )


if __name__ == "__main__":
    main()
