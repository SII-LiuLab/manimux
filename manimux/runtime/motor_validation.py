"""CPU-only verification of runtime-owned IK and physical motor commands."""

import numpy as np


def validate_motor_audit(audit, mode):
    c = audit["execution_contract"]
    if (c.get("controller_location") != "manimux_runtime" or c.get("action_space") != "joint_position"
            or c.get("command_semantics") != "physical_motor_positions"
            or not c.get("manimux_ik") or not c.get("manimux_interpolation")
            or c.get("robodojo_interpolation") or c.get("control_hz_target") != 250
            or c.get("policy_action_hz") != 25 or c.get("physics_during_inference") != (mode != "fresh_paused")
            or not c.get("commands") or not c.get("solves") or c.get("runtime_pid") == c.get("ik_worker_pid")
            or c.get("interpolation_rule") != "linear_80pct_then_hold"
            or c.get("ik_seed") != "current_measured_joint_state"):
        raise RuntimeError("runtime motor execution contract mismatch")
    plans = {e["plan_id"]: e["groups"] for e in audit["manimux_events"] if e["kind"] == "plan_committed"}
    solves = {}
    for solve in c["solves"]:
        key = (solve["plan_id"], solve["source_index"])
        plan = plans.get(key[0])
        if plan is None or len(solve["sequence"]) != 10 or solve["worker_pid"] != c["ik_worker_pid"]:
            raise RuntimeError("runtime solve lacks a source plan or isolated worker")
        if not all(np.array_equal(solve["groups"][k], plan[k][key[1]]) for k in plan):
            raise RuntimeError("runtime altered an EEF source row")
        for name, source in solve["measured"].items():
            side = name.removesuffix("_arm")
            sequence = np.asarray([item[name] for item in solve["sequence"]])
            if sequence.shape != (10, 8) or not np.isfinite(sequence).all():
                raise RuntimeError("invalid runtime motor sequence")
            if solve["ik_status"][side] == "Success":
                initial, target = np.array(source[:6]), sequence[-1, :6]
            else:
                initial = target = np.array(solve["held"][name][:6])
            expected = [initial+(target-initial)*(i+1)/9 for i in range(8)] + [target,target]
            if not np.allclose(sequence[:,:6], expected, rtol=0, atol=1e-6):
                raise RuntimeError("runtime interpolation differs from native 8+2 rule")
        solves[key] = solve
    previous = 0
    for command in c["commands"]:
        key = (command["plan_id"], command["source_index"])
        solve = solves.get(key)
        if (solve is None or command["action_space"] != "joint_position" or command["command_seq"] <= previous
                or not 0 <= command["interpolation_tick"] < 10):
            raise RuntimeError("invalid motor command provenance or order")
        expected = {k:list(v) for k,v in solve["sequence"][command["interpolation_tick"]].items()}
        for robot in c["robots"]:
            name = robot["prefix"] + "_arm"
            measured = command["measured_motor_state"][name][6]
            limit = .2 * (robot["scale"][1] - robot["scale"][0])
            value = max(measured-limit, min(measured+limit, expected[name][6]))
            expected[name][6:] = [value, value * robot["mimic"][1] + robot["mimic"][2]]
        if not all(np.array_equal(command["groups"][k], expected[k]) for k in expected):
            raise RuntimeError("motor command differs from runtime interpolation")
        previous = command["command_seq"]
    return {"motor_commands": len(c["commands"]), "ik_solves": len(c["solves"])}
