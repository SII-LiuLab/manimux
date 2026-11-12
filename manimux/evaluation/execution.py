"""Read-only execution evidence shared by reference metrics.

Never infer ownership from acceptance order. Tick plan IDs, lifecycle events,
committed plans and recorded commands jointly qualify each interval.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

from manimux.evaluation.trajectory import ReferenceTrajectory, _sample


def read_execution(episode: Path, groups, max_gap_s):
    root = zarr.open_group(str(episode / "data.zarr"), mode="r")
    ticks = root["ticks"]
    times = np.asarray(ticks["monotonic_ns"][:], dtype=np.int64)
    ids = np.asarray(ticks["plan_id"][:]).astype(str)
    if not len(times) or ids.shape != times.shape or np.any(np.diff(times) <= 0):
        raise ValueError("missing_or_nonmonotonic_ticks")
    plans = {}
    digest = hashlib.sha256(times.tobytes() + json.dumps(ids.tolist()).encode())
    for name, item in root["plans"].groups():
        c = item["committed"]
        action_space = c.attrs.get("action_space")
        if action_space not in {"joint_position", "absolute_tcp_pose"}:
            raise ValueError("unsupported_committed_action_space")
        arrays = {g: np.asarray(c[g][:], float) for g in groups}
        if any(v.ndim != 2 or not len(v) or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("invalid_committed_plan")
        if action_space == "absolute_tcp_pose":
            from manimux.policy_adapter.pose_output import validate_pose_rows

            arrays = {g: validate_pose_rows(v) for g, v in arrays.items()}
        plan = ReferenceTrajectory(
            int(name),
            str(c.attrs["plan_id"]),
            int(item.attrs["request_seq"]),
            int(c.attrs["start_time_ns"]),
            int(c.attrs["dt_ns"]),
            arrays,
            action_space=action_space,
        )
        if plan.dt_ns <= 0 or plan.plan_id in plans:
            raise ValueError("invalid_or_duplicate_plan")
        plans[plan.plan_id] = plan
        digest.update(json.dumps(dict(c.attrs), sort_keys=True).encode())
        for g, v in arrays.items():
            digest.update(g.encode() + str(v.shape).encode() + v.tobytes())
    events_path = episode / "events.jsonl"
    event_bytes = events_path.read_bytes()
    digest.update(event_bytes)
    events = [json.loads(line) for line in event_bytes.splitlines() if line.strip()]
    breaks = np.zeros(len(times), bool)
    breaks[0] = True
    breaks[1:] |= np.diff(times) > max_gap_s * 1e9
    for e in events:
        restart = (e.get("kind") == "robogui_control_state" and e.get("paused")) or (
            e.get("kind") in {"robogui_home_requested", "robogui_reset_requested"}
        )
        if restart:
            i = (
                int(np.searchsorted(times, e["monotonic_ns"]))
                if "monotonic_ns" in e
                else int(e.get("step", -1))
            )
            if 0 <= i < len(times):
                breaks[i] = True
    references, commands = {}, {}
    for g in groups:
        references[g] = np.asarray(ticks["scheduled"][g][:], float)
        commands[g] = np.asarray(ticks["command"][g][:], float)
        if (
            references[g].ndim != 2
            or references[g].shape != commands[g].shape
            or len(references[g]) != len(times)
            or not np.isfinite(references[g]).all()
            or not np.isfinite(commands[g]).all()
        ):
            raise ValueError("invalid_tick_reference_or_command")
        digest.update(g.encode() + references[g].tobytes() + commands[g].tobytes())
    valid = np.ones(len(times), bool)
    reasons = np.full(len(times), "", dtype=object)
    corrected = np.zeros(len(times), bool)
    for i, pid in enumerate(ids):
        if pid not in plans:
            valid[i], reasons[i] = False, "no_plan_identity" if not pid else "unknown_plan_identity"
            continue
        plan = plans[pid]
        if times[i] < plan.start_ns:
            valid[i], reasons[i] = False, "before_committed_start"
            continue
        for g in groups:
            end = plan.start_ns + (len(plan.groups[g]) - 1) * plan.dt_ns
            if times[i] <= end:
                continue
            # Expired references can be measured-state placeholders. Accept only
            # a recorded constant command, equal to the last command or endpoint.
            command = commands[g][i]
            held = (
                i > 0
                and ids[i - 1] == pid
                and not breaks[i]
                and np.allclose(command, commands[g][i - 1], atol=1e-7, rtol=0)
            )
            # Pose endpoints are not joint commands, even for 8-DOF groups.
            # A pose plan's expired interval needs an observed constant hold.
            endpoint = plan.action_space == "joint_position" and np.allclose(
                command, plan.groups[g][-1], atol=1e-7, rtol=0
            )
            if not (held or endpoint):
                valid[i], reasons[i] = False, "unverified_expired_plan_hold"
            elif not np.allclose(references[g][i], command, atol=1e-7, rtol=0):
                references[g][i] = command
                corrected[i] = True
    return dict(
        times=times,
        ids=ids,
        plans=plans,
        events=events,
        breaks=breaks,
        references=references,
        commands=commands,
        valid=valid,
        reasons=reasons,
        corrected=corrected,
        input_sha256=digest.hexdigest(),
    )


def segments(evidence):
    """All maximal valid runs; never interpolate across pause or missing identity."""
    active = None
    result = []
    for i, valid in enumerate(evidence["valid"]):
        if active is not None and (not valid or evidence["breaks"][i]):
            result.append((active, i))
            active = None
        if valid and active is None:
            active = i
    if active is not None:
        result.append((active, len(evidence["times"])))
    return result


def sample(plan, group, time_ns):
    values = plan.groups[group]
    end = plan.start_ns + (len(values) - 1) * plan.dt_ns
    if not plan.start_ns <= time_ns <= end:
        raise ValueError("sample_outside_plan")
    return _sample(values, np.array([time_ns - plan.start_ns]), plan.dt_ns, plan.action_space)[0]
