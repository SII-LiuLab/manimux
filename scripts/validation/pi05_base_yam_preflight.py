#!/usr/bin/env python3
"""Run real-camera, measured-state Pi05 inference without sending robot commands."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.policies import build_policy_adapter, build_policy_model
from manimux.policies.base import action_interval, prepare_policy_request
from manimux.robots import build_robot
from manimux.runtime.rtc.mask import inpainting_condition
from manimux.runtime.rtc.request import RtcInferenceRequest
from manimux.sensors import build_sensor
from manimux.types import ActionContext, InferenceRequest, ObservationSnapshot

DEFAULT_CONFIG = Path("configs/pi05/yam/infra/base-rtc.yaml")


def _request(
    session_id: str,
    sequence: int,
    snapshot: ObservationSnapshot,
    instruction: str,
) -> InferenceRequest:
    return InferenceRequest(
        session_id=session_id,
        request_seq=sequence,
        observation_time_ns=snapshot.state.monotonic_ns,
        deadline_ns=time.monotonic_ns() + 120_000_000_000,
        observation=snapshot,
        instruction=instruction,
    )


def _decode(model: Any, adapter: Any, request: InferenceRequest) -> tuple[Any, float]:
    started = time.monotonic()
    raw = model.infer(prepare_policy_request(adapter, request))
    elapsed_s = time.monotonic() - started
    chunk = adapter.decode_action(
        raw,
        ActionContext(
            request_seq=request.request_seq,
            observation_time_ns=request.observation_time_ns,
            created_time_ns=time.monotonic_ns(),
        ),
    )
    return chunk, elapsed_s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    robot_config = deepcopy(config["robot"])
    robot_config["options"]["move_to_start_on_connect"] = False
    robot_config["options"]["home_on_close"] = False
    policy_config = deepcopy(config["policy"])
    policy_config["options"]["request_timeout_s"] = 120.0

    clock = SystemClock()
    robot = build_robot(robot_config, clock)
    sensor = build_sensor(config["sensors"][0], clock)
    model = build_policy_model(policy_config)
    adapter = build_policy_adapter(robot_config, policy_config)
    session_id = f"pi05-preflight-{uuid.uuid4().hex[:8]}"

    try:
        robot.connect()
        sensor.start()
        model.reset(session_id)
        state = robot.get_state()
        frames = sensor.read()
        snapshot = ObservationSnapshot(state=state, frames=frames)

        first_request = _request(session_id, 1, snapshot, config["run"]["task"])
        first_chunk, compile_latency_s = _decode(model, adapter, first_request)
        second_request = _request(session_id, 2, snapshot, config["run"]["task"])
        steady_chunk, steady_latency_s = _decode(model, adapter, second_request)

        rows = np.concatenate(
            [steady_chunk.groups[name] for name in ("left_arm", "right_arm")],
            axis=1,
        )
        measured = np.concatenate([state.groups[name] for name in ("left_arm", "right_arm")])
        arm_indices = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
        gripper_indices = np.array([6, 13])
        first_joint_delta = float(np.max(np.abs(rows[0, arm_indices] - measured[arm_indices])))
        grippers = rows[:, gripper_indices]

        contract_checks = {
            "shape_matches_config": rows.shape
            == (policy_config["horizon_steps"], sum(robot_config["group_dims"].values())),
            "finite": bool(np.isfinite(rows).all()),
            "grippers_in_0_1": bool(np.all((grippers >= 0.0) & (grippers <= 1.0))),
            "absolute_position_limit": bool(np.all(np.abs(rows[:, arm_indices]) <= 3.14)),
        }
        report = {
            "contract_checks": contract_checks,
            "measured_state": measured.tolist(),
            "first_action": rows[0].tolist(),
            "max_first_joint_delta_rad": first_joint_delta,
            "gripper_range": [float(grippers.min()), float(grippers.max())],
            "latency_s": {
                "unconditioned_compile": compile_latency_s,
                "unconditioned_steady": steady_latency_s,
            },
            "camera_shapes": {name: list(frame.data.shape) for name, frame in frames.items()},
        }
        if config["execution"]["runtime"] == "rtc":
            delay_steps = max(
                1,
                math.ceil(steady_latency_s / action_interval(policy_config)),
            )
            executed_steps = max(config["execution"]["rtc"]["min_execute_steps"] or 1, delay_steps)
            if not delay_steps <= executed_steps <= len(rows) - delay_steps:
                raise RuntimeError(
                    "steady inference latency is not RTC-feasible: "
                    f"d={delay_steps}, s={executed_steps}, H={len(rows)}"
                )
            condition, weights = inpainting_condition(
                rows,
                executed_steps=executed_steps,
                delay_steps=delay_steps,
            )
            rtc_request = RtcInferenceRequest(
                session_id=session_id,
                request_seq=3,
                observation_time_ns=state.monotonic_ns,
                deadline_ns=time.monotonic_ns() + 120_000_000_000,
                observation=snapshot,
                instruction=config["run"]["task"],
                action_condition=condition.astype(np.float64),
                condition_weights=weights.astype(np.float64),
                rtc_beta=config["execution"]["rtc"]["beta"],
            )
            _, rtc_compile_latency_s = _decode(model, adapter, rtc_request)
            rtc_request = dataclasses.replace(rtc_request, request_seq=4)
            _, rtc_steady_latency_s = _decode(model, adapter, rtc_request)
            contract_checks["rtc_steady_feasible"] = (
                math.ceil(rtc_steady_latency_s / action_interval(policy_config)) <= len(rows) // 2
            )
            report["latency_s"].update(
                rtc_compile=rtc_compile_latency_s,
                rtc_steady=rtc_steady_latency_s,
            )
            report["rtc_alignment"] = {
                "delay_steps": delay_steps,
                "executed_steps": executed_steps,
                "horizon_steps": len(rows),
            }
        print(json.dumps(report, indent=2))
        return 0
    finally:
        model.close()
        sensor.close()
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
