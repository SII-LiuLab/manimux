#!/usr/bin/env python3
"""Time UMI Tianji pose decoding through the real IK library without hardware."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    from manimux.cli import load_config
    from manimux.policy_adapter.umi_dp.history import WindowSnapshot
    from manimux.policy_adapter.umi_dp.ik_config import bind_diff_ik_profile
    from manimux.policy_adapter.umi_dp.tianji import UmiDpTianjiAdapter, matrix_pose
    from manimux.types import (
        ActionContext,
        InferenceRequest,
        ObservationSnapshot,
        RobotState,
        SensorFrame,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "manimux/configs/experiments/pass_ball/tianji_umi_dp_default.yaml",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ik-backend", choices=("analytic", "diff"))
    args = parser.parse_args()
    if args.repeat < 1 or any(horizon < 1 for horizon in args.horizons):
        parser.error("repeat and horizons must be positive")
    config = load_config(args.config)
    if args.ik_backend:
        config["policy"]["adapter"]["ik_backend"] = args.ik_backend
    bind_diff_ik_profile(config)
    # Only kinematics is constructed: neither a RobotBase nor sensor is opened.
    config["robot"]["type"] = "mock"
    report = {}
    for horizon in args.horizons:
        config["policy"]["horizon_steps"] = horizon
        adapter = UmiDpTianjiAdapter(config["robot"], config["policy"])
        start = np.radians([50, -40, -30, -100, -65, 0, 40])
        state = RobotState(
            {side + "_arm": np.r_[start, 0.8] for side in ("left", "right")}, 1000000000, 1
        )
        previous = RobotState(
            {key: value.copy() for key, value in state.groups.items()}, 900000000, 0
        )
        camera_names = tuple(dict.fromkeys(config["policy"]["adapter"]["camera_map"].values()))
        frames = {
            name: SensorFrame(name, np.zeros((8, 8, 3), np.uint8), state.monotonic_ns, 1)
            for name in camera_names
        }
        window = WindowSnapshot(state, frames, ObservationSnapshot(previous, frames))
        actions = []
        for index in range(horizon):
            step = {}
            for side in ("left", "right"):
                joints = start.copy()
                joints[0] += np.radians(0.1 * (index + 1))
                step[side + "_ee_pose"] = matrix_pose(adapter.kin[side].fk(joints, 0.8))
                step[side + "_ee_joint_state"] = np.array([0.8])
            actions.append(step)
        timings = []
        for repeat in range(args.repeat):
            adapter.prepare_request(InferenceRequest("bench", repeat, 1000000000, 10**12, window))
            begin = time.perf_counter()
            chunk = adapter.decode_action(
                {"actions": actions},
                ActionContext(
                    repeat,
                    state.monotonic_ns,
                    state.monotonic_ns,
                    measured_state=state,
                ),
            )
            timings.append((time.perf_counter() - begin) * 1000)
        assert chunk.horizon_steps == horizon
        report[str(horizon)] = {
            "decode_ms": timings,
            "hardware_connected": False,
            "targets": "small reachable synthetic FK poses",
            "ik_backend": adapter.ik_backend,
            "ik_library": "vendored libKine" if adapter.ik_backend == "analytic" else "OSQP",
        }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
