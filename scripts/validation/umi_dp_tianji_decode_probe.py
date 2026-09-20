#!/usr/bin/env python3
"""Time UMI Tianji pose decoding through the real IK library without hardware."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main():
    from manimux.cli import load_config
    from manimux.integrations.umi_dp_tianji.history import WindowSnapshot
    from manimux.integrations.umi_dp_tianji.ik_config import bind_diff_ik_profile
    from manimux.integrations.umi_dp_tianji.policy_plugin import UmiDpTianjiAdapter, matrix_pose
    from manimux.policies.base import action_interval
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
        default=REPO / "configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml",
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
        config["policy"]["options"]["ik_backend"] = args.ik_backend
    if config["policy"]["options"].get("ik_backend") == "diff":
        config["policy"]["options"]["diff_ik"] = yaml.safe_load(
            (REPO / "configs/policy/umi_dp/adapter/tianji_diff.yaml").read_text()
        )
    bind_diff_ik_profile(config)
    # Only the assembly's offline model is constructed; no hardware component is opened.
    report = {}
    for horizon in args.horizons:
        config["policy"]["horizon_steps"] = horizon
        config["policy"]["options"]["deployment_bound"] = True
        config["policy"]["expected_backend"]["model"].update(
            checkpoint_sha256="offline-probe",
            training_config_sha256="offline-probe",
            checkpoint_path="offline-probe",
            weight_key="ema",
            rgb_normalize=True,
            action_horizon=horizon,
            action_dt_s=action_interval(config["policy"]),
            first_action_offset_s=config["policy"]["options"]["first_action_offset_s"],
            observation_period_s=config["policy"]["options"]["observation_period_s"],
        )
        adapter = UmiDpTianjiAdapter(config["robot"], config["policy"])
        start = np.radians([50, -40, -30, -100, -65, 0, 40])
        state = RobotState(
            {side + "_arm": np.r_[start, 0.8] for side in ("left", "right")}, 1000000000, 1
        )
        previous = RobotState(
            {key: value.copy() for key, value in state.groups.items()}, 900000000, 0
        )
        frames = {
            name: SensorFrame(name, np.zeros((8, 8, 3), np.uint8), state.monotonic_ns, 1)
            for name in ("left_wrist", "right_wrist", "left_wrist_prev", "right_wrist_prev")
        }
        window = WindowSnapshot(state, frames, ObservationSnapshot(previous, frames))
        actions = []
        for index in range(horizon):
            step = {}
            for side in ("left", "right"):
                joints = start.copy()
                joints[0] += np.radians(0.1 * (index + 1))
                step[side + "_ee_pose"] = matrix_pose(
                    adapter.kin[side].fk(np.r_[joints, 0.8])
                )
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
                    1000000000,
                    1000000000,
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
