#!/usr/bin/env python3
"""Read measured Flexiv joints and base-frame flange FK without commanding motion.

This is the first live-hardware gate for a Flexiv station. It refuses execution,
enable-on-connect and any send/stop path, connects each configured arm, samples
fresh feedback, prints the measured joint vector and offline FK pose, and closes
the SDK session. No camera, model server or policy is used.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot
from manimux.kinematics.poses import matrix_pose


def _refuse_active_config(config: dict) -> None:
    options = config["robot"].get("options", {})
    if options.get("execute") is not False:
        raise SystemExit(
            "refusing: robot.options.execute must be false for the read-only probe"
        )
    enabled = [
        name
        for name, binding in options.get("component_hardware", {}).items()
        if binding.get("enable_on_connect") is True
    ]
    if enabled:
        raise SystemExit(
            "refusing: enable_on_connect is true for " + ", ".join(sorted(enabled))
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=".local/flexiv/joint.yaml",
        help="Deployment YAML; its execute option must be false",
    )
    parser.add_argument(
        "--local",
        default=None,
        help="Station YAML; omitted means the experiment's local reference or the default station",
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--json", action="store_true", help="emit one JSON object per sample")
    args = parser.parse_args()
    if args.seconds <= 0 or args.hz <= 0:
        parser.error("--seconds and --hz must be positive")

    config = load_config(args.config, local=args.local)
    _refuse_active_config(config)
    robot = build_robot(config["robot"], SystemClock())

    # These checks run before connect and fail closed if the assembled controller
    # would enable or dispatch during this probe.
    for name, arm in robot.arm_components.items():
        controller = arm.controller
        if getattr(controller, "execute", None) is not False:
            raise SystemExit(f"refusing: {name} controller is not read-only")
        if getattr(controller, "enable_on_connect", None) is True:
            raise SystemExit(f"refusing: {name} controller enables on connect")

    connected = False
    try:
        robot.connect()
        connected = True
        print(
            json.dumps(
                {
                    "event": "connected",
                    "robot": robot.model.name if robot.model is not None else type(robot).__name__,
                    "groups": list(robot.arm_components),
                    "execute": config["robot"]["options"].get("execute"),
                    "metadata": robot.runtime_metadata(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        period_s = 1.0 / args.hz
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            state = robot.get_state()
            poses = robot.fk(state.groups)
            sample = {
                "event": "sample",
                "state_time_ns": state.monotonic_ns,
                "sequence": state.sequence,
                "groups": {
                    name: {
                        "q_rad": np.asarray(values, dtype=float).tolist(),
                        "fk_base_flange_xyz_wxyz": matrix_pose(poses[name]).tolist(),
                    }
                    for name, values in state.groups.items()
                },
            }
            if args.json:
                print(json.dumps(sample, sort_keys=True), flush=True)
            else:
                for name, values in sample["groups"].items():
                    q = np.round(values["q_rad"], 6).tolist()
                    xyz = np.round(values["fk_base_flange_xyz_wxyz"][:3], 6).tolist()
                    print(f"{name}: q_rad={q} fk_xyz_m={xyz}", flush=True)
            time.sleep(period_s)
    finally:
        # close() is read-only here: FlexivController.stop() returns without
        # an SDK Stop because no session was enabled or commanded.
        robot.close()
        if connected:
            print(json.dumps({"event": "closed"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
