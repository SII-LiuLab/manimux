"""Read-only Tianji bring-up: stream the real joint and gripper state.

Nothing is enabled or commanded: the control profile's ``execute`` flag is forced
off. With ``--viewer`` each state is also published to a running viewer, so the
digital twin can be compared with the real arms.

    .venv/bin/python scripts/validation/tianji_dry_run.py --seconds 10
    .venv/bin/manimux-viewer --robot tianji          # second terminal
    .venv/bin/python scripts/validation/tianji_dry_run.py --viewer --seconds 120

Quit MarvinPlatform first; it holds the controller's UDP port.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from manimux.cli import control_profile_parameters
from manimux.clock import SystemClock
from manimux.embodiments.robot import robot_parameters
from manimux.kinematics.tianji import TianjiKinematics
from manimux.robots import build_robot

REPO = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=REPO / "configs/robots/tianji/common.yaml")
    parser.add_argument("--robot-ip", help="override robot.options.robot_ip")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--no-gripper", action="store_true", help="read the arms only")
    parser.add_argument("--viewer", action="store_true", help="publish states to the viewer")
    parser.add_argument("--viewer-endpoint", default="tcp://127.0.0.1:5568")
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    profile = control_profile_parameters(**yaml.safe_load(args.profile.read_text()))
    options = dict(profile["robot"]["options"])
    options["execute"] = False
    group_dims = dict(profile["robot"]["group_dims"])
    end_effector = None if args.no_gripper else str(options.get("end_effector", "umi_follower"))
    if args.no_gripper:
        options["end_effector"] = "none"
        group_dims = {name: 7 for name in group_dims}
    if args.robot_ip:
        options["robot_ip"] = args.robot_ip
    robot = build_robot(
        robot_parameters(driver=profile["robot"].driver, group_dims=group_dims, options=options),
        SystemClock(),
    )
    kinematics = TianjiKinematics(end_effector=end_effector)
    client = None
    if args.viewer:
        from manimux.viewer.publisher import ViewerClient

        client = ViewerClient(
            robot="tianji-taccap", policy="read-only bring-up", endpoint=args.viewer_endpoint
        )

    steps = max(1, int(args.seconds * args.hz))
    period = 1.0 / args.hz
    robot.connect()
    try:
        started = time.monotonic()
        for step in range(steps):
            state = robot.get_state()
            if client is not None:
                client.step_executed(
                    groups=state.groups,
                    cameras={},
                    step=step,
                    max_steps=steps,
                    action_index=0,
                    chunk_id=0,
                )
            if step % max(1, int(args.hz)) == 0:
                for name in ("left_arm", "right_arm"):
                    values = state.groups[name]
                    joints = np.array2string(np.degrees(values[:7]), precision=2)
                    tcp_mm = np.array2string(kinematics.pose(values)[:3, 3] * 1e3, precision=1)
                    aperture = f" aperture {values[7]:.3f}" if values.size > 7 else ""
                    print(f"{name:9s} joints deg {joints}  tcp mm {tcp_mm}{aperture}")
            time.sleep(max(0.0, started + (step + 1) * period - time.monotonic()))
    finally:
        robot.close()
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
