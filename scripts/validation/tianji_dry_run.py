"""Read-only Tianji bring-up: stream the real joint and gripper state.

Nothing is enabled or commanded: the assembly's ``execute`` flag is forced
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

from manimux.cli import load_config
from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot

REPO = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "manimux/configs/experiments/pass_ball/umi_dp/tianji_taccap_umi_dp.yaml",
    )
    parser.add_argument("--local", type=Path, help="local controller and component bindings")
    parser.add_argument("--robot-ip", help="override robot.options.hardware.ip")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--viewer", action="store_true", help="publish states to the viewer")
    parser.add_argument("--viewer-endpoint", default="tcp://127.0.0.1:5568")
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, local=args.local)
    options = config["robot"]["options"]
    # 复用整机装配与设备绑定；此入口只读反馈，不启用机械臂或夹爪控制。
    options["execute"] = False
    options["end_effector_control"] = False
    if args.robot_ip:
        options.setdefault("hardware", {})["ip"] = args.robot_ip
    robot = build_robot(config["robot"], SystemClock())
    client = None
    if args.viewer:
        from manimux.viewer.publisher import ViewerClient

        client = ViewerClient(
            robot="tianji-taccap", policy="read-only bring-up", endpoint=args.viewer_endpoint
        )

    steps = max(1, int(args.seconds * args.hz))
    period = 1.0 / args.hz
    try:
        robot.connect()
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
                poses = robot.fk(state.groups)
                for name in state.groups:
                    values = state.groups[name]
                    joints = np.array2string(np.degrees(values[:7]), precision=2)
                    tcp_mm = np.array2string(poses[name][:3, 3] * 1e3, precision=1)
                    aperture = f" aperture {values[7]:.3f}" if values.size > 7 else ""
                    print(f"{name:9s} joints deg {joints}  tcp mm {tcp_mm}{aperture}")
            time.sleep(max(0.0, started + (step + 1) * period - time.monotonic()))
    finally:
        robot.close()
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
