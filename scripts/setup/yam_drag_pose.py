"""Hand-guide YAM arms in gravity compensation and record joint poses for initial_poses.

Commands, one per line on stdin or --command-fifo:
  record NAME   save every arm's measured joints under NAME
  release       close CAN; arms away from zero Home must be physically supported first
Live state is rewritten to <out>/latest.json every 0.5 s. Ctrl-C does not release.
"""

import argparse
import json
import os
import select
import signal
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from manimux.embodiments.arm.yam.arm import YamController
from manimux.embodiments.arm.yam.kinematics import YamManipulatorKinematics

ROOT = Path(__file__).resolve().parents[2]
ARM_CONFIG = ROOT / "manimux/configs/embodiment/arm/yam.yaml"


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


class DragPose:
    def __init__(self, channels, out):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.poses_path = out / "poses.json"
        self.poses = json.loads(self.poses_path.read_text()) if self.poses_path.exists() else {}
        self.kin = YamManipulatorKinematics().solver
        self.data = mujoco.MjData(self.kin.model)
        hardware = yaml.safe_load(ARM_CONFIG.read_text())["hardware"]
        self.controllers = {}
        for group, channel in channels.items():
            controller = YamController(channel=channel, **hardware, zero_gravity_mode=True)
            controller.connect()
            self.controllers[group] = controller
            self.set_drag(controller)

    @staticmethod
    def set_drag(controller):
        # Arm joints: zero stiffness plus SDK gravity compensation. The gripper keeps its gains.
        native = controller.robot
        controller.check_control_health()
        info = native.get_robot_info()
        kp = np.asarray(info["kp"], dtype=float).copy()
        kd = np.asarray(info["grav_comp_kd"], dtype=float).copy()
        kp[:6] = 0.0
        kd[6] = np.asarray(info["kd"])[6]
        q = native.get_joint_pos().copy()
        native.command_joint_state({"pos": q, "vel": np.zeros(7), "kp": kp, "kd": kd})

    def arm_status(self, q):
        # Smallest singular value of the grasp-site position Jacobian: how well the
        # weakest Cartesian direction can still move (m/rad). Zero Home is ~0.044.
        self.data.qpos[:] = self.kin.robot_state_to_qpos(q[:6], q[6])
        mujoco.mj_forward(self.kin.model, self.data)
        jp = np.zeros((3, self.kin.model.nv))
        mujoco.mj_jacSite(self.kin.model, self.data, jp, None, self.kin.site_id)
        sigma = np.linalg.svd(jp[:, :6], compute_uv=False)
        return {
            "q": np.round(q, 4).tolist(),
            "grasp_site_xyz": np.round(self.data.site_xpos[self.kin.site_id], 4).tolist(),
            "sigma_min": round(float(sigma[-1]), 3),
        }

    def status(self):
        return {
            group: self.arm_status(controller.check_control_health().pos.copy())
            for group, controller in self.controllers.items()
        }

    def record(self, name):
        samples = {group: [] for group in self.controllers}
        for _ in range(10):
            for group, controller in self.controllers.items():
                samples[group].append(controller.check_control_health().pos.copy())
            time.sleep(0.01)
        joints = {group: np.mean(values, axis=0) for group, values in samples.items()}
        self.poses[name] = {
            "time_ns": time.time_ns(),
            "joints": {group: q.tolist() for group, q in joints.items()},
            "status": {group: self.arm_status(q) for group, q in joints.items()},
        }
        self.poses_path.write_text(json.dumps(self.poses, indent=2))
        snippet = {name: {
            "label": name,
            "duration_s": 5.0,
            "joints": {group: [round(float(x), 4) for x in q] for group, q in joints.items()},
            "source": {"fields": "measured joint_pos and gripper_pos after hand-guided drag"},
        }}
        return {"recorded": name, **self.poses[name]["status"], "initial_pose_yaml": yaml.safe_dump(snippet, sort_keys=False)}

    def release(self):
        for controller in self.controllers.values():
            if controller._home_error(controller.check_control_health()):
                controller.confirm_supported_release()
            controller.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", type=Path, default=ROOT / "manimux/configs/local/station.yaml")
    parser.add_argument("--arms", nargs="+", choices=["left", "right"], default=["left", "right"])
    parser.add_argument("--out", type=Path, default=ROOT / ".local/drag-poses")
    parser.add_argument("--command-fifo", type=Path)
    args = parser.parse_args()

    components = yaml.safe_load(args.local.read_text())["robot"]["components"]
    channels = {f"{side}_arm": components[f"{side}_yam"]["channel"] for side in args.arms}
    if args.command_fifo is not None and not args.command_fifo.exists():
        os.mkfifo(args.command_fifo)
    # O_RDWR keeps the FIFO open with no writer, so readers never see EOF.
    commands = (
        os.fdopen(os.open(args.command_fifo, os.O_RDWR), "r")
        if args.command_fifo is not None
        else sys.stdin
    )

    session = DragPose(channels, args.out)
    signal.signal(signal.SIGINT, lambda *_: emit({"message": "Ctrl-C ignored; send release after supporting the arms"}))
    signal.signal(signal.SIGTERM, lambda *_: emit({"message": "SIGTERM ignored; send release after supporting the arms"}))
    emit({"drag": True, "channels": channels, **session.status()})
    while True:
        ready, _, _ = select.select([commands], [], [], 0.5)
        if not ready:
            (args.out / "latest.json").write_text(json.dumps(session.status(), indent=2))
            continue
        line = commands.readline().strip()
        if not line:
            continue
        op, _, name = line.partition(" ")
        try:
            if op == "record" and name:
                emit(session.record(name.strip()))
            elif op == "release":
                session.release()
                emit({"released": True})
                return
            elif op == "status":
                emit(session.status())
            else:
                emit({"error": "use: record NAME | status | release"})
        except Exception as error:
            emit({"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
