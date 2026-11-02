"""Derive the DLS singularity_gate for a YAM station from its own demonstrations.

The gate compares each IK seed's smallest 6D Jacobian singular value (sigma_min)
with [low, high]. Its scale depends on the TCP site and on where the station's tasks
place the arm, so take it from joint trajectories recorded on that station:
teleoperation demonstrations (what the policy was trained on) or ManiMux rollouts.

Inputs, any mix:
  *.npy                 one arm's joints, shape (T, 6) or (T, 7) with the gripper last;
                        a sibling *gripper_pos* file is appended to a (T, 6) *joint_pos* file
  <rollout>/data.zarr   ManiMux recording; every ticks/state/<group> with 7 columns
  directory             searched recursively for *joint_pos*.npy and data.zarr

The suggested gate is low = the 10th percentile of sigma_min over all frames and
high = 2 * low: the regularizer is fully on for the lowest tenth of ordinary poses
and off above twice that, where tracking is exact like mink. On the reference YAM
station (80 teleop arm trajectories, grasp_site) this gives [0.030, 0.060], the gate
that kept lift-off and wrist damping equal to the always-on cost of 0.1 while cutting
the p95 error near gripper events from 1.3 mm to 0.06 mm. Validate any other
station's gate on its own lift-off and wrist-limit motions before relying on it.
"""

import argparse
from pathlib import Path

import numpy as np

from manimux.embodiments.arm.yam.kinematics import YamManipulatorKinematics


def trajectories(paths):
    for path in map(Path, paths):
        if path.is_dir() and path.name != "data.zarr":
            yield from trajectories(sorted(path.rglob("*joint_pos*.npy")) + sorted(path.rglob("data.zarr")))
        elif path.suffix == ".npy":
            joints = np.load(path)
            if joints.ndim != 2 or joints.shape[1] < 6:
                raise ValueError(f"{path}: expected (T, 6|7) joints, got {joints.shape}")
            gripper = path.with_name(path.name.replace("joint_pos", "gripper_pos"))
            if joints.shape[1] == 6 and gripper != path and gripper.exists():
                joints = np.c_[joints, np.load(gripper).reshape(len(joints), -1)[:, 0]]
            yield str(path), joints
        elif path.name == "data.zarr":
            import zarr

            states = zarr.open(str(path), "r")["ticks/state"]
            for group in states:
                joints = np.asarray(states[group])
                if joints.ndim == 2 and joints.shape[1] == 7:
                    yield f"{path}:{group}", joints
        else:
            raise ValueError(f"{path}: not a .npy file, data.zarr or directory")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--ee-site", default="grasp_site", help="IK control site (the arm's TCP)")
    parser.add_argument("--stride", type=int, default=5, help="use every Nth frame")
    parser.add_argument("--percentile", type=float, default=10.0, help="low = this sigma_min percentile")
    parser.add_argument("--ratio", type=float, default=2.0, help="high = ratio * low")
    args = parser.parse_args()

    solver = YamManipulatorKinematics(ee_site=args.ee_site).solver
    sigmas, starts, sources = [], [], 0
    for _, joints in trajectories(args.inputs):
        gripper = joints[:, 6] if joints.shape[1] > 6 else np.ones(len(joints))
        values = [solver.min_singular_value(q[:6], g) for q, g in zip(joints[::args.stride], gripper[::args.stride])]
        sigmas += values
        starts.append(values[0])
        sources += 1
    if not sigmas:
        raise SystemExit("no trajectories found")
    sigmas = np.asarray(sigmas)
    table = {p: float(np.percentile(sigmas, p)) for p in (1, 5, 10, 25, 30, 50, 75)}
    low = float(np.percentile(sigmas, args.percentile))
    high = args.ratio * low
    print(f"{sources} trajectories, {len(sigmas)} frames (stride {args.stride}), ee_site {args.ee_site}")
    print("sigma_min percentiles: " + "  ".join(f"p{p} {v:.3f}" for p, v in table.items()))
    print(f"trajectory start sigma_min: median {np.median(starts):.3f}, max {np.max(starts):.3f}")
    share = float(np.mean(sigmas >= high))
    print(f"\nsingularity_gate: [{low:.3f}, {high:.3f}]  "
          f"# p{args.percentile:g} and {args.ratio:g}x of {len(sigmas)} frames; {share:.0%} of frames ungated")


if __name__ == "__main__":
    main()
