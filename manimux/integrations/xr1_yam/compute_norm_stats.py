"""Compute XR-1 normalization statistics from recorded YAM episodes.

XR-1 does not ship statistics with the checkpoint: upstream's ``deploy.py`` reads
``mean``/``std``/``q01``/``q99`` out of the *training* config, so they always
belong to the same run that produced the weights. The released 5B checkpoint was
never fine-tuned for a robot, so no such file exists for it, and the demo
statistics in the upstream repo describe a different machine with different
gripper units.

This script derives the statistics from our own YAM recordings using exactly the
delta construction upstream trains on (``json_dataset.py``)::

    rotm   = proprio.{arm}_ee_rotm[t]
    pos    = proprio.{arm}_ee_pos[t]
    dpos   = rotm.T @ (action.{arm}_ee_pos[t:t+H] - pos)
    daa    = rotm2aa(rotm.T @ action.{arm}_ee_rotm[t:t+H])
    dgrip  = action.{arm}_gripper[t:t+H] - proprio.{arm}_gripper[t]

End-effector poses come from :mod:`manimux.kinematics`, which reproduces the
recorder's forward kinematics exactly.

Usage::

    envs/yam/.venv/bin/python -m manimux.integrations.xr1_yam.compute_norm_stats \\
        --episodes /path/to/data/episodes --out yam.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manimux.integrations.xr1_yam.mibot.utils.io import (
    ACTION_DIM,
    ACTION_PARTS,
    STATE_DIM,
    rotm2aa_batch,
)

ACTION_LENGTH = 30
ARM_JOINTS = 6
# State slot layout from ``compose_state``: 7 wide per arm, joints left-aligned.
STATE_SLOTS = {
    "left": {"joints": slice(0, ARM_JOINTS), "gripper": 7},
    "right": {"joints": slice(8, 8 + ARM_JOINTS), "gripper": 15},
}
REQUIRED = (
    "left-joint_pos.npy",
    "left-gripper_pos.npy",
    "right-joint_pos.npy",
    "right-gripper_pos.npy",
    "action-left-joint.npy",
    "action-left-gripper.npy",
    "action-right-joint.npy",
    "action-right-gripper.npy",
)


def usable_episodes(root: Path) -> list[Path]:
    episodes = []
    for candidate in sorted(root.glob("*/*/")):
        if not (candidate / "write_complete.flag").exists():
            continue
        if all((candidate / name).exists() for name in REQUIRED):
            episodes.append(candidate)
    return episodes


def _poses(kinematics, joints: np.ndarray, grippers: np.ndarray) -> np.ndarray:
    transforms = np.empty((len(joints), 4, 4), dtype=np.float64)
    for index, (joint, gripper) in enumerate(zip(joints, grippers, strict=True)):
        transforms[index] = kinematics.fk(joint, float(gripper[0]))
    return transforms


def episode_samples(
    kinematics,
    episode: Path,
    horizon: int = ACTION_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(states, actions)`` shaped ``(n, 60)`` and ``(n, horizon, 60)``."""
    data = {name[:-4]: np.load(episode / name) for name in REQUIRED}
    length = len(data["left-joint_pos"])
    if length <= horizon:
        return np.zeros((0, STATE_DIM)), np.zeros((0, horizon, ACTION_DIM))

    poses = {}
    for arm in ("left", "right"):
        poses[arm] = {
            "proprio": _poses(
                kinematics, data[f"{arm}-joint_pos"], data[f"{arm}-gripper_pos"]
            ),
            "action": _poses(
                kinematics, data[f"action-{arm}-joint"], data[f"action-{arm}-gripper"]
            ),
        }

    count = length - horizon
    states = np.zeros((count, STATE_DIM), dtype=np.float32)
    actions = np.zeros((count, horizon, ACTION_DIM), dtype=np.float32)
    parts = dict(ACTION_PARTS)

    for arm in ("left", "right"):
        slots = STATE_SLOTS[arm]
        proprio, command = poses[arm]["proprio"], poses[arm]["action"]
        joints = data[f"{arm}-joint_pos"]
        grip_state = data[f"{arm}-gripper_pos"][:, 0]
        grip_action = data[f"action-{arm}-gripper"][:, 0]

        for frame in range(count):
            window = slice(frame, frame + horizon)
            rotation = proprio[frame, :3, :3]
            position = proprio[frame, :3, 3]

            states[frame, slots["joints"]] = joints[frame]
            states[frame, slots["gripper"]] = grip_state[frame]

            delta_pos = (command[window, :3, 3] - position) @ rotation
            delta_rot = rotm2aa_batch(rotation.T @ command[window, :3, :3])
            actions[frame, :, parts[f"{arm}_ee_pos"]] = delta_pos
            actions[frame, :, parts[f"{arm}_ee_aa"]] = delta_rot
            actions[frame, :, parts[f"{arm}_gripper"]] = (
                grip_action[window] - grip_state[frame]
            )[:, None]

    return states, actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        default="/home/ubuntu/yam-abc-reproduce/data/episodes",
        help="root directory holding <task>/<episode>/ recordings",
    )
    parser.add_argument("--out", required=True, help="destination JSON file")
    parser.add_argument("--horizon", type=int, default=ACTION_LENGTH)
    args = parser.parse_args()

    from manimux.kinematics import build_kinematics

    kinematics = build_kinematics("yam")
    episodes = usable_episodes(Path(args.episodes))
    if not episodes:
        raise SystemExit(f"no usable episodes under {args.episodes}")

    all_states, all_actions = [], []
    for index, episode in enumerate(episodes, 1):
        states, actions = episode_samples(kinematics, episode, args.horizon)
        if len(states):
            all_states.append(states)
            all_actions.append(actions)
        print(f"[{index}/{len(episodes)}] {episode.parent.name}/{episode.name}: {len(states)}")

    states = np.concatenate(all_states)
    actions = np.concatenate(all_actions)
    print(f"\nstates {states.shape}  actions {actions.shape}")

    # Per-step statistics, exactly the shapes upstream's validate_stats expects.
    mean = actions.mean(axis=0)
    std = actions.std(axis=0)
    q01 = np.quantile(states, 0.01, axis=0)[None]
    q99 = np.quantile(states, 0.99, axis=0)[None]

    # Columns this embodiment never drives (waist, mobile base, reserved) must
    # stay dead: leaving q99 == q01 makes the server zero them, as upstream does
    # for a robot without those degrees of freedom.
    live = np.zeros(STATE_DIM, dtype=bool)
    for arm in ("left", "right"):
        live[STATE_SLOTS[arm]["joints"]] = True
        live[STATE_SLOTS[arm]["gripper"]] = True
    q01[0, ~live] = 0.0
    q99[0, ~live] = 0.0

    payload = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "action_length": args.horizon,
        "_source": (
            f"computed from {len(episodes)} YAM episodes ({len(states)} windows) under "
            f"{args.episodes} by manimux.integrations.xr1_yam.compute_norm_stats"
        ),
    }
    Path(args.out).write_text(json.dumps(payload))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
