from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from manimux.kinematics import build_kinematics
from manimux.policies.xpolicylab.aac import ee_pose_increment

GROUP_FILES = {
    "left_arm": (
        "left-joint_pos.npy",
        "left-gripper_pos.npy",
        "action-left-joint.npy",
        "action-left-gripper.npy",
    ),
    "right_arm": (
        "right-joint_pos.npy",
        "right-gripper_pos.npy",
        "action-right-joint.npy",
        "action-right-gripper.npy",
    ),
}
MOTION_HORIZONS = (16, 50)


def _usable_episodes(root: Path) -> list[Path]:
    required = {name for names in GROUP_FILES.values() for name in names}
    return [
        episode
        for episode in sorted(root.glob("*/*/"))
        if (episode / "write_complete.flag").is_file()
        and all((episode / name).is_file() for name in required)
    ]


def _poses(kinematics, joints: np.ndarray, grippers: np.ndarray) -> np.ndarray:
    result = np.empty((len(joints), 4, 4), dtype=np.float64)
    for index, (joint, gripper) in enumerate(zip(joints, grippers, strict=True)):
        result[index] = kinematics.fk(joint, float(np.asarray(gripper).reshape(-1)[0]))
    return result


def _episode_increments(kinematics, episode: Path, files: tuple[str, ...]) -> np.ndarray:
    state_joints, state_grippers, action_joints, action_grippers = (
        np.load(episode / name) for name in files
    )
    lengths = {
        len(state_joints),
        len(state_grippers),
        len(action_joints),
        len(action_grippers),
    }
    if len(lengths) != 1:
        raise ValueError(f"mismatched recording lengths in {episode}: {sorted(lengths)}")
    state_poses = _poses(kinematics, state_joints, state_grippers)
    action_poses = _poses(kinematics, action_joints, action_grippers)

    state_to_action = np.stack(
        [
            ee_pose_increment(state, action)
            for state, action in zip(state_poses, action_poses, strict=True)
        ]
    )
    if len(action_poses) <= 1:
        return state_to_action
    action_to_action = np.stack(
        [
            ee_pose_increment(previous, target)
            for previous, target in zip(action_poses[:-1], action_poses[1:], strict=True)
        ]
    )
    return np.concatenate([state_to_action, action_to_action])


def _episode_motion(kinematics, episode: Path, horizon: int = 16) -> np.ndarray:
    per_arm = []
    for files in GROUP_FILES.values():
        state_joints, state_grippers, action_joints, action_grippers = (
            np.load(episode / name) for name in files
        )
        state_poses = _poses(kinematics, state_joints, state_grippers)
        action_poses = _poses(kinematics, action_joints, action_grippers)
        grippers = action_grippers.reshape(len(action_grippers), -1)[:, 0]
        arm_motion = []
        for start in range(max(0, len(action_poses) - horizon + 1)):
            poses = np.concatenate(
                [state_poses[start : start + 1], action_poses[start : start + horizon]]
            )
            increments = np.stack(
                [
                    ee_pose_increment(previous, target)
                    for previous, target in zip(poses[:-1], poses[1:], strict=True)
                ]
            )
            rotation = Rotation.identity()
            for rotvec in increments[:, 3:6]:
                rotation = Rotation.from_rotvec(rotvec) * rotation
            gripper = grippers[start : start + horizon] >= 0.5
            arm_motion.append(
                np.linalg.norm(np.sum(increments[:, :3], axis=0))
                + np.linalg.norm(rotation.as_rotvec())
                + 0.2 * float(np.any(np.diff(gripper)))
            )
        per_arm.append(np.asarray(arm_motion))
    return np.mean(np.stack(per_arm), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute fixed min/max stats for YAM AAC incremental EE actions."
    )
    parser.add_argument(
        "--episodes",
        default="/home/ubuntu/yam-abc-reproduce/data/episodes",
        help="recording root containing <task>/<episode>/ directories",
    )
    parser.add_argument("--out", required=True, help="destination JSON file")
    args = parser.parse_args()

    root = Path(args.episodes).expanduser().resolve()
    episodes = _usable_episodes(root)
    if not episodes:
        raise SystemExit(f"no complete YAM episodes under {root}")

    kinematics = build_kinematics("yam")
    grouped: dict[str, list[np.ndarray]] = {group: [] for group in GROUP_FILES}
    motion: dict[int, list[np.ndarray]] = {horizon: [] for horizon in MOTION_HORIZONS}
    for index, episode in enumerate(episodes, 1):
        for group, files in GROUP_FILES.items():
            grouped[group].append(_episode_increments(kinematics, episode, files))
        for horizon in MOTION_HORIZONS:
            values = _episode_motion(kinematics, episode, horizon=horizon)
            if values.size:
                motion[horizon].append(values)
        print(f"[{index}/{len(episodes)}] {episode.parent.name}/{episode.name}")

    groups = {}
    for group, parts in grouped.items():
        values = np.concatenate(parts)
        groups[group] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
            "samples": int(len(values)),
        }

    motion_calibration = {}
    for horizon, parts in motion.items():
        if not parts:
            continue
        motion_values = np.concatenate(parts)
        motion_calibration[str(horizon)] = {
            "horizon_steps": horizon,
            "dual_arm_mean_samples": int(len(motion_values)),
            "q10": float(np.quantile(motion_values, 0.10)),
            "q25": float(np.quantile(motion_values, 0.25)),
            "q50": float(np.quantile(motion_values, 0.50)),
            "q75": float(np.quantile(motion_values, 0.75)),
            "q90": float(np.quantile(motion_values, 0.90)),
            "q99": float(np.quantile(motion_values, 0.99)),
        }
    payload = {
        "format": "manimux.aac.ee_increment_min_max.v1",
        "groups": groups,
        "source": {
            "episodes_root": str(root),
            "episode_count": len(episodes),
            "translation_frame": "yam_arm_base",
            "rotation_convention": "R_target @ R_previous.T then rotation_vector",
            "sample_construction": ["measured_to_action", "action_to_next_action"],
        },
        "motion_calibration": motion_calibration,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
