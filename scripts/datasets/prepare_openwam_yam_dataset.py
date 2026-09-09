#!/usr/bin/env python3
"""Export recorded YAM achieved poses to OpenWAM native real HDF5.

Actions follow the upstream reader: future achieved states, NOT recorded
controller commands. All input episodes must belong to the training split.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from XPolicyLab.utils.process_data import images_encoding

CAMERAS = {
    "cam_head": "top-images-rgb.mp4",
    "cam_left_wrist": "left-images-rgb.mp4",
    "cam_right_wrist": "right-images-rgb.mp4",
}


def export_episode(source, target, instruction, frequency):
    if target.exists():
        raise ValueError(f"Refusing to overwrite {target}")
    metadata = json.loads((source / "metadata.json").read_text())
    if not (source / "write_complete.flag").exists():
        raise ValueError(f"Incomplete recording: {source}")
    if not metadata.get("extra", {}).get("eepose", {}).get("enabled", False):
        raise ValueError("Recording must declare achieved EE poses")
    count = int(metadata["num_frames"])
    if count < 2:
        raise ValueError("At least two frames required")
    state = {}
    for side in ("left", "right"):
        transform = np.load(source / f"{side}-ee_transform.npy", allow_pickle=False)
        grip = np.load(source / f"{side}-gripper_pos.npy", allow_pickle=False)
        if transform.shape != (count, 4, 4) or not np.isfinite(transform).all():
            raise ValueError("Expected finite recorded (T, 4, 4) achieved EE transforms")
        rotation = transform[:, :3, :3]
        if not np.allclose(
            rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-5
        ) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("Recorded rotations must be proper orthogonal matrices")
        if (
            grip.shape != (count, 1)
            or not np.isfinite(grip).all()
            or np.any((grip < 0) | (grip > 1))
        ):
            raise ValueError("Expected recorded grippers in [0, 1]")
        quat = Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]
        state[f"{side}_ee_poses"] = np.concatenate([transform[:, :3, 3], quat], axis=1)
        state[f"{side}_ee_joint_states"] = grip
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial")
    try:
        with h5py.File(temporary, "x") as data:
            data.attrs.update(
                frequency_hz=frequency,
                pose_frame="per_arm_robot_base",
                action_target="next_achieved_state",
                source=str(source.resolve()),
            )
            data.create_dataset("instruction", data=instruction)
            for key, value in state.items():
                data.create_dataset(f"state/{key}", data=value)
            for camera, filename in CAMERAS.items():
                capture = cv2.VideoCapture(str(source / filename))
                column = data.create_dataset(
                    f"vision/{camera}/colors", (count,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
                )
                try:
                    for index in range(count):
                        ok, frame = capture.read()
                        if not ok:
                            raise ValueError(f"Video shorter than state: {filename}")
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        encoded, _ = images_encoding([rgb])
                        column[index] = np.frombuffer(encoded[0], dtype=np.uint8)
                    if capture.read()[0]:
                        raise ValueError(f"Video longer than state: {filename}")
                finally:
                    capture.release()
        temporary.rename(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--frequency", type=float, required=True, help="Recorded frame rate; no resampling"
    )
    args = parser.parse_args()
    if Path(args.task).name != args.task or args.task in {".", ".."}:
        parser.error("task must be a directory name")
    if not np.isfinite(args.frequency) or args.frequency <= 0:
        parser.error("frequency must be positive")
    episodes = sorted(path for path in args.episodes.iterdir() if path.is_dir())
    if not episodes:
        parser.error("No episodes found")
    frames = 0
    for index, episode in enumerate(episodes):
        frames += export_episode(
            episode,
            args.output / args.task / "yam_dual/data" / f"episode_{index:06d}.hdf5",
            args.instruction,
            args.frequency,
        )
    print(
        json.dumps(
            {"episodes": len(episodes), "frames": frames, "action_target": "next_achieved_state"}
        )
    )


if __name__ == "__main__":
    main()
