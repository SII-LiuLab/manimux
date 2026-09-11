#!/usr/bin/env python3
"""Convert recorded YAM episodes into the LeRobot v3 dataset used by Pi05 and LingBot-VLA2.

This is a standalone copy of the conversion path used for
``yam_assemble_screwdriver_20260825_v1``.  It preserves the recorded data
contract:

* measured joints/grippers become ``observation.state``;
* commanded joints/grippers become ``action``;
* camera videos become ``observation.images.<role>_<image_key>``;
* all values written here remain absolute.  Pi05 converts arm joints to
  anchor-relative actions and normalizes them later in its training pipeline;
* with ``--include-ee-pose``, recorded observation/command end-effector poses
  are retained both as 24D position+rotation-matrix columns for Pi05 and as
  LingBot-VLA2's native 14D position+quaternion(xyzw) ``end.position`` fields.

Source implementation:
``yam-abc-reproduce/yam_abc_reproduce/data/formats/lerobot_format.py`` at
commit ``e65f7e4dba718e99913a7bd02385b30110b49540``.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

WRITE_COMPLETE_FLAG = "write_complete.flag"


def _lerobot_image_name(role: str, image_key: str) -> str:
    return f"observation.images.{role}_{image_key}"


def _nearest_indices(reference_timestamps: np.ndarray, source_timestamps: np.ndarray) -> np.ndarray:
    """Return the nearest source-frame index for every reference timestamp."""
    source_timestamps = np.asarray(source_timestamps)
    if source_timestamps.shape[0] <= 1:
        return np.zeros(len(reference_timestamps), dtype=int)
    indices = np.clip(
        np.searchsorted(source_timestamps, reference_timestamps),
        1,
        source_timestamps.shape[0] - 1,
    )
    prefer_left = (
        reference_timestamps - source_timestamps[indices - 1]
        <= source_timestamps[indices] - reference_timestamps
    )
    return indices - prefer_left.astype(int)


def _read_video(path: Path) -> list[np.ndarray]:
    """Decode an MP4 into HWC RGB uint8 frames."""
    import av

    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def _load_metadata(episode_dir: Path) -> dict[str, Any]:
    with (episode_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    if "arm_names" not in metadata and "arm_name" in metadata:
        metadata["arm_names"] = [metadata["arm_name"]]
    return metadata


def _load_episode(episode_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    buffers: dict[str, Any] = {}
    for path in sorted(episode_dir.glob("*.npy")):
        buffers[path.stem] = np.load(path)
    for camera in metadata.get("cameras", []):
        role = camera["role"]
        for image_key in camera.get("image_keys", []):
            key = f"{role}-images-{image_key}"
            buffers[key] = _read_video(episode_dir / f"{key}.mp4")
    return buffers


def _ee_pose_names(arms: list[str]) -> list[str]:
    return [
        name
        for arm in arms
        for name in (
            f"{arm}_ee_x",
            f"{arm}_ee_y",
            f"{arm}_ee_z",
            *[f"{arm}_ee_rotm_{row}{column}" for row in range(3) for column in range(3)],
        )
    ]


def _pack_ee_pose(buffers: dict[str, Any], arms: list[str], *, action: bool) -> np.ndarray:
    prefix = "action-" if action else ""
    return np.concatenate(
        [
            part
            for arm in arms
            for part in (
                buffers[f"{prefix}{arm}-ee_pos"],
                buffers[f"{prefix}{arm}-ee_rotm"].reshape(-1, 9),
            )
        ],
        axis=1,
    ).astype(np.float32)


def _end_pose_names(arms: list[str]) -> list[str]:
    return [
        name
        for arm in arms
        for name in (
            f"{arm}_ee_x",
            f"{arm}_ee_y",
            f"{arm}_ee_z",
            f"{arm}_ee_qx",
            f"{arm}_ee_qy",
            f"{arm}_ee_qz",
            f"{arm}_ee_qw",
        )
    ]


def _pack_end_pose(buffers: dict[str, Any], arms: list[str], *, action: bool) -> np.ndarray:
    """Pack absolute EE poses as per-arm xyz+quaternion(xyzw)."""
    from scipy.spatial.transform import Rotation

    prefix = "action-" if action else ""
    parts: list[np.ndarray] = []
    for arm in arms:
        position = np.asarray(buffers[f"{prefix}{arm}-ee_pos"], dtype=np.float32)
        rotation = np.asarray(buffers[f"{prefix}{arm}-ee_rotm"]).reshape(-1, 3, 3)
        quaternion_xyzw = Rotation.from_matrix(rotation).as_quat().astype(np.float32)
        parts.append(np.concatenate([position, quaternion_xyzw], axis=1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def _build_features(metadata: dict[str, Any], *, include_ee_pose: bool = False) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for camera in metadata.get("cameras", []):
        for image_key in camera.get("image_keys", []):
            features[_lerobot_image_name(camera["role"], image_key)] = {
                "dtype": "video",
                "shape": (camera["height"], camera["width"], 3),
                "names": ["height", "width", "channels"],
            }

    arms = metadata.get("arm_names") or ["left"]
    num_joints = int(metadata["num_arm_joints"])
    names = [
        name
        for arm in arms
        for name in (
            *[f"{arm}_joint_{index}" for index in range(num_joints)],
            f"{arm}_gripper",
        )
    ]
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(names),),
        "names": names,
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (len(names),),
        "names": names,
    }
    if include_ee_pose:
        ee_pose_names = _ee_pose_names(arms)
        end_pose_names = _end_pose_names(arms)
        features["observation.ee_pose"] = {
            "dtype": "float32",
            "shape": (len(ee_pose_names),),
            "names": ee_pose_names,
        }
        features["action.ee_pose"] = {
            "dtype": "float32",
            "shape": (len(ee_pose_names),),
            "names": ee_pose_names,
        }
        features["observation.state.end.position"] = {
            "dtype": "float32",
            "shape": (len(end_pose_names),),
            "names": end_pose_names,
        }
        features["action.end.position"] = {
            "dtype": "float32",
            "shape": (len(end_pose_names),),
            "names": end_pose_names,
        }
    return features


def _episode_dirs(source: Path) -> list[Path]:
    if (source / WRITE_COMPLETE_FLAG).exists():
        return [source]
    return sorted(path for path in source.iterdir() if (path / WRITE_COMPLETE_FLAG).exists())


def _add_episode(dataset: Any, episode_dir: Path, *, include_ee_pose: bool = False) -> None:
    metadata = _load_metadata(episode_dir)
    buffers = _load_episode(episode_dir, metadata)
    arms = metadata.get("arm_names") or ["left"]

    state = np.concatenate(
        [
            part
            for arm in arms
            for part in (buffers[f"{arm}-joint_pos"], buffers[f"{arm}-gripper_pos"])
        ],
        axis=1,
    ).astype(np.float32)
    action = np.concatenate(
        [
            part
            for arm in arms
            for part in (buffers[f"action-{arm}-joint"], buffers[f"action-{arm}-gripper"])
        ],
        axis=1,
    ).astype(np.float32)
    if include_ee_pose:
        state_ee_pose = _pack_ee_pose(buffers, arms, action=False)
        action_ee_pose = _pack_ee_pose(buffers, arms, action=True)
        state_end_pose = _pack_end_pose(buffers, arms, action=False)
        action_end_pose = _pack_end_pose(buffers, arms, action=True)

    cameras = metadata.get("cameras", [])
    reference_role = cameras[0]["role"] if cameras else None
    reference_timestamps = buffers[f"{reference_role}-timestamp"] if reference_role else None
    num_frames = len(reference_timestamps) if reference_timestamps is not None else len(state)

    camera_indices: dict[str, np.ndarray] = {}
    for camera in cameras:
        role = camera["role"]
        timestamps = buffers[f"{role}-timestamp"]
        if reference_timestamps is not None and len(timestamps) != num_frames:
            camera_indices[role] = _nearest_indices(reference_timestamps, timestamps)
        else:
            camera_indices[role] = np.arange(num_frames)

    for frame_index in range(num_frames):
        frame: dict[str, Any] = {
            "observation.state": state[frame_index],
            "action": action[frame_index],
            "task": metadata.get("task_name") or "task",
        }
        if include_ee_pose:
            frame["observation.ee_pose"] = state_ee_pose[frame_index]
            frame["action.ee_pose"] = action_ee_pose[frame_index]
            frame["observation.state.end.position"] = state_end_pose[frame_index]
            frame["action.end.position"] = action_end_pose[frame_index]
        for camera in cameras:
            role = camera["role"]
            camera_frame_index = int(camera_indices[role][frame_index])
            for image_key in camera.get("image_keys", []):
                frame[_lerobot_image_name(role, image_key)] = buffers[f"{role}-images-{image_key}"][
                    camera_frame_index
                ]
        dataset.add_frame(frame)
    dataset.save_episode()


def convert(
    source: Path,
    repo_id: str,
    output_root: Path | None,
    *,
    include_ee_pose: bool = False,
    video_codec: str = "libsvtav1",
    streaming_encoding: bool = False,
    encoder_threads: int | None = None,
) -> None:
    """Convert one completed episode or a directory of completed episodes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = _episode_dirs(source)
    if not episodes:
        raise FileNotFoundError(f"No completed YAM episodes found under {source}")

    first_metadata = _load_metadata(episodes[0])
    # Offline conversion can feed faster than an encoder. LeRobot's live-capture
    # default queue (30) drops frames after a short timeout; reserve an entire
    # episode per camera so offline conversion never takes that lossy path.
    encoder_queue_maxsize = 30
    if streaming_encoding:
        encoder_queue_maxsize = max(int(_load_metadata(ep)["num_frames"]) for ep in episodes) + 1
    create_options: dict[str, Any] = {
        "repo_id": repo_id,
        "fps": max(1, int(round(float(first_metadata["control_hz"])))),
        "features": _build_features(first_metadata, include_ee_pose=include_ee_pose),
        "root": output_root,
        "use_videos": True,
    }
    create_parameters = inspect.signature(LeRobotDataset.create).parameters
    accepts_extra_options = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in create_parameters.values()
    )
    if "vcodec" in create_parameters or accepts_extra_options:
        create_options.update(
            vcodec=video_codec,
            streaming_encoding=streaming_encoding,
            encoder_queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
        )
    dataset = LeRobotDataset.create(
        **create_options,
    )
    try:
        for episode_dir in episodes:
            print(f"Converting {episode_dir}", flush=True)
            _add_episode(dataset, episode_dir, include_ee_pose=include_ee_pose)
    finally:
        # LeRobot v3 buffers parquet data and episode metadata until finalized.
        dataset.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Completed YAM episode or parent directory")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo ID")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="LeRobot dataset root; omit to use LeRobot's default location",
    )
    parser.add_argument(
        "--include-ee-pose",
        action="store_true",
        help="retain absolute observation/command EE poses for joint+EE auxiliary training",
    )
    parser.add_argument("--video-codec", default="libsvtav1")
    parser.add_argument("--streaming-encoding", action="store_true")
    parser.add_argument("--encoder-threads", type=int, default=None)
    args = parser.parse_args()
    convert(
        args.source,
        args.repo_id,
        args.output_root,
        include_ee_pose=args.include_ee_pose,
        video_codec=args.video_codec,
        streaming_encoding=args.streaming_encoding,
        encoder_threads=args.encoder_threads,
    )


if __name__ == "__main__":
    main()
