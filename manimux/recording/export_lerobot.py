"""Export one finalized absolute-joint recording through LeRobot's v3 writer."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import zarr


def _times(values, label):
    times = np.asarray(values)
    if (
        times.ndim != 1
        or not len(times)
        or times.dtype.kind != "i"
        or times[0] < 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(f"{label} needs strictly increasing integer capture timestamps")
    return times


def export_episode(episode, destination, *, group, arm_dofs, cameras, fps, max_age_ms):
    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episode, destination = Path(episode).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(episode):
        raise ValueError("choose a new destination outside the source recording")
    if not 1 <= fps <= 1000 or not np.isfinite(max_age_ms) or max_age_ms <= 0:
        raise ValueError("require 1..1000 FPS and a positive finite age limit")
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("specify each selected camera once")
    result = json.loads((episode / "result.json").read_text())
    if episode.name.endswith(".partial") or result.get("incomplete"):
        raise ValueError("source recording is incomplete")
    task = json.loads((episode / "meta.json").read_text())["task"]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("missing recorded task instruction")
    ticks = zarr.open_group(str(episode / "data.zarr"), mode="r")["ticks"]
    tick_times = _times(ticks["monotonic_ns"][:], "ticks")
    state = ticks["state"][group][:]
    command = ticks["command"][group][:]
    if (
        state.ndim != 2
        or state.shape != command.shape
        or len(state) != len(tick_times)
        or arm_dofs < 1
        or state.shape[1] - arm_dofs not in (0, 1)
    ):
        raise ValueError("expected arm joints and at most one normalized gripper")
    names = [f"{group}.joint_{i + 1}.rad" for i in range(arm_dofs)]
    if state.shape[1] > arm_dofs:
        names.append(f"{group}.gripper.open_fraction")
    features = {
        key: {"dtype": "float32", "shape": (len(names),), "names": names}
        for key in ("observation.state", "action")
    }
    videos = episode / "videos"
    index = json.loads((videos / "index.json").read_text())
    if index.get("error"):
        raise ValueError(f"source video recorder failed: {index['error']}")
    times = {"tick": tick_times}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp, ExitStack() as stack:
        streams = {}
        for camera in cameras:
            entry = index["cameras"][camera]
            path = (videos / entry["file"]).resolve()
            if path.parent != videos or not camera.isidentifier():
                raise ValueError("invalid camera file or feature name")
            key = f"observation.images.{camera}"
            times[key] = _times(entry["capture_monotonic_ns"], camera)
            container = stack.enter_context(av.open(str(path)))
            video = container.streams.video[0]
            features[key] = {
                "dtype": "video",
                "shape": (video.height, video.width, 3),
                "names": ["height", "width", "channels"],
            }
            streams[key] = iter(container.decode(video))

        start = max(int(t[0]) for t in times.values())
        end = min(int(t[-1]) for t in times.values())
        if end < start:
            raise ValueError("ticks and video have no common time range")
        count = (end - start) * fps // 1_000_000_000 + 1
        grid = start + np.arange(count, dtype=np.int64) * 1_000_000_000 // fps
        rows = {key: np.searchsorted(t, grid, side="right") - 1 for key, t in times.items()}
        age_ms = {}
        for key, indices in rows.items():
            age_ms[key] = float(np.max(grid - times[key][indices])) / 1e6
            if age_ms[key] > max_age_ms:
                raise ValueError(f"{key} exceeds sample age limit ({age_ms[key]:.3f} ms)")
        root = Path(temp) / "dataset"
        dataset = LeRobotDataset.create(
            repo_id="manimux/episode",
            root=root,
            fps=fps,
            features=features,
            robot_type=f"manimux_{group}",
            vcodec="h264",
            video_backend="pyav",
        )
        decoded = dict.fromkeys(streams, -1)
        images = {}
        for i, tick in enumerate(rows["tick"]):
            frame = {
                "task": task,
                "observation.state": state[tick].astype(np.float32),
                "action": command[tick].astype(np.float32),
            }
            if not all(np.isfinite(frame[k]).all() for k in ("observation.state", "action")):
                raise ValueError(f"non-finite state or command at tick {tick}")
            for key, stream in streams.items():
                while decoded[key] < rows[key][i]:
                    decoded_frame = next(stream, None)
                    if decoded_frame is None:
                        raise ValueError(f"{key}: video ended before its timestamp index")
                    decoded[key] += 1
                    images[key] = decoded_frame.to_ndarray(format="rgb24")
            dataset.add_frame({**frame, **images})
        for key, stream in streams.items():
            if decoded[key] + 1 + sum(1 for _ in stream) != len(times[key]):
                raise ValueError(f"{key}: video length disagrees with its timestamp index")
        dataset.save_episode()
        dataset.finalize()
        provenance = {
            "source_episode": episode.name,
            "group": group,
            "coordinates": names,
            "action_source": "ticks/command",
            "state_source": "ticks/state",
            "alignment": "latest sample at or before each output timestamp",
            "start_monotonic_ns": start,
            "max_age_ms": age_ms,
        }
        (root / "meta/manimux.json").write_text(json.dumps(provenance, indent=2) + "\n")
        root.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--arm-dofs", required=True, type=int)
    parser.add_argument("--camera", dest="cameras", action="append", required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--max-age-ms", type=float, required=True)
    print(export_episode(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
