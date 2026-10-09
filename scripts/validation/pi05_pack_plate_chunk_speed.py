"""Collect native Pi05 pack-plate speed and image evidence without robot hardware.

Run in the existing Pi05 environment. Model transforms and inference are provided
by XPolicyLab; this probe only aligns dataset frames and records diagnostics.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Keep the separate offline process from preallocating the live server's GPU memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import av
import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "XPolicyLab"),
    str(ROOT / "XPolicyLab/policy/Pi_05/openpi/src"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_frames(dataset, row, side, indices, fps):
    key = f"observation.images.{side}_wrist"
    path = (
        dataset
        / f"videos/{key}"
        / f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}"
        / f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
    )
    if not indices:
        raise ValueError("No complete future action windows in this episode")
    start = float(row[f"videos/{key}/from_timestamp"])
    wanted = set(indices)
    found = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(0, start - 1) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            pts = float(frame.pts * stream.time_base)
            index = round((pts - start) * fps)
            if index in wanted:
                if abs(pts - start - index / fps) > 0.5 / fps:
                    raise ValueError("Video frame not aligned to dataset time")
                found[index] = (frame.to_ndarray(format="rgb24"), pts)
            if index > max(indices):
                break
    if set(found) != wanted:
        raise ValueError(f"Missing {side} video frames: {sorted(wanted - set(found))}")
    return found, {
        "path": str(path.resolve()),
        "from_timestamp": start,
        "file_size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }


def preview(rgb):
    image = Image.fromarray(rgb)
    image.thumbnail((256, 192))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", required=True, help="Comma-separated dataset episode indices")
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    for key in ("dataset", "checkpoint", "recipe", "out"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    out = args.out
    if args.stride <= 0 or args.seed < 0:
        parser.error("stride must be positive and seed nonnegative")
    selected = [int(x) for x in args.episodes.split(",")]
    if min(selected) < 0 or len(selected) != len(set(selected)):
        parser.error("episode indices must be nonnegative and unique")
    if any(out == source or source in out.parents for source in (args.dataset, args.checkpoint)):
        parser.error("Output must be outside dataset and checkpoint directories")
    if out.exists() and any(out.iterdir()):
        parser.error("Use an empty output directory to preserve completed evidence")
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.recipe.read_text())
    config["model_path"] = str(args.checkpoint.resolve())
    config["norm_stats_path"] = str(
        args.checkpoint.resolve() / "assets" / Path(config["norm_stats_path"]).name
    )
    from XPolicyLab.policy.Pi_05.model import Model
    from XPolicyLab.policy.Pi_05.pack_plate_contract import validate_artifacts

    contract = validate_artifacts(config)
    meta_path = args.dataset / "meta/info.json"
    info = json.loads(meta_path.read_text())
    fps = float(info["fps"])
    if fps <= 0 or not np.isclose(1 / fps, contract["action_dt_s"]):
        raise ValueError("Dataset fps must match the deployment action interval")
    data_paths = sorted((args.dataset / "data").glob("*/*.parquet"))
    episode_paths = sorted((args.dataset / "meta/episodes").glob("*/*.parquet"))
    data = pd.concat(
        [
            pd.read_parquet(
                p,
                columns=[
                    "episode_index",
                    "frame_index",
                    "timestamp",
                    "observation.state",
                    "action",
                ],
            )
            for p in data_paths
        ]
    )
    episodes = pd.concat([pd.read_parquet(p) for p in episode_paths])
    if not set(selected).issubset(set(episodes.episode_index)):
        raise ValueError("Selected episode is absent from dataset metadata")
    provenance = {
        "started_utc": datetime.now(UTC).isoformat(),
        "recipe": str(args.recipe.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "contract": contract,
        "checkpoint_source": config["checkpoint_source"],
        "norm_stats_sha256": config["norm_stats_sha256"],
        "dataset": str(args.dataset.resolve()),
        "dataset_split": info["splits"],
        "episodes": selected,
        "stride_frames": args.stride,
        "fps": fps,
        "horizon": 32,
        "num_steps": config["num_steps"],
        "prompt": args.prompt,
        "seed": args.seed,
        "seed_rule": "seed + episode * 100000 + frame",
        "sampler": "default; no RTC condition; one independent sample per observation",
        "preprocessing": (
            "Existing PackPlateZeroPoseModel transforms; source RGB and recorded gripper "
            "openings; zero pose"
        ),
        "output": "native right/left current-TCP-relative actions, before IK/waypoint/executor",
        "speed": (
            "Adjacent predicted positions, 31 intervals for 32 future rows; "
            "mm/s at dataset fps; anchor-to-first action excluded"
        ),
        "gt": "observation.state[t+1:t+33]; verified action[:-1] equals state[1:] in each episode",
        "latency": (
            "Wall time around policy.infer plus synchronized ndarray conversion; "
            "image decode excluded; first JIT warmup excluded from steady-state latency"
        ),
        "images": (
            "Every sampled observation shown as 256x192 JPEG preview; "
            "inference uses original RGB, source frame SHA-256 retained"
        ),
        "hashes": {
            str(p.resolve()): sha(p)
            for p in [
                args.recipe,
                meta_path,
                *data_paths,
                *episode_paths,
                Path(__file__),
                Path(config["norm_stats_path"]) / "norm_stats.json",
            ]
        },
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("Loading isolated offline model", flush=True)
    start = time.perf_counter()
    model = Model(config)
    policy = model.policy
    print(f"Model loaded in {time.perf_counter() - start:.1f}s", flush=True)
    native_actions, record_order, all_episodes = [], [], []
    first = True
    for ep in provenance["episodes"]:
        row = episodes[episodes.episode_index == ep].iloc[0]
        ed = data[data.episode_index == ep].sort_values("frame_index")
        states = np.stack(ed["observation.state"]).astype(np.float64)
        actions = np.stack(ed["action"]).astype(np.float64)
        if states.shape != (len(ed), 20) or actions.shape != states.shape:
            raise ValueError("Expected left-then-right [xyz, rotation6d, opening] state/action")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Non-finite dataset state or action")
        if not np.array_equal(ed["frame_index"].to_numpy(), np.arange(len(ed))):
            raise ValueError("Noncontiguous episode frame indices")
        if not np.allclose(np.diff(ed["timestamp"]), 1 / fps, rtol=0, atol=1e-4):
            raise ValueError("Dataset timestamps do not match metadata fps")
        if not np.allclose(actions[:-1], states[1:], rtol=0, atol=1e-6):
            raise ValueError("Dataset action is not next state")
        indices = list(range(3, len(states) - 32, args.stride))
        left, left_src = read_frames(args.dataset, row, "left", indices, fps)
        right, right_src = read_frames(args.dataset, row, "right", indices, fps)
        result = {
            "id": ep,
            "duration": (len(states) - 1) / fps,
            "frames": len(states),
            "videos": {"left": left_src, "right": right_src},
            "chunks": [],
            "gt": {},
        }
        for side, offset in (("left_arm", 0), ("right_arm", 10)):
            result["gt"][side] = {
                "t": ((np.arange(len(states) - 1) + 0.5) / fps).tolist(),
                "speed": (
                    np.linalg.norm(np.diff(states[:, offset : offset + 3], axis=0), axis=1)
                    * 1000
                    * fps
                ).tolist(),
            }
        for t in indices:
            seed = args.seed + ep * 100000 + t
            policy.reset_rng(seed)
            observation = {
                "state": np.r_[states[t, 10:20], states[t, 0:10]].astype(np.float32),
                "images": {"left_wrist": left[t][0], "right_wrist": right[t][0]},
                "prompt": args.prompt,
            }
            start = time.perf_counter()
            prediction = policy.infer(observation, num_steps=config["num_steps"])
            pred = np.asarray(prediction["actions"], dtype=np.float64)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if pred.shape != (32, 20) or not np.isfinite(pred).all():
                raise ValueError("Invalid native prediction")
            native_actions.append(pred)
            record_order.append([ep, t, seed])
            rec = {
                "frame": t,
                "t": t / fps,
                "seed": seed,
                "latency_ms": elapsed_ms,
                "warmup": first,
                "arms": {},
                "images": {},
                "image_source": {},
            }
            first = False
            for side, src_offset, pred_offset in (("left_arm", 0, 10), ("right_arm", 10, 0)):
                p = pred[:, pred_offset : pred_offset + 3]
                gt = states[t + 1 : t + 33, src_offset : src_offset + 3]
                rec["arms"][side] = {
                    "pred": (np.linalg.norm(np.diff(p, axis=0), axis=1) * 1000 * fps).tolist(),
                    "gt": (np.linalg.norm(np.diff(gt, axis=0), axis=1) * 1000 * fps).tolist(),
                    "speed_now": float(
                        np.linalg.norm(
                            states[t, src_offset : src_offset + 3]
                            - states[t - 3, src_offset : src_offset + 3]
                        )
                        * 1000
                        * fps
                        / 3
                    ),
                }
            for side, values in (("left", left[t]), ("right", right[t])):
                rgb, pts = values
                rec["images"][side] = preview(rgb)
                rec["image_source"][side] = {
                    "pts_seconds": pts,
                    "rgb_shape": list(rgb.shape),
                    "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                }
            result["chunks"].append(rec)
        all_episodes.append(result)
        (out / "episodes.json").write_text(json.dumps(all_episodes, separators=(",", ":")))
        np.savez_compressed(
            out / "native-actions.npz",
            actions=np.stack(native_actions),
            index=np.array(record_order),
        )
        print(
            f"episode {ep}: {len(result['chunks'])} chunks; total {len(native_actions)}", flush=True
        )
    latency = [c["latency_ms"] for e in all_episodes for c in e["chunks"] if not c["warmup"]]
    provenance.update(
        {
            "finished_utc": datetime.now(UTC).isoformat(),
            "chunks": len(native_actions),
            "latency_ms": {
                "n": len(latency),
                "median": float(np.median(latency)) if latency else None,
                "p95": float(np.percentile(latency, 95)) if latency else None,
            },
            "model_metadata": model.runtime_metadata(),
        }
    )
    provenance["hashes"][str(out / "native-actions.npz")] = sha(out / "native-actions.npz")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(
        json.dumps({"chunks": len(native_actions), "latency_ms": provenance["latency_ms"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
