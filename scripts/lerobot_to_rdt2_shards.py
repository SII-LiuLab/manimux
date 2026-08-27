#!/usr/bin/env python
"""Convert a LeRobot v3.0 UMI dataset into RDT2 webdataset shards.

Run with the openpi venv, which already has pyarrow/av/cv2:

    XPolicyLab/policy/Pi_05/openpi/.venv/bin/python scripts/lerobot_to_rdt2_shards.py \
        --dataset /home/jw/Desktop/dataset/exchange_ball_v0 \
        --out /home/jw/Desktop/dataset/exchange_ball_v0_rdt2

Conventions are documented in docs/rdt2-umi-runbook.md; every one of them was read out
of the upstream source rather than the README, and the file:line references are in the
docstrings below. The ones that fail silently when wrong:

* the action vector is right-arm-first while the stitched image is left-half-left-arm,
* the 6D rotation block holds the first two *columns* of R, not the rows,
* the gripper channel stays absolute while the pose channels go relative.

`--self-test` exercises the pure transforms without touching the dataset.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

# --- layout constants -------------------------------------------------------
#
# Ours (meta/info.json "names"):  [left TCP(9) | left grip(1) | right TCP(9) | right grip(1)]
# RDT2 (ckpt/RVQ/README.md:55-62): [right pos(3) rot6d(6) grip(1) | left pos(3) rot6d(6) grip(1)]
#
# The stitched image runs the other way round: left half is the LEFT wrist.
#   RDT2/configs/robots/*.yaml       "# robot configurations, 0->right 1->left"
#   RDT2/deploy/inference_real_fm.py:390-391   left_stereo <- camera1_rgb (robot1 = left)
#   RDT2/models/rdt_inferencer.py:266-268      concat order ["left_stereo", "right_stereo"]
ACTION_DIM = 20
ARM_BLOCK = 10          # pos(3) + rot6d(6) + gripper(1)
CHUNK_T = 24            # 0.8 s at 30 Hz
IMAGE_SIDE = 384
RDT2_GRIPPER_FULL_OPEN_M = 0.088   # ckpt/RVQ/README.md:59 "0.088 means fully open"
YAM_LINEAR_4310_STROKE_M = 0.096   # i2rt/robots/config/linear_4310.yml:43

LEFT_WRIST_KEY = "observation.images.left_wrist"
RIGHT_WRIST_KEY = "observation.images.right_wrist"


# --- pure transforms --------------------------------------------------------

def swap_arm_order(vec: np.ndarray) -> np.ndarray:
    """Swap the two 10-dim arm blocks. Self-inverse, so it converts either way."""
    if vec.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected last dim {ACTION_DIM}, got {vec.shape}")
    first, second = vec[..., :ARM_BLOCK], vec[..., ARM_BLOCK:]
    return np.concatenate([second, first], axis=-1)


def rot6d_to_mat(r6: np.ndarray) -> np.ndarray:
    """6D -> 3x3. r6[:3] and r6[3:] are the first two COLUMNS of R.

    Same convention on both sides: RDT2/data/umi/pose_util.py:98-105 stacks the
    Gram-Schmidt basis with axis=-1, and our capture pipeline documents the same at
    teleop/drivers/umi_source.py:46-53 ("the first two *columns* of R"). Gram-Schmidt
    is a guard, not a fix -- recorded data is orthonormal to ~4e-8.
    """
    a1 = np.asarray(r6[..., :3], dtype=np.float64)
    a2 = np.asarray(r6[..., 3:], dtype=np.float64)
    n1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    if np.any(n1 < 1e-9):
        raise ValueError("first column of the 6D rotation is ~zero; data is broken")
    b1 = a1 / n1
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    n2 = np.linalg.norm(b2, axis=-1, keepdims=True)
    if np.any(n2 < 1e-9):
        raise ValueError("6D rotation columns are collinear; data is broken")
    b2 = b2 / n2
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1)


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """3x3 -> 6D, taking the first two columns (RDT2/data/umi/pose_util.py:152-156)."""
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)


def gripper_to_metres(pos: np.ndarray, mapping: str, stroke_m: float) -> np.ndarray:
    """Our [0,1] stroke fraction -> RDT2's absolute width in metres.

    Ours is a normalised fraction, not a length: src/manimux/kinematics/yam.py:135
    ("normalized fraction of the slide joint's stroke") and the observed max of exactly
    1.00000 in meta/stats.json.

    full_open_normalized  pos=1 -> 0.088, i.e. our fully-open maps onto RDT2's
                          fully-open. Keeps every sample inside the training
                          distribution. This is a modelling choice, not a fact: our
                          jaw geometry is not theirs.
    stroke_absolute       pos * 0.096, physically true but anything above pos=0.917
                          lands outside what RDT2 ever saw.

    The `/0.088*0.10` step in the upstream README is NOT a unit conversion -- it only
    appears in deploy/inference_real_{fm,vq}.py, never under data/ or rdt/, and it
    calibrates their own 0.12 m Franka jaw. Do not apply it here.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if mapping == "full_open_normalized":
        return pos * RDT2_GRIPPER_FULL_OPEN_M
    if mapping == "stroke_absolute":
        return pos * stroke_m
    raise ValueError(f"unknown gripper mapping {mapping!r}")


def to_relative(anchor: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Express 9-dim (pos + rot6d) poses relative to a 9-dim anchor pose.

    Inverse of the upstream reconstruction T_abs = T_anchor @ T_rel
    (RDT2/data/umi/common/pose_repr_util.py:37-38), so:
        rel_pos = R_anchor^T (pos - pos_anchor)
        rel_R   = R_anchor^T R
    Zero rotation therefore comes out as the 6D identity [1,0,0,0,1,0], not as zeros.
    """
    anchor_pos = anchor[:3]
    anchor_rot = rot6d_to_mat(anchor[3:9])
    anchor_rot_t = anchor_rot.T

    pos = np.asarray(poses[..., :3], dtype=np.float64)
    rot = rot6d_to_mat(np.asarray(poses[..., 3:9], dtype=np.float64))

    rel_pos = (anchor_rot_t @ (pos - anchor_pos)[..., None])[..., 0]
    rel_rot = anchor_rot_t @ rot
    return np.concatenate([rel_pos, mat_to_rot6d(rel_rot)], axis=-1)


def build_chunk(anchor_state: np.ndarray, actions: np.ndarray, mapping: str,
                stroke_m: float) -> np.ndarray:
    """Our (T,20) absolute left-first actions -> RDT2's (T,20) relative right-first chunk.

    The anchor is the observation at the frame the image was rendered from, not the
    first row of the chunk (RDT2/data/umi_video_dataset.py:407-418 passes
    base_pose_mat=pose_mat[-1]). Row 0 is already one control step ahead, which is why
    the upstream shards show ~1 mm there instead of exact zeros.

    Gripper channels stay absolute -- umi_video_dataset.py:423 slices them straight out
    without going through convert_pose_mat_rep.
    """
    if actions.shape[-1] != ACTION_DIM or anchor_state.shape[-1] != ACTION_DIM:
        raise ValueError("expected 20-dim state and actions")

    out = np.zeros((actions.shape[0], ACTION_DIM), dtype=np.float64)
    for out_slot, src in ((0, 10), (10, 0)):        # out右臂<-src右臂, out左臂<-src左臂
        anchor = anchor_state[src:src + 9]
        out[:, out_slot:out_slot + 9] = to_relative(anchor, actions[:, src:src + 9])
        out[:, out_slot + 9] = gripper_to_metres(actions[:, src + 9], mapping, stroke_m)
    return out.astype(np.float32)


def stitch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Two wrist frames -> (384, 768, 3) uint8, LEFT wrist in the LEFT half.

    Deliberately the opposite ordering from the action vector; see the module header.
    """
    import cv2

    def one(img):
        img = np.ascontiguousarray(img)
        if img.shape[:2] != (IMAGE_SIDE, IMAGE_SIDE):
            img = cv2.resize(img, (IMAGE_SIDE, IMAGE_SIDE), interpolation=cv2.INTER_AREA)
        return img

    out = np.concatenate([one(left), one(right)], axis=1)
    assert out.shape == (IMAGE_SIDE, 2 * IMAGE_SIDE, 3), out.shape
    return out.astype(np.uint8)


# --- dataset reading --------------------------------------------------------

def read_episodes(root: Path):
    import pyarrow.parquet as pq

    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {root}/meta/episodes")
    import pyarrow as pa
    table = pa.concat_tables([pq.read_table(f) for f in files])
    cols = table.column_names
    need = ["episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index"]
    for cam in (LEFT_WRIST_KEY, RIGHT_WRIST_KEY):
        need += [f"videos/{cam}/chunk_index", f"videos/{cam}/file_index",
                 f"videos/{cam}/from_timestamp", f"videos/{cam}/to_timestamp"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise KeyError(f"episode metadata is missing {missing}")
    return table.select(need).to_pylist()


def read_frames(root: Path):
    import pyarrow.parquet as pq
    import pyarrow as pa

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no frame data under {root}/data")
    table = pa.concat_tables([pq.read_table(f) for f in files])
    state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))
    action = np.stack(table.column("action").to_numpy(zero_copy_only=False))
    return state.astype(np.float64), action.astype(np.float64)


def decode_range(path: Path, t0: float, t1: float, expected: int):
    """Yield RGB frames whose presentation time falls in [t0, t1).

    LeRobot v3.0 concatenates several episodes into one mp4, so the episode is a time
    range inside the file rather than the whole thing.
    """
    import av

    eps = 1e-6
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        seek_to = max(0.0, t0 - 1.0)
        container.seek(int(seek_to / stream.time_base), stream=stream)
        count = 0
        for frame in container.decode(stream):
            ts = float(frame.pts * stream.time_base)
            if ts < t0 - eps:
                continue
            if ts >= t1 - eps:
                break
            yield frame.to_ndarray(format="rgb24")
            count += 1
            if count >= expected:
                break


# --- conversion -------------------------------------------------------------

def convert(args) -> int:
    import cv2

    root = Path(args.dataset).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    episodes = read_episodes(root)
    state, action = read_frames(root)
    print(f"{len(episodes)} episodes / {len(state)} frames from {root}")

    instructions: dict[str, str] = {}
    key = 0
    shard_index = 0
    written = 0
    dropped = 0
    tar = None

    def open_shard():
        nonlocal tar, shard_index
        path = shard_dir / f"shard-{shard_index:06d}.tar"
        tar = tarfile.open(path, "w")
        return path

    def add(name: str, payload: bytes):
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    open_shard()
    try:
        for ep in episodes:
            idx = ep["episode_index"]
            lo, hi = ep["dataset_from_index"], ep["dataset_to_index"]
            length = hi - lo
            if length != ep["length"]:
                print(f"  ep{idx}: metadata length {ep['length']} != index span {length}",
                      file=sys.stderr)

            task = (ep["tasks"] or ["unknown"])[0]
            instr_key = f"{root.name}/episode_{idx:06d}"
            instructions[instr_key] = task
            meta_blob = json.dumps({"sub_task_instruction_key": instr_key}).encode()

            def video(cam):
                c, f = ep[f"videos/{cam}/chunk_index"], ep[f"videos/{cam}/file_index"]
                return root / "videos" / cam / f"chunk-{c:03d}" / f"file-{f:03d}.mp4"

            left = decode_range(video(LEFT_WRIST_KEY),
                                ep[f"videos/{LEFT_WRIST_KEY}/from_timestamp"],
                                ep[f"videos/{LEFT_WRIST_KEY}/to_timestamp"], length)
            right = decode_range(video(RIGHT_WRIST_KEY),
                                 ep[f"videos/{RIGHT_WRIST_KEY}/from_timestamp"],
                                 ep[f"videos/{RIGHT_WRIST_KEY}/to_timestamp"], length)

            ep_written = 0
            for i, (lf, rf) in enumerate(zip(left, right)):
                stop = lo + i + CHUNK_T
                if stop > hi:
                    if args.tail == "discard":
                        dropped += length - i
                        break
                    tail = action[lo + i:hi]
                    pad = np.repeat(action[hi - 1:hi], CHUNK_T - len(tail), axis=0)
                    window = np.concatenate([tail, pad], axis=0)
                else:
                    window = action[lo + i:stop]

                chunk = build_chunk(state[lo + i], window, args.gripper_mapping,
                                    args.gripper_stroke_m)
                ok, jpg = cv2.imencode(".jpg", stitch(lf, rf)[:, :, ::-1],
                                       [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                if not ok:
                    raise RuntimeError(f"jpeg encode failed at ep{idx} frame {i}")

                add(f"{key}.image.jpg", jpg.tobytes())
                buf = io.BytesIO()
                np.save(buf, chunk)
                add(f"{key}.action.npy", buf.getvalue())
                add(f"{key}.meta.json", meta_blob)

                key += 1
                written += 1
                ep_written += 1
                if written % args.shard_size == 0:
                    tar.close()
                    shard_index += 1
                    open_shard()

            print(f"  ep{idx}: {ep_written}/{length} samples")
            if ep_written == 0:
                print(f"  ep{idx}: decoded no frames -- check the video time range",
                      file=sys.stderr)
    finally:
        if tar is not None:
            tar.close()

    (out_dir / "instructions.json").write_text(
        json.dumps(instructions, ensure_ascii=False, indent=2))

    cfg = out_dir / "dataset.yaml"
    cfg.write_text(
        f"name: manimux/{root.name}\n"
        "type: single\n"
        f"shards_dir: {shard_dir}\n"
        "kwargs:\n"
        f"  instruction_path: {out_dir / 'instructions.json'}\n"
        "  normalizer_path: /home/jw/Desktop/project/RDT2/ckpt/RVQ/"
        "umi_normalizer_wo_downsample_indentity_rot.pt\n"
    )

    print(f"\n{written} samples in {shard_index + 1} shards -> {shard_dir}")
    if dropped:
        print(f"{dropped} tail frames discarded (--tail discard)")
    print(f"instructions.json: {len(instructions)} entries")
    print(f"dataset config: {cfg}")
    print(f"gripper mapping: {args.gripper_mapping} "
          f"(must match policy/RDT2/deploy.yml)")
    return 0


# --- self test --------------------------------------------------------------

def self_test() -> int:
    rng = np.random.default_rng(0)

    vec = np.arange(ACTION_DIM, dtype=np.float64)
    assert np.array_equal(swap_arm_order(swap_arm_order(vec)), vec)
    assert np.array_equal(swap_arm_order(vec)[:ARM_BLOCK], vec[ARM_BLOCK:])

    identity6d = np.array([1.0, 0, 0, 0, 1, 0])
    assert np.allclose(rot6d_to_mat(identity6d), np.eye(3))
    assert np.allclose(mat_to_rot6d(np.eye(3)), identity6d)

    for _ in range(200):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        assert np.allclose(rot6d_to_mat(mat_to_rot6d(q)), q, atol=1e-12)
        # first three numbers are column 0, which is what a transposed reading breaks
        assert np.allclose(mat_to_rot6d(q)[:3], q[:, 0])

    anchor = np.concatenate([rng.normal(size=3), identity6d])
    poses = np.stack([np.concatenate([rng.normal(size=3), identity6d]) for _ in range(5)])
    rel = to_relative(anchor, poses)
    back_pos = rot6d_to_mat(anchor[3:9]) @ rel[:, :3].T + anchor[:3, None]
    assert np.allclose(back_pos.T, poses[:, :3])
    assert np.allclose(to_relative(anchor, anchor[None])[0], np.concatenate([np.zeros(3), identity6d]))

    assert np.isclose(gripper_to_metres(np.array([1.0]), "full_open_normalized", 0.096)[0], 0.088)
    assert np.isclose(gripper_to_metres(np.array([1.0]), "stroke_absolute", 0.096)[0], 0.096)
    assert np.isclose(gripper_to_metres(np.array([0.0]), "full_open_normalized", 0.096)[0], 0.0)

    ours = np.zeros((CHUNK_T, ACTION_DIM))
    ours[:, 3:9] = identity6d
    ours[:, 13:19] = identity6d
    ours[:, 0] = np.linspace(0, 0.1, CHUNK_T)      # left arm moves in +x
    ours[:, 9] = 1.0                               # left gripper fully open
    anchor_state = ours[0].copy()
    chunk = build_chunk(anchor_state, ours, "full_open_normalized", 0.096)
    assert chunk.shape == (CHUNK_T, ACTION_DIM) and chunk.dtype == np.float32
    assert np.allclose(chunk[0, 10:13], 0, atol=1e-9), "row0 anchors to the observation"
    assert np.allclose(chunk[:, 13:19], identity6d), "zero rotation is the 6D identity"
    assert np.isclose(chunk[-1, 10], 0.1), "left arm motion landed in the LEFT block (d10-12)"
    assert np.isclose(chunk[0, 19], 0.088), "left gripper landed in d19, in metres"
    assert np.allclose(chunk[:, :10], np.concatenate([np.zeros(3), identity6d, [0.0]]))

    left = np.zeros((480, 640, 3), np.uint8); left[:, :, 0] = 255      # red = left wrist
    right = np.zeros((480, 640, 3), np.uint8); right[:, :, 1] = 255    # green = right wrist
    img = stitch(left, right)
    assert img.shape == (384, 768, 3) and img.dtype == np.uint8
    assert img[:, :384, 0].mean() > 250 and img[:, 384:, 1].mean() > 250, \
        "left half must hold the LEFT wrist -- opposite of the action arm order"

    print("self-test OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="LeRobot v3.0 dataset root")
    ap.add_argument("--out", help="output directory (shards/ and instructions.json go here)")
    ap.add_argument("--gripper-mapping", default="full_open_normalized",
                    choices=["full_open_normalized", "stroke_absolute"],
                    help="must match policy/RDT2/deploy.yml; mismatch fails silently")
    ap.add_argument("--gripper-stroke-m", type=float, default=YAM_LINEAR_4310_STROKE_M,
                    help="only used by stroke_absolute; linear_4310 is 0.096, crank_4310 is 0.071")
    ap.add_argument("--tail", default="pad", choices=["pad", "discard"],
                    help="episode tails shorter than 24 frames: repeat the last action, or drop")
    ap.add_argument("--shard-size", type=int, default=20000,
                    help="samples per shard (upstream uses 20000)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--self-test", action="store_true",
                    help="check the transforms and exit; needs no dataset")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.dataset or not args.out:
        ap.error("--dataset and --out are required unless --self-test is given")
    return convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
