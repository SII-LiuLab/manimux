"""Recorded manipulation scope on the front camera (agent workflow: SKILL.md).

Draws recorded gripper close/open events, projected into a calibrated front camera,
over a rollout's first frame or a layout photo. Use demonstrations for training
coverage or finalized rollouts for inference coverage; neither event detection nor
the hull establishes grasp success. Points are FK of measured joints in each arm's base
frame; the camera is a pinhole fitted either to FK<->pixel gripper correspondences
(calibrate) or, for teleop recordings, to the arms' silhouettes (silhouette: FK mesh
points vs. motion foreground, no clicking). Other frames are ECC-aligned to the
calibration image to absorb small camera shifts.

Datasets: ABC episodes (<split>/<episode>/states_actions.bin), teleop episodes
(<episode>/<side>-joint_pos.npy, <side>-gripper_pos.npy, top-images-rgb.mp4), or
finalized ManiMux sessions (rollout-*/data.zarr plus videos/index.json).

  python -m manimux.evaluation.manipulation_scope points \
      --dataset DIR --robot-config YAML --out points.json
  python -m manimux.evaluation.manipulation_scope calibrate --camera camera.json
  python -m manimux.evaluation.manipulation_scope silhouette \
      --dataset TELEOP_TASK_DIR --robot-config YAML --camera camera.json
  python -m manimux.evaluation.manipulation_scope render \
      --profile manipulation_scope.yaml --input A [B ...] --out out.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ARMS = ("left_arm", "right_arm")
COLORS = {
    "left_arm": (255, 140, 0),
    "right_arm": (230, 40, 200),
    "release": (30, 120, 255),
    "left_release": (30, 120, 255),
    "right_release": (40, 190, 60),
}
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def _episodes(dataset: Path):
    """Yield (name, {arm: (T, 7) joints + gripper}, table_z) per real episode.

    ABC: float64 rows of 14 state + 14 action values; sim episodes are skipped. ABC
    throws, so grasps are closings below table_z and releases openings above it.
    Teleop: every closing is a grasp and every opening a release.
    """
    for result in sorted(dataset.glob("rollout-*/result.json")):
        episode = result.parent
        _, arms = _rollout_states(episode)
        yield episode.name, arms, None
    for meta in sorted(dataset.glob("*/*/episode_metadata.json")):
        if json.loads(meta.read_text())["task_name"].startswith("sim"):
            continue
        rows = np.fromfile(meta.with_name("states_actions.bin"), np.float64).reshape(-1, 28)
        yield meta.parent.name, {"left_arm": rows[:, :7], "right_arm": rows[:, 7:14]}, 0.12
    for flag in sorted(dataset.glob("*/write_complete.flag")):  # skip episodes still recording
        e = flag.parent
        yield (
            e.name,
            {
                f"{side}_arm": np.c_[
                    np.load(e / f"{side}-joint_pos.npy"), np.load(e / f"{side}-gripper_pos.npy")
                ]
                for side in ("left", "right")
            },
            None,
        )


def _rollout_states(episode: Path):
    """Read measured states only; recorded control ticks are not video frame indices."""
    import zarr

    root = zarr.open_group(str(episode / "data.zarr"), mode="r")
    times = np.asarray(root["ticks/monotonic_ns"][:], dtype=np.int64)
    arms = {arm: np.asarray(root[f"ticks/state/{arm}"][:]) for arm in ARMS}
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError(f"{episode}: empty or non-increasing control timestamps")
    for arm, q in arms.items():
        if q.shape != (len(times), 7) or not np.isfinite(q).all():
            raise ValueError(f"{episode}: invalid measured {arm} states {q.shape}")
    return times, arms


def _nearest_indices(times, query):
    right = np.clip(np.searchsorted(times, query), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(np.abs(times[left] - query) <= np.abs(times[right] - query), left, right)


def _video_states(episode: Path, arms: dict, video: str):
    """Return video path and states aligned by monotonic capture time for rollouts."""
    if not (episode / "data.zarr").exists():
        return episode / video, arms, None
    times, _ = _rollout_states(episode)
    index = json.loads((episode / "videos/index.json").read_text())
    spec = index["cameras"][Path(video).stem]
    capture = np.asarray(spec["capture_monotonic_ns"], dtype=np.int64)
    nearest = _nearest_indices(times, capture)
    valid = (
        (capture >= times[0])
        & (capture <= times[-1])
        & (np.abs(times[nearest] - capture) <= 50_000_000)
    )
    return episode / "videos" / spec["file"], {a: q[nearest] for a, q in arms.items()}, valid


def extract_points(dataset: Path, robot_config: Path, *, gripper_open=0.6):
    """Grasp, release and start grasp-site points per arm, plus teleop paths."""
    from manimux.embodiments.robot.base import RobotModel

    dataset = dataset.resolve()
    models = RobotModel.from_config(str(robot_config)).kinematics.models
    out = {arm: {"grasp": [], "release": [], "start": [], "paths": []} for arm in ARMS}
    episodes, events = 0, []
    inference = any(dataset.glob("rollout-*/result.json"))
    for name, arms, table_z in _episodes(dataset):
        times = _rollout_states(dataset / name)[0] if inference else None
        episodes += 1
        for arm, q in arms.items():
            p = np.array([models[arm].fk(row)[:3, 3] for row in q])
            out[arm]["start"].append(p[0].tolist())
            g = q[:, 6]
            for i in np.flatnonzero((g[:-1] > gripper_open) & (g[1:] <= gripper_open)) + 1:
                if table_z is None or p[i, 2] < table_z:
                    out[arm]["grasp"].append(p[i].tolist())
                    if times is not None:
                        events.append(
                            dict(
                                episode=name,
                                arm=arm,
                                event="close",
                                tick=int(i),
                                monotonic_ns=int(times[i]),
                                xyz=p[i].tolist(),
                            )
                        )
            for i in np.flatnonzero((g[:-1] < gripper_open) & (g[1:] >= gripper_open)) + 1:
                if table_z is None or p[i, 2] >= table_z:
                    out[arm]["release"].append(p[i].tolist())
                    if times is not None:
                        events.append(
                            dict(
                                episode=name,
                                arm=arm,
                                event="open",
                                tick=int(i),
                                monotonic_ns=int(times[i]),
                                xyz=p[i].tolist(),
                            )
                        )
            if table_z is None:
                out[arm]["paths"].append(p[::3].round(4).tolist())
    return {
        "source": str(dataset),
        "source_kind": "inference_measured_state" if inference else "demonstrations",
        "episodes": episodes,
        "frame": "arm base",
        "gripper_threshold": gripper_open,
        "events": events,
        **out,
    }


def _project(params, points, arm):
    """Pinhole projection in the left base frame; the right base sits at (dx, dy), yawed."""
    p = np.asarray(points, float).reshape(-1, 3).copy()
    if arm == "right_arm":
        c, s = np.cos(params.get("right_base_yaw", 0.0)), np.sin(params.get("right_base_yaw", 0.0))
        p[:, :2] = p[:, :2] @ np.array([[c, -s], [s, c]]).T + params["right_base_offset_xy"]
    f, (cx, cy) = params["f"], params["principal_point"]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    q, _ = cv2.projectPoints(p, np.array(params["rvec"]), np.array(params["tvec"]), K, None)
    return q[:, 0]


def calibrate(camera: dict) -> dict:
    """Fit f, pose and right-base offset to camera['correspondences'] (Huber loss)."""
    from scipy.optimize import least_squares

    corr = camera["correspondences"]
    xyz = np.array([c["xyz"] for c in corr], float)
    px = np.array([c["px"] for c in corr], float)
    right = np.array([c["arm"] == "right_arm" for c in corr])
    cx, cy = camera["principal_point"]

    def unpack(th):
        return {
            "f": th[6],
            "principal_point": [cx, cy],
            "rvec": th[:3],
            "tvec": th[3:6],
            "right_base_offset_xy": th[7:9],
        }

    def residual(th):
        p = unpack(th)
        q = np.empty_like(px)
        q[~right] = _project(p, xyz[~right], "left_arm")
        q[right] = _project(p, xyz[right], "right_arm")
        return (q - px).ravel()

    best = None
    for f0 in (350, 420, 500, 600):
        for dy0 in (-0.5, -0.6, -0.7, -0.8):
            th = np.zeros(9)
            th[6], th[8] = f0, dy0
            world = xyz.copy()
            world[right, 1] += dy0
            K = np.array([[f0, 0, cx], [0, f0, cy], [0, 0, 1.0]])
            _, rv, tv = cv2.solvePnP(world, px, K, None, flags=cv2.SOLVEPNP_SQPNP)
            th[:3], th[3:6] = rv.ravel(), tv.ravel()
            fit = least_squares(residual, th, loss="huber", f_scale=8.0)
            err = float(np.median(np.linalg.norm(fit.fun.reshape(-1, 2), axis=1)))
            if best is None or err < best[1]:
                best = (fit.x, err)
    params = {
        k: np.asarray(v).tolist() if not np.isscalar(v) else float(v)
        for k, v in unpack(best[0]).items()
    }
    return {**camera, **params, "median_error_px": best[1]}


def _look_at(eye, target):
    """OpenCV extrinsics for a camera at `eye` looking at `target` (z up, no roll)."""
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(forward, right), forward])
    return cv2.Rodrigues(rotation)[0].ravel(), -rotation @ eye


def calibrate_silhouette(
    dataset: Path, robot_config: Path, *, video="top-images-rgb.mp4", max_episodes=9
):
    """Fit the camera to teleop video: FK mesh points of both arms vs. motion foreground.

    Foreground is each frame minus its episode's median frame (the arms move, the
    scene does not). The score is a two-sided chamfer distance in pixels between
    projected mesh points and foreground pixels; a coarse look-at grid behind the
    bases seeds Powell over f, pose and the right base's offset and yaw.
    """
    import mujoco
    from scipy.optimize import minimize

    from manimux.embodiments.robot.base import RobotModel

    solvers = {
        a: m.solver for a, m in RobotModel.from_config(str(robot_config)).kinematics.models.items()
    }
    model, rng = solvers["left_arm"].model, np.random.default_rng(0)
    geoms = []
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        if body in ("world", "base") or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue  # the bases never move, so they are not in the foreground
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[start : start + count]
        geoms.append((g, verts[rng.choice(count, min(count, 250), replace=False)]))

    def mesh_points(arm, configuration):
        solver = solvers[arm]
        solver.data.qpos[:] = solver.robot_state_to_qpos(configuration[:6], configuration[6])
        mujoco.mj_forward(solver.model, solver.data)
        d = solver.data
        return np.vstack([d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T for g, v in geoms])

    frames, reference = [], None
    episodes = list(_episodes(dataset))  # one camera pose per dataset; 9 episodes suffice
    pick = np.linspace(0, len(episodes) - 1, min(max_episodes, len(episodes))).round().astype(int)
    for name, arms, _ in (episodes[i] for i in pick):
        path, arms, valid = _video_states(dataset / name, arms, video)
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 16 or any(len(q) != count for q in arms.values()):
            raise ValueError(f"{path}: video and aligned state lengths differ")
        samples = list(range(10, count - 5, max(1, count // 6)))
        background_ids = set(np.linspace(0, count - 1, min(count, 120)).round().astype(int))
        wanted = background_ids | set(samples) | {0}
        images = {}
        for i in range(count):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"{path}: decode failed at frame {i}")
            if i in wanted:
                images[i] = image
        capture.release()
        reference = reference if reference is not None else images[0]
        background = np.median([images[i] for i in sorted(background_ids)], 0).astype(np.int16)
        for i in samples:
            if valid is not None and not valid[i]:
                continue
            fg = (np.abs(images[i].astype(np.int16) - background).max(2) > 35).astype(np.uint8)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            ys, xs = np.nonzero(fg)
            if not len(xs):
                continue
            pick = rng.choice(len(xs), min(len(xs), 1500), replace=False)
            frames.append(
                {
                    "dt": cv2.distanceTransform(1 - fg, cv2.DIST_L2, 3),
                    "fg": np.c_[xs[pick], ys[pick]],
                    "points": {a: mesh_points(a, arms[a][i]) for a in ARMS},
                }
            )
    if not frames:
        raise ValueError("No timestamp-aligned moving frames for camera calibration")
    h, w = reference.shape[:2]

    def unpack(th):
        return {
            "f": th[6],
            "principal_point": [w / 2, h / 2],
            "rvec": th[:3],
            "tvec": th[3:6],
            "right_base_offset_xy": th[7:9],
            "right_base_yaw": th[9],
        }

    def chamfer(th, subset):
        params, total = unpack(th), 0.0
        for f in subset:
            q = np.vstack([_project(params, f["points"][a], a) for a in ARMS])
            inside = (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < h - 1)
            qi = q[inside].astype(int)
            to_fg = np.full(len(q), 60.0)
            to_fg[inside] = f["dt"][qi[:, 1], qi[:, 0]]
            drawn = np.zeros((h, w), np.uint8)
            drawn[qi[:, 1], qi[:, 0]] = 1
            dt = cv2.distanceTransform(
                1 - cv2.dilate(drawn, np.ones((5, 5), np.uint8)), cv2.DIST_L2, 3
            )
            to_arm = dt[f["fg"][:, 1], f["fg"][:, 0]]
            total += np.minimum(to_fg, 60).mean() + np.minimum(to_arm, 60).mean()
        return total / len(subset)

    coarse, seeds = frames[::3], []
    for f0 in (380, 450, 600):
        for dy in (-0.5, -0.6, -0.7):
            for back in (0.1, 0.3, 0.5):
                for height in (0.4, 0.6, 0.8):
                    for ahead in (0.2, 0.4):
                        rvec, tvec = _look_at(
                            np.array([-back, dy / 2, height]), np.array([ahead, dy / 2, 0.0])
                        )
                        th = np.r_[rvec, tvec, f0, 0.0, dy, 0.0]
                        seeds.append((chamfer(th, coarse), th))
    seeds.sort(key=lambda z: z[0])
    fits = [
        minimize(
            chamfer,
            th,
            args=(frames[::2],),
            method="Powell",
            options={"maxfev": 1500, "xtol": 1e-4, "ftol": 1e-4},
        )
        for _, th in seeds[:3]
    ]
    best = min(fits, key=lambda r: r.fun)
    best = minimize(
        chamfer,
        best.x,
        args=(frames,),
        method="Powell",
        options={"maxfev": 1500, "xtol": 1e-4, "ftol": 1e-4},
    )
    params = {
        k: np.asarray(v).tolist() if not np.isscalar(v) else float(v)
        for k, v in unpack(best.x).items()
    }
    return {
        **params,
        "method": "silhouette",
        "chamfer_px": float(best.fun),
        "frames": len(frames),
    }, reference


def _core(points, frac=0.8):
    """Points inside the central `frac` by Mahalanobis distance in table XY."""
    p = np.asarray(points, float).reshape(-1, 3)
    if len(p) < 4:
        return p
    xy = p[:, :2] - np.median(p[:, :2], 0)
    d = np.einsum("ij,jk,ik->i", xy, np.linalg.pinv(np.cov(xy.T)), xy)
    return p[d <= np.quantile(d, frac)]


def _align(reference, image, mask_rows):
    """Affine warp mapping reference pixels to image pixels (ECC on static rows).

    Only absorbs small camera shifts. Measured on the YAM station with rows 300-480:
    same mount gives ECC >= 0.84 and affine deviation <= 0.04; a moved camera gives
    ECC <= 0.74 and deviation >= 0.15. Beyond that the projection is wrong, so refuse.
    """
    if not mask_rows:  # frames from the calibration session itself
        return np.eye(2, 3, dtype=np.float32)
    a = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0)
    mask = np.zeros(a.shape, np.uint8)
    for lo, hi in mask_rows:
        mask[lo:hi] = 255
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 300, 1e-6)
    cc, warp = cv2.findTransformECC(
        a, b, np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE, criteria, mask, 5
    )
    deviation = float(np.abs(warp[:, :2] - np.eye(2)).max())
    if cc < 0.75 or deviation > 0.1:
        raise ValueError(
            f"camera differs from the calibration image (ECC {cc:.2f}, affine deviation "
            f"{deviation:.2f}); recalibrate front_camera.json for this camera pose"
        )
    return warp


def load_profile(path: Path) -> dict:
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    if profile.get("schema") != "manipulation-scope-v1":
        raise ValueError("manipulation scope profile requires schema: manipulation-scope-v1")
    base = path.parent
    camera = json.loads((base / profile["camera"]).read_text())
    return {
        **profile,
        "source_files": {
            "camera": str((base / profile["camera"]).resolve()),
            "points": str((base / profile["points"]).resolve()),
            "reference": str((base / camera["reference_image"]).resolve()),
        },
        "camera": camera,
        "points": json.loads((base / profile["points"]).read_text()),
        "reference": np.asarray(Image.open(base / camera["reference_image"]).convert("RGB")),
    }


def first_frame(episode: Path, camera_name: str) -> np.ndarray:
    capture = cv2.VideoCapture(str(episode / "videos" / f"{camera_name}.mp4"))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"cannot read {camera_name} video in {episode}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render(profile: dict, image: np.ndarray, title: str = "") -> Image.Image:
    camera, points, labels = profile["camera"], profile["points"], profile["labels"]
    warp = _align(profile["reference"], image, profile["align_mask_rows"])
    offset = np.asarray(profile.get("points_offset_m", [0.0, 0.0, 0.0]))

    def to_px(p, arm):
        q = _project(camera, np.asarray(p, float) + offset, arm)
        return np.c_[q, np.ones(len(q))] @ warp.T

    def hull(q):
        return [tuple(p) for p in cv2.convexHull(q.astype(np.float32))[:, 0]]

    def draw_hull(q, color, alpha):
        if len(q) >= 3:
            draw.polygon(hull(q), fill=color + (alpha,), outline=color + (255,), width=3)

    im = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", im.size)
    draw = ImageDraw.Draw(overlay)
    for arm in ARMS:  # demonstration paths of the grasp site (teleop datasets)
        for path in points[arm].get("paths", []):
            if len(path) >= 2:
                draw.line([tuple(q) for q in to_px(path, arm)], fill=COLORS[arm] + (70,), width=1)
    legend = []
    if "release" in labels:
        # ABC throws ~45 cm up; drop releases onto the table plane to show where the bin goes.
        release = []
        for arm in ARMS:
            r = _core(points[arm]["release"])
            if not len(r):
                continue
            r[:, 2] = 0.0
            release.append(to_px(r, arm))
        color = COLORS["release"]
        if release:
            draw_hull(np.vstack(release), color, 45)
    else:  # placements: per arm, at the recorded height
        for arm in ARMS:
            release, color = (
                np.asarray(points[arm]["release"]),
                COLORS[f"{arm.split('_')[0]}_release"],
            )
            if not len(release):
                continue
            q = to_px(release, arm)
            draw_hull(to_px(_core(release), arm), color, 45)
            for x, y in q:
                draw.rectangle((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=color + (230,))
    for arm in ARMS:
        grasp, color = np.asarray(points[arm]["grasp"]), COLORS[arm]
        if not len(grasp):
            continue
        draw_hull(to_px(_core(grasp), arm), color, 55)
        for x, y in to_px(grasp, arm):
            draw.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=color + (230,))
        mx, my = to_px(np.median(grasp, 0), arm)[0]
        draw.line((mx - 7, my, mx + 7, my), fill=(255, 255, 255, 255), width=3)
        draw.line((mx, my - 7, mx, my + 7), fill=(255, 255, 255, 255), width=3)
    legend += [(COLORS[arm], labels[arm].format(n=len(points[arm]["grasp"]))) for arm in ARMS]
    if "release" in labels:
        legend.append((COLORS["release"], labels["release"]))
    else:
        legend += [
            (COLORS[key], labels[key].format(n=len(points[f"{key.split('_')[0]}_arm"]["release"])))
            for key in ("left_release", "right_release")
        ]
    for region in profile.get("regions", []):
        color = tuple(region["color"])
        poly = (
            np.c_[np.asarray(region["polygon_px"], float), np.ones(len(region["polygon_px"]))]
            @ warp.T
        )
        for p, q in zip(poly, np.roll(poly, -1, 0), strict=True):  # dashed outline
            for s in np.arange(0, 1, 1 / 7):
                draw.line(
                    (*(p + (q - p) * s), *(p + (q - p) * (s + 1 / 14))),
                    fill=color + (255,),
                    width=2,
                )
        legend.append((color, region["label"]))
    im = Image.alpha_composite(im, overlay).convert("RGB")

    font, bold = ImageFont.truetype(FONT, 14), ImageFont.truetype(FONT_BOLD, 17)
    canvas = Image.new("RGB", (im.width, im.height + 8 + 26 * len(legend)), (250, 250, 250))
    canvas.paste(im)
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, font=bold, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    y = im.height + 6
    for color, text in legend:
        draw.rectangle((8, y + 4, 22, y + 18), fill=color)
        draw.text((28, y), text, font=font, fill=(20, 20, 20))
        y += 26
    return canvas


def _load_input(value: str, camera_name: str) -> tuple[np.ndarray, str]:
    path = Path(value).expanduser()
    if path.suffix == ".mp4":
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"cannot read {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), path.parent.name
    if path.is_dir():
        return first_frame(path, camera_name), f"{path.parent.name[-24:]}/{path.name}"
    return np.asarray(Image.open(path).convert("RGB")), path.stem


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("points", help="extract demonstration or inference close/open points")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--robot-config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("calibrate", help="refit camera.json from its correspondences")
    c.add_argument("--camera", type=Path, required=True)
    si = sub.add_parser("silhouette", help="fit camera.json to teleop video, no clicking")
    si.add_argument("--dataset", type=Path, required=True)
    si.add_argument("--robot-config", type=Path, required=True)
    si.add_argument("--camera", type=Path, required=True)
    si.add_argument(
        "--video", default="top-images-rgb.mp4", help="Video name; use d405_front.mp4 for rollouts"
    )
    si.add_argument("--max-episodes", type=int, default=9)
    r = sub.add_parser("render", help="overlay on images or episode dirs, side by side")
    r.add_argument("--profile", type=Path, required=True)
    r.add_argument("--input", nargs="+", required=True, help="images, mp4 files or episode dirs")
    r.add_argument("--title", nargs="*", default=[])
    r.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "points":
        result = extract_points(args.dataset, args.robot_config)
        args.out.write_text(json.dumps(result) + "\n")
        print(
            f"{result['episodes']} episodes:",
            {arm: {k: len(v) for k, v in result[arm].items()} for arm in ARMS},
        )
    elif args.command == "calibrate":
        camera = calibrate(json.loads(args.camera.read_text()))
        args.camera.write_text(json.dumps(camera, indent=2) + "\n")
        print(
            f"f {camera['f']:.1f}, right base {np.round(camera['right_base_offset_xy'], 3)}, "
            f"median error {camera['median_error_px']:.1f} px"
        )
    elif args.command == "silhouette":
        camera, reference = calibrate_silhouette(
            args.dataset, args.robot_config, video=args.video, max_episodes=args.max_episodes
        )
        image = args.camera.with_name(args.camera.stem + "_reference.png")
        cv2.imwrite(str(image), reference)
        camera.update(reference_image=image.name, source=f"silhouette fit on {args.dataset}")
        args.camera.write_text(json.dumps(camera, indent=2) + "\n")
        print(
            f"f {camera['f']:.1f}, right base {np.round(camera['right_base_offset_xy'], 3)} "
            f"yaw {camera['right_base_yaw']:.3f}, chamfer {camera['chamfer_px']:.1f} px "
            f"over {camera['frames']} frames"
        )
    else:
        profile = load_profile(args.profile)
        panels = []
        for i, value in enumerate(args.input):
            image, name = _load_input(value, profile["camera_name"])
            panels.append(render(profile, image, args.title[i] if i < len(args.title) else name))
        out = Image.new(
            "RGB",
            (sum(p.width for p in panels) + 10 * (len(panels) - 1), max(p.height for p in panels)),
            "white",
        )
        x = 0
        for panel in panels:
            out.paste(panel, (x, 0))
            x += panel.width + 10
        out.save(args.out, quality=92)
        print(args.out)
