#!/usr/bin/env python3
"""UMI demo source — LeRobot v3 episode -> XRFrame-compatible frame stream.

The handheld UMI rig (TacCap leader gripper + Pico4 tracker) records episodes
through xense-taccap-lerobot's ``bi_taccap_gripper`` robot class. This module
turns one such episode back into a frame stream that algos.retarget.Retargeter
consumes *unchanged*, so a recorded demo runs through the exact same
lowpass -> clutch -> Cartesian-limiter -> IK -> safety-gate chain as live
teleop. Nothing in the safety path knows the frames came from a file.

See docs/drivers.md#umi_sourcepy for the frame/units derivation.

Recorded layout (``meta/info.json`` -> ``features.action.names``):
    {side}_tcp.x/y/z        position, METERS
    {side}_tcp.r1..r3       first column of the rotation matrix
    {side}_tcp.r4..r6       second column  (6D rotation, Zhou et al.)
    {side}_gripper.pos      normalized jaw, 0 = CLOSED, 1 = OPEN

Coordinate frame: the Pico VR world frame — X forward, Y left, Z up,
gravity-aligned, origin = the headset position at the moment the Unity app
started. That origin moves on every VR restart, which is exactly why this
feeds the *relative-mapping* Retargeter rather than being commanded as
absolute poses. The replay path takes that relative motion in the gripper's
own EE frame (Retargeter's delta_frame='ee'), so the world frame's axes
never have to be mapped onto the arm's base frame at all.

`action[t]` is bit-identical to `observation.state[t+1]` in these datasets —
the rig is a passive recorder (`send_action()` is a no-op), so its "action" is
just the next measured pose. We load `action`, so frame i's target is the pose
the demonstrator actually reached.
"""
import glob
import json
import math
import os

import numpy as np

HANDS = ('left', 'right')

# Per-hand slice of the 20-D action vector; resolved from the recorded
# feature names rather than hardcoded, so a layout change fails loudly.
_POSE_KEYS = ('x', 'y', 'z', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6')


# ---------- math ----------

def rot6d_to_mat(r6):
    """6D rotation representation -> 3x3 rotation matrix.

    r6[0:3] and r6[3:6] are the first two *columns* of R (that is the
    convention bi_taccap_gripper documents and emits). Gram-Schmidt makes the
    result a proper rotation even if the stored columns drifted; recorded data
    is already orthonormal to ~4e-8, so this is a guard, not a fix.
    """
    a1 = np.asarray(r6[0:3], dtype=float)
    a2 = np.asarray(r6[3:6], dtype=float)
    n1 = np.linalg.norm(a1)
    if n1 < 1e-9:
        raise ValueError('6D 旋转的第一列接近零向量，数据坏了')
    b1 = a1 / n1
    a2 = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2)
    if n2 < 1e-9:
        raise ValueError('6D 旋转的两列共线，数据坏了')
    b2 = a2 / n2
    return np.column_stack((b1, b2, np.cross(b1, b2)))


def mat_to_quat(R):
    """3x3 rotation -> quaternion, xyzw order (config.XR_QUAT_ORDER).

    Inverse of retarget.quat_to_mat; verified round-trip in self_test().
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (x, y, z, w)


def quat_slerp(q0, q1, t):
    """Spherical interpolation between two xyzw quaternions.

    Hemisphere-corrected: q and -q are the same rotation, so without the sign
    flip an interpolation can take the 350-degree way around.
    """
    a = np.asarray(q0, dtype=float)
    b = np.asarray(q1, dtype=float)
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:                      # nearly parallel: lerp + renormalize
        r = a + (b - a) * t
        return r / np.linalg.norm(r)
    th0 = math.acos(max(-1.0, min(1.0, d)))
    th = th0 * t
    b2 = b - a * d
    b2 /= np.linalg.norm(b2)
    return a * math.cos(th) + b2 * math.sin(th)


# ---------- frame ----------

class UmiFrame:
    """One recorded bimanual sample, quacking like drivers.xr_source.XRFrame.

    Retargeter only ever calls .pose(hand) and .button(hand, name, default),
    so implementing those two is enough to reuse it verbatim. The clutch is
    reported permanently pressed: a recorded episode has no clutch, the
    operator's intent for every frame in it was "follow me".

    Consequence worth knowing: Retargeter's fault latch only clears when the
    clutch is *released*, so a recording can never clear one. After any
    sustained fault the retargeter stays disengaged for the rest of the
    episode and every later frame reports 'clutch_released'. A runner must
    therefore treat rt.fault_count > 0 as fatal and stop, rather than let the
    run coast to the end looking healthy -- scripts/umi_replay.py does.
    """

    __slots__ = ('host_ns', 'poses', 'grips')

    def __init__(self, host_ns, poses, grips):
        self.host_ns = host_ns
        self.poses = poses      # {'left': (x,y,z,qx,qy,qz,qw), 'right': ...}
        self.grips = grips      # {'left': 0..1, 'right': 0..1}, 0 = CLOSED

    def pose(self, hand):
        """hand: 'left'|'right' -> 7-tuple in METERS + xyzw quat, or None."""
        return self.poses.get(hand)

    def button(self, hand, name, default=0.0):
        if name == 'grip':          # clutch: always engaged, see class docstring
            return 1.0 if hand in self.poses else 0.0
        if name == 'trigger':       # gripper axis, if anything asks via the XR API
            return self.grips.get(hand, default)
        return default              # axisX etc: no joystick in a recording

    def gripper(self, hand, default=None):
        """Normalized jaw, 0 = CLOSED, 1 = OPEN — same convention as the
        TacCap follower's set_target(), so this passes straight through."""
        return self.grips.get(hand, default)


# ---------- dataset loading ----------

def _feature_index(info, key='action'):
    """{'left': {'x': 0, ...}, 'right': {...}} from the recorded feature names."""
    feat = (info.get('features') or {}).get(key)
    if not feat or not feat.get('names'):
        raise ValueError('meta/info.json 里没有 features.%s.names' % key)
    names = list(feat['names'])
    idx = {}
    for hand in HANDS:
        cols = {}
        for k in _POSE_KEYS:
            n = '%s_tcp.%s' % (hand, k)
            if n in names:
                cols[k] = names.index(n)
        g = '%s_gripper.pos' % hand
        if len(cols) == len(_POSE_KEYS) and g in names:
            cols['grip'] = names.index(g)
            idx[hand] = cols
    if not idx:
        raise ValueError('features.%s.names 里没有认识的 {left,right}_tcp.* 字段：\n  %s'
                         % (key, names))
    return idx


def load_episode(root, episode=0, key='action'):
    """Load one episode of a LeRobot v3 dataset -> (frames, meta).

    :param root:    dataset directory (the one containing meta/ and data/)
    :param episode: episode_index to extract
    :param key:     'action' (default) or 'observation.state'
    :return: (list[UmiFrame] at the recorded fps, info dict)
    """
    import pyarrow.parquet as pq            # heavy; only needed for this path

    info_path = os.path.join(root, 'meta', 'info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError('%s 不存在 —— 这个目录不像 LeRobot 数据集根目录'
                                % info_path)
    with open(info_path) as fh:
        info = json.load(fh)
    idx = _feature_index(info, key)

    files = sorted(glob.glob(os.path.join(root, 'data', 'chunk-*', 'file-*.parquet')))
    if not files:
        raise FileNotFoundError('%s/data/chunk-*/file-*.parquet 一个都没找到' % root)

    rows = []
    for path in files:
        # Small datasets only; a multi-GB one would want a pushdown filter on
        # episode_index instead of reading every file whole.
        tbl = pq.read_table(path, columns=[key, 'episode_index', 'frame_index',
                                           'timestamp'])
        d = tbl.to_pydict()
        for vec, ep, fi, ts in zip(d[key], d['episode_index'],
                                   d['frame_index'], d['timestamp']):
            if int(ep) == int(episode):
                rows.append((int(fi), float(ts), np.asarray(vec, dtype=float)))
    if not rows:
        raise ValueError('数据集里没有 episode_index=%d（共 %s 条 episode）'
                         % (episode, info.get('total_episodes', '?')))
    rows.sort(key=lambda r: r[0])

    frames = []
    for _, ts, vec in rows:
        poses, grips = {}, {}
        for hand, cols in idx.items():
            p = [vec[cols[k]] for k in ('x', 'y', 'z')]
            R = rot6d_to_mat([vec[cols[k]] for k in
                              ('r1', 'r2', 'r3', 'r4', 'r5', 'r6')])
            poses[hand] = tuple(p) + mat_to_quat(R)
            grips[hand] = float(vec[cols['grip']])
        frames.append(UmiFrame(int(ts * 1e9), poses, grips))
    return frames, info


def slice_frames(frames, spec):
    """`A:B` (half-open, in RECORDED frame indices) -> that slice.

    Recorded indices, not control periods: they are what a LeRobot dataset is
    indexed by and what scripts/umi_filter.py emits after cutting the spans
    this robot cannot reproduce. Called before resample(), so the slice keeps
    its own timeline and the replay of a span is bit-identical to that span
    inside the full episode.
    """
    try:
        a, _, b = str(spec).partition(':')
        lo = int(a) if a.strip() else 0
        hi = int(b) if b.strip() else len(frames)
    except ValueError:
        raise ValueError('--frames 格式是 A:B（录制帧下标），给的是 %r' % spec)
    return frames[max(0, lo):hi]


def resample(frames, hz):
    """Interpolate a recorded stream up to the control rate.

    Deliberately NOT the zero-order hold replay.py uses for XR recordings:
    those come in at 90Hz, these at 30Hz, and a ZOH at 30->250Hz hands the
    Cartesian rate limiter a 15mm step every 33ms. At the demo's peak speed
    that step needs ~8 of the 8.3 control periods per recorded frame just to
    catch up, so the limiter would sit permanently saturated and every
    reported cart_lag would be an artifact of the resampler rather than of
    the trajectory. Lerp on position, slerp on orientation, hold on the jaw
    (it is a normalized position, and interpolating it buys nothing the
    follower's own impedance loop doesn't already do).
    """
    if len(frames) < 2:
        return list(frames)
    t0, t1 = frames[0].host_ns, frames[-1].host_ns
    period_ns = 1e9 / hz
    n = int((t1 - t0) / period_ns) + 1
    out, j = [], 0
    for k in range(n):
        t = t0 + k * period_ns
        while j + 1 < len(frames) - 1 and frames[j + 1].host_ns <= t:
            j += 1
        a, b = frames[j], frames[j + 1]
        span = b.host_ns - a.host_ns
        u = 0.0 if span <= 0 else max(0.0, min(1.0, (t - a.host_ns) / span))
        poses, grips = {}, {}
        for hand in a.poses:
            if hand not in b.poses:
                continue
            pa, pb = a.poses[hand], b.poses[hand]
            p = [pa[i] + (pb[i] - pa[i]) * u for i in range(3)]
            q = quat_slerp(pa[3:], pb[3:], u)
            poses[hand] = tuple(p) + tuple(float(v) for v in q)
            grips[hand] = a.grips[hand]
        out.append(UmiFrame(int(t), poses, grips))
    return out


# ---------- self-test ----------

def self_test():
    """Round-trip 6D -> R -> quat -> R against retarget.quat_to_mat."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from algos.retarget import quat_to_mat

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        r6 = list(Q[:, 0]) + list(Q[:, 1])
        R = rot6d_to_mat(r6)
        back = quat_to_mat(mat_to_quat(R), order='xyzw')
        worst = max(worst, float(np.abs(back - R).max()))
    ok = worst < 1e-9
    print('6D -> R -> quat -> R 最大偏差 %.2e  %s' % (worst, '✅' if ok else '❌'))

    # slerp endpoints and midpoint sanity
    q0 = mat_to_quat(np.eye(3))
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # +90 deg
    q1 = mat_to_quat(Rz)
    mid = quat_to_mat(quat_slerp(q0, q1, 0.5), order='xyzw')
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(mid) - 1.0) / 2.0))))
    ok2 = abs(ang - 45.0) < 1e-6
    print('slerp 中点 %.6f°（应为 45）  %s' % (ang, '✅' if ok2 else '❌'))
    return ok and ok2


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(
        description='Inspect a UMI/LeRobot episode, or self-test the math.')
    ap.add_argument('root', nargs='?', help='dataset directory (contains meta/, data/)')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--key', default='action',
                    choices=['action', 'observation.state'])
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()

    if a.self_test or not a.root:
        raise SystemExit(0 if self_test() else 1)

    fr, info = load_episode(a.root, a.episode, a.key)
    fps = float(info.get('fps') or 30.0)
    print('%s  episode %d  key=%s' % (a.root, a.episode, a.key))
    print('  robot_type %s  codebase %s  fps %.0f'
          % (info.get('robot_type'), info.get('codebase_version'), fps))
    print('  %d 帧 = %.2f 秒' % (len(fr), (fr[-1].host_ns - fr[0].host_ns) / 1e9))
    dt = 1.0 / fps
    for hand in HANDS:
        if fr[0].pose(hand) is None:
            continue
        P = np.array([f.pose(hand)[:3] for f in fr])
        g = np.array([f.gripper(hand) for f in fr])
        v = np.linalg.norm(np.diff(P, axis=0), axis=1) / dt * 1000.0
        span = (P.max(0) - P.min(0)) * 1000.0
        print('  %-5s 位置包络 %s mm  速度 均值%.0f / p95 %.0f / 峰值%.0f mm/s'
              % (hand, np.round(span, 1), v.mean(), np.percentile(v, 95), v.max()))
        print('        夹爪 %.3f..%.3f (0=闭合)  起始 %.3f' % (g.min(), g.max(), g[0]))
