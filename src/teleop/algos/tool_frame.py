#!/usr/bin/env python3
"""Tool frame — the constant flange → fingertip transform, so retargeting
happens at the point that actually touches the object.

See docs/algos.md#tool_framepy.

Why it exists
-------------
A recorded UMI pose is the **leader gripper's TCP: the two-finger midpoint**
(xense-taccap-lerobot's `taccap_gripper/ee_transform.py`, checked in Rerun
2026-08-02). This stack's FK returns the **flange**: `config.KINE_CFG`'s DH
chain ends at the flange face and nothing here calls `set_tool()`. Feed a
fingertip trajectory to an IK that targets the flange and the *flange* traces
the demonstrator's fingertips, while the robot's own fingertips trace a path
|t| further out.

Under pure translation that is invisible — a constant offset cancels in a
relative mapping, which is why the existing UMI replay looks fine. It shows
up under **rotation**, which is where a manipulation demo spends its
interesting frames: `replay_test` peaks at 106°/s, and a 90° wrist turn with
a 180mm tool sweeps the flange through ~250mm nobody asked for. An embodiment
filter run without this is judging the wrong trajectory (algos/embodiment.py).

Only the translation matters
----------------------------
A tool frame is (R_ft, t_ft). Replace R_ft with R_ft·A for any constant
rotation A and every commanded *flange* pose comes out bit-identical:

    tip:     R_t' = R_f·R_ft·A          p_t' = p_f + R_f·t_ft   (unchanged)
    map:     R_des' = dR·R_t0' = R_des·A            p_des'  = p_des
    back:    R_f'  = R_des'·R_ft'ᵀ = R_des·A·Aᵀ·R_ftᵀ = R_f
             p_f'  = p_des - R_f'·t_ft = p_f

The rate limiter is invariant too (it only ever uses the *angle* of
R_des·R_nowᵀ, and the A's cancel there as well). So a **pivot calibration —
which recovers t_ft and can say nothing about R_ft, the touched point being
rotationally symmetric — is sufficient** for this pipeline. Self-tested in
`__main__` against the real Retargeter, not just asserted here.

The 6-vector form is kept anyway because `configs/tool/<name>.yaml`'s
`kine_offset` is what `Marvin_Robot.set_tool()` wants, and the controller's
own Cartesian targeting does need the rotation.

Where the number comes from
---------------------------
`configs/tool/umi.yaml`'s per-arm `kine_offset`, measured with
`scripts/tip_calib.py` (pivot calibration in `release` mode). All-zero means
NOT MEASURED, and `load()` returns None for it rather than pretending a
zero-length tool is a measurement — callers warn and fall back to
flange-frame behaviour, which is exactly today's behaviour.

Self-test (no robot, no SDK):  python3 -m algos.tool_frame
"""
import math

import numpy as np

from algos.retarget import mat_to_xyzabc, xyzabc_to_rot


class ToolFrame:
    """Constant flange → tool-tip transform, in the pipeline's XYZABC form.

    :param xyzabc: [x, y, z, a, b, c] — mm and degrees, tip expressed in the
                   flange frame. Same convention and same numbers as
                   `configs/tool/<name>.yaml`'s `kine_offset`.
    """

    __slots__ = ('xyzabc', 't', 'R', 'label')

    def __init__(self, xyzabc, label=''):
        v = [float(x) for x in xyzabc]
        if len(v) == 3:
            v = v + [0.0, 0.0, 0.0]
        if len(v) != 6:
            raise ValueError('tool offset must be [x,y,z] or [x,y,z,a,b,c], '
                             'got %d values' % len(v))
        self.xyzabc = v
        self.t = np.array(v[:3], dtype=float)
        self.R = xyzabc_to_rot(v)
        self.label = label

    def __repr__(self):
        return ('<ToolFrame %s t=[%.1f %.1f %.1f]mm |t|=%.1fmm rpy=[%.1f %.1f '
                '%.1f]°>' % ((self.label or '?',) + tuple(self.xyzabc[:3])
                             + (self.reach_mm,) + tuple(self.xyzabc[3:])))

    @property
    def reach_mm(self):
        """How far the tip sits from the flange — the lever arm that turns a
        wrist rotation into flange travel."""
        return float(np.linalg.norm(self.t))

    def tip_from_flange(self, x_flange):
        """FK output (flange XYZABC) → tip pose XYZABC."""
        R_f = xyzabc_to_rot(x_flange)
        p_f = np.array(x_flange[:3], dtype=float)
        return mat_to_xyzabc(R_f @ self.R, p_f + R_f @ self.t)

    def flange_from_tip(self, x_tip):
        """Desired tip pose XYZABC → the flange pose to hand the IK."""
        R_t = xyzabc_to_rot(x_tip)
        p_t = np.array(x_tip[:3], dtype=float)
        R_f = R_t @ self.R.T
        return mat_to_xyzabc(R_f, p_t - R_f @ self.t)

    # ----- loading -----

    @classmethod
    def load(cls, name, arm, repo_root):
        """Read one arm's `kine_offset` out of `configs/tool/<name>.yaml`.

        :return: (ToolFrame, cfg_path), or (None, cfg_path) when the entry is
            all zeros — i.e. not measured yet. A zero-length tool is not a
            measurement, and silently treating it as one is how the flange
            ends up impersonating the fingertip.
        :raises ValueError: file missing / invalid / no entry for this arm
            (drivers.tool_config.load_tool_config's own message).
        """
        from drivers.tool_config import load_tool_config

        cfg, cfg_path = load_tool_config(name, arm, repo_root)
        v = list(cfg.kine_offset)
        if not any(abs(x) > 0 for x in v):
            return None, cfg_path
        return cls(v, label='%s/%s' % (name, arm)), cfg_path


def solve_pivot(poses):
    """Least-squares flange→tip translation from poses that touch one point.

    Classic pivot (a.k.a. 4-point TCP) calibration: park the fingertip
    midpoint on one fixed point in space, sweep the wrist through as many
    orientations as the arm allows, record the *flange* pose each time. Every
    sample satisfies

        R_i · t + p_i = c            (c = the touched point, base frame)

    which is linear in the unknowns [t; c]:  [R_i  −I]·[t; c] = −p_i.

    Recovers translation only — the touched point is rotationally symmetric,
    so nothing here constrains the tool's orientation. That is fine: under
    relative mapping the tool rotation does not change a single commanded
    pose (see the module docstring), so translation is the whole answer for
    this pipeline.

    :param poses: iterable of flange XYZABC (mm/deg), ≥3, with genuinely
                  different orientations.
    :return: (t_mm[3], point_mm[3], residual_mm, spread_deg) — residual is
             the RMS distance between each sample's implied tip and the fitted
             point; spread is the largest pairwise rotation angle among the
             samples, the thing that decides whether the fit is conditioned
             at all.
    """
    poses = [list(x) for x in poses]
    if len(poses) < 3:
        raise ValueError('pivot calibration needs at least 3 poses, got %d '
                         '(4+ with widely different wrist orientations is '
                         'what actually conditions the fit)' % len(poses))
    Rs = [xyzabc_to_rot(x) for x in poses]
    ps = [np.array(x[:3], dtype=float) for x in poses]

    A = np.zeros((3 * len(poses), 6))
    b = np.zeros(3 * len(poses))
    for i, (R, p) in enumerate(zip(Rs, ps)):
        A[3 * i:3 * i + 3, 0:3] = R
        A[3 * i:3 * i + 3, 3:6] = -np.eye(3)
        b[3 * i:3 * i + 3] = -p
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    t, c = sol[:3], sol[3:]

    resid = math.sqrt(float(np.mean([np.sum((R @ t + p - c) ** 2)
                                     for R, p in zip(Rs, ps)])))
    spread = 0.0
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            ang = (np.trace(Rs[i] @ Rs[j].T) - 1.0) / 2.0
            spread = max(spread, math.degrees(
                math.acos(max(-1.0, min(1.0, ang)))))
    return list(t), list(c), resid, spread


if __name__ == '__main__':
    rng = np.random.default_rng(0)

    # ---- round trip ----
    tool = ToolFrame([12.0, -3.0, 185.0, 10.0, -5.0, 30.0], label='test')
    worst = 0.0
    for _ in range(200):
        q = rng.normal(size=4)
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]
        x_f = mat_to_xyzabc(R, rng.uniform(-500, 500, 3))
        back = tool.flange_from_tip(tool.tip_from_flange(x_f))
        d = max(abs(a - b) for a, b in zip(x_f[:3], back[:3]))
        d = max(d, max(abs((a - b + 180) % 360 - 180)
                       for a, b in zip(x_f[3:], back[3:])))
        worst = max(worst, d)
    print('flange -> tip -> flange 最大偏差 %.2e  %s'
          % (worst, '✅' if worst < 1e-9 else '❌'))

    # ---- the claim that makes pivot calibration sufficient ----
    # Same translation, wildly different tool rotation => identical commanded
    # flange poses. Run through the REAL Retargeter, not a re-derivation.
    import config
    from algos.retarget import Retargeter, quat_to_mat

    class _F:
        def __init__(self, p, q):
            self.p, self.q = p, q

        def pose(self, hand):
            return tuple(self.p) + tuple(self.q)

        def button(self, hand, name, default=0.0):
            return 1.0 if name == 'grip' else default

    def _run(tool_, delta_frame='base'):
        # A fake 2-link "FK": joint vector is (position mm, rpy deg) itself,
        # so this exercises the frame algebra without needing the SDK.
        def fk(q):
            return list(q)

        rt = Retargeter(config, 'A', tool=tool_, delta_frame=delta_frame)
        q = [420.0, 300.0, 180.0, 0.0, 0.0, 0.0]
        out = []
        for i in range(60):
            t = i * 0.02
            p = [0.30 + 0.05 * math.sin(t), 1.10, -0.25 + 0.05 * math.cos(t)]
            ang = 0.9 * t                       # a real wrist rotation
            quat = [math.sin(ang / 2), 0.0, 0.0, math.cos(ang / 2)]
            rt.update(_F(p, quat), 'left', q, fk, lambda _q: None, dt=0.004)
            out.append(list(rt.target_xyzabc))
            q = list(rt.target_xyzabc)          # perfect tracking
        return out

    t_only = ToolFrame([12.0, -3.0, 185.0])
    rotated = ToolFrame([12.0, -3.0, 185.0, 37.0, -61.0, 149.0])
    a, b = _run(t_only), _run(rotated)
    worst = max(max(abs(u - v) for u, v in zip(ra[:3], rb[:3]))
                for ra, rb in zip(a, b))
    worst_r = max(max(abs((u - v + 180) % 360 - 180)
                      for u, v in zip(ra[3:], rb[3:])) for ra, rb in zip(a, b))
    ok = worst < 1e-9 and worst_r < 1e-9
    print('工具旋转对下发位姿无影响：位置 %.2e mm / 姿态 %.2e °  %s'
          % (worst, worst_r, '✅' if ok else '❌'))

    # ... and that the tool is not simply being ignored: no tool at all must
    # give a *different* answer, or the test above would pass vacuously.
    c = _run(None)
    diff = max(max(abs(u - v) for u, v in zip(ra[:3], rc[:3]))
               for ra, rc in zip(a, c))
    print('有无工具的差异 %.1f mm（应远大于 0，否则上面那条是空跑）  %s'
          % (diff, '✅' if diff > 1.0 else '❌'))

    # The claim above is a property of delta_frame='base' ONLY. Under 'ee'
    # the commanded flange is T_f(0).X.dT.X^-1, a conjugation, so the tool
    # rotation must NOT cancel -- that is the whole reason recorded replay
    # uses that frame and why kine_offset's a/b/c has to be right there.
    # Asserted, not just commented: a future 'simplification' that made the
    # rotation cancel again would silently un-align replay from the policy
    # pipeline and from deployment.
    d, e = _run(t_only, 'ee'), _run(rotated, 'ee')
    worst_ee = max(max(abs(u - v) for u, v in zip(rd[:3], re_[:3]))
                   for rd, re_ in zip(d, e))
    print('delta_frame=ee 下工具旋转必须有影响：位置差 %.1f mm  %s'
          % (worst_ee, '✅' if worst_ee > 1.0 else '❌ 旋转被约掉了，ee 路径退化成 base'))

    # ---- pivot calibration ----
    truth = np.array([11.0, -4.0, 183.0])
    point = np.array([500.0, 120.0, 300.0])
    poses = []
    for _ in range(8):
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]
        poses.append(mat_to_xyzabc(R, point - R @ truth))
    t, c, resid, spread = solve_pivot(poses)
    err = float(np.linalg.norm(np.array(t) - truth))
    print('pivot 标定 恢复 t=[%.2f %.2f %.2f] 误差 %.2e mm  残差 %.2e mm  '
          '姿态张角 %.0f°  %s'
          % (t[0], t[1], t[2], err, resid, spread, '✅' if err < 1e-6 else '❌'))

    # noisy, and only a small orientation spread — the realistic bad case
    poses_n = []
    for _ in range(8):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        ang = math.radians(rng.uniform(-10, 10))    # only ±10° of sweep
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0]])
        R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
        poses_n.append(mat_to_xyzabc(R, point - R @ truth
                                     + rng.normal(scale=0.5, size=3)))
    t2, _, resid2, spread2 = solve_pivot(poses_n)
    err2 = float(np.linalg.norm(np.array(t2) - truth))
    print('  同样 8 个点但只扫 ±10° + 0.5mm 噪声：误差 %.1f mm（残差只有 %.2f mm）'
          % (err2, resid2))
    print('  → 残差小不代表标定准。看姿态张角：%.0f° 太小，t 沿视线方向没有约束。'
          % spread2)
