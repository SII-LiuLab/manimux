#!/usr/bin/env python3
"""Embodiment filter — which parts of a human demo can THIS robot actually do.

See docs/algos.md#embodimentpy. CLI: scripts/umi_filter.py.

A hand-held UMI demo is recorded by an arm with 7 joints, no joint limits, no
velocity budget and no self-collision to speak of. Replaying it on the Tianji
turns some of those frames into "IK has no solution", "that joint would leave
its limit" or "the wrist would have to turn faster than the speed budget
allows". This module walks a recorded episode through the **real** pipeline —
`Retargeter` → `ArmIK.solve_with_backoff` → `SafetyGate`, the same objects
`core.arm_channel.ArmChannel` wires up — and labels every control period, then
cuts the episode into spans the robot can reproduce and spans it cannot.

Three things worth knowing before trusting the output:

**1. Feasibility is not a property of the demo alone.** Relative mapping
reproduces the demonstrator's *displacement* from wherever the arm starts, so
the same 200mm reach is easy from one start configuration and off the edge of
the workspace from another. Every verdict here is "…from `q0`", and `q0` is an
argument. `config.UMI_START_JOINTS` is the sane default because that is what
`scripts/umi_replay.py --goto-start` puts the arm at.

**2. Run it with the tool frame set.** A recorded pose is the leader gripper's
fingertip midpoint. Without an `algos.tool_frame.ToolFrame` the filter judges
a flange trajectory the robot will never be asked to follow, and it gets
rotation-heavy frames wrong in both directions. `scripts/umi_filter.py` warns
loudly when the offset is still unmeasured.

**3. "Too fast" is not "impossible".** The speed budget scales with
`--speed`/`VEL_RATIO`; the workspace does not. Those two failure modes get
different codes on purpose (`speed` vs `ik:*`/`gate:*`) because only one of
them is fixed by slowing the replay down.

There is deliberately no self-collision model here, same as everywhere else in
this stack. `keepout` re-uses `config.UMI_REPLAY_KEEPOUT`, which is a
provisional box, not a measured body envelope — see config.py's own comment.
"""
import bisect
import math

from algos.ik_solver import ArmIK
from algos.nullspace import JointLimitAvoidance, NullSpaceController
from algos.retarget import Retargeter
from algos.safety import LowPass, SafetyGate

# Verdict codes. Anything not in FEASIBLE is a frame the robot did not
# reproduce; the prefix before ':' is what to blame.
OK = 'ok'                   # followed inside every budget
SLOW = 'slow'               # limiter clamped, but the tip stayed within
                            # max_lag -- degraded, still a faithful path
FEASIBLE = (OK, SLOW)
SPEED = 'speed'             # over the speed budget for long enough that the
                            # tip fell behind the demo
BROKEN = 'broken'           # after a fault, with resync disabled


class FrameVerdict:
    """One control period's verdict. Small on purpose: one per 4ms."""

    __slots__ = ('code', 'margin_deg', 'rate_ratio', 'lag_mm', 'rot_lag_deg',
                 'tip', 'flange')

    def __init__(self, code, margin_deg=None, rate_ratio=0.0, lag_mm=0.0,
                 rot_lag_deg=0.0, tip=None, flange=None):
        self.code = code
        self.margin_deg = margin_deg    # min distance to a joint limit, deg
        self.rate_ratio = rate_ratio    # required joint rate / budget
        self.lag_mm = lag_mm            # tip distance behind the demo
        self.rot_lag_deg = rot_lag_deg
        self.tip = tip                  # [x,y,z] mm, base frame
        self.flange = flange

    @property
    def ok(self):
        return self.code in FEASIBLE

    def __repr__(self):
        return '<%s lag=%.1fmm rate=%.2f>' % (self.code, self.lag_mm,
                                              self.rate_ratio)


class Segment:
    """A contiguous span of control periods, feasible or not."""

    __slots__ = ('start', 'end', 'ok', 'reasons', 'bridged', 'worst_margin',
                 'max_rate_ratio', 'max_lag_mm', 'lo', 'hi', 'rec_start',
                 'rec_end', 't_start', 't_end', 'recheck')

    def __init__(self, start, end, ok):
        self.start = start          # inclusive, control-period index
        self.end = end              # exclusive
        self.ok = ok
        self.reasons = {}           # code -> count
        self.bridged = 0            # infeasible frames swallowed by bridging
        self.worst_margin = None
        self.max_rate_ratio = 0.0
        self.max_lag_mm = 0.0
        self.lo = None              # tip envelope, base frame mm
        self.hi = None
        self.rec_start = None       # recorded (30Hz) frame index, inclusive
        self.rec_end = None         # recorded frame index, exclusive
        self.t_start = 0.0          # seconds into the episode
        self.t_end = 0.0
        self.recheck = None         # filled by recheck_from_start()

    @property
    def n(self):
        return self.end - self.start

    def __repr__(self):
        return '<Segment %s %d..%d (%d帧)>' % ('ok' if self.ok else 'CUT',
                                               self.start, self.end, self.n)


class ArmScan:
    """One arm's pass over one episode."""

    def __init__(self, arm, hand, q0, tool, cfg):
        self.arm = arm
        self.hand = hand
        self.q0 = list(q0)
        self.tool = tool
        self.cfg = cfg
        self.verdicts = []
        self.breaks = []            # frames where following had to re-anchor
        self.segments = []
        self.max_rate_ratio = 0.0
        self.max_lag_mm = 0.0
        self.max_rot_lag_deg = 0.0
        self.worst_margin = None
        self.ik_stats = {}

    @property
    def n(self):
        return len(self.verdicts)

    def counts(self):
        out = {}
        for v in self.verdicts:
            out[v.code] = out.get(v.code, 0) + 1
        return out

    @property
    def n_feasible(self):
        return sum(1 for v in self.verdicts if v.ok)

    @property
    def kept(self):
        """Frames inside kept segments — the number that actually matters,
        since short feasible slivers get dropped by min_len."""
        return sum(s.n for s in self.segments if s.ok)


def _make_channel(arm, cfg, axis_map, tool, ik, delta_frame='base'):
    """The same Retargeter/SafetyGate wiring ArmChannel builds, minus the
    driver. Kept in one place so the filter cannot drift from the pipeline
    it is supposed to be predicting."""
    lo, hi = ik.lim_n, ik.lim_p
    gate = SafetyGate(lo, hi, cfg.MAX_JOINT_RATE_DEG_S, cfg.LIMIT_MARGIN_DEG,
                      cfg.MAX_TRACKING_ERR_DEG, cfg.MAX_CONSEC_REJECT,
                      cfg.XR_STALE_S, dt_max=cfg.MAX_STEP_DT_S,
                      max_tracking_err_s=cfg.MAX_TRACKING_ERR_S)
    ns = None
    if cfg.NULLSPACE_ENABLED:
        ns = NullSpaceController(
            [JointLimitAvoidance(lo, hi,
                                 activation_deg=cfg.NULLSPACE_ACTIVATION_DEG)],
            rate_deg_s=cfg.NULLSPACE_RATE_DEG_S,
            limit_deg=cfg.ARM_ANGLE_LIMIT,
            probe_deg=cfg.NULLSPACE_PROBE_DEG)
    rt = Retargeter(cfg, arm, lowpass=LowPass(cfg.LOWPASS_HZ), nullspace=ns,
                    axis_map=axis_map, tool=tool, delta_frame=delta_frame)
    return rt, gate


def make_ik(arm, cfg):
    """One ArmIK for an arm. Hand it back to scan_arm() when scanning the
    same arm repeatedly (recheck, speed sweep) — constructing one reloads
    and re-inits the whole kinematics config."""
    return ArmIK(arm_type=0 if arm == 'A' else 1, config_path=cfg.KINE_CFG,
                 pos_tol_mm=cfg.IK_POS_TOL_MM, rot_tol_deg=cfg.IK_ROT_TOL_DEG,
                 max_step_deg=cfg.IK_MAX_STEP_DEG,
                 limit_margin_deg=cfg.LIMIT_MARGIN_DEG,
                 limit_override=cfg.JOINT_LIMIT_OVERRIDE.get(arm))


def _keepout_violation(box, flange, tip):
    """Which axis (if any) leaves the keep-out box. Both the flange and the
    tip are checked: the box guards the robot's own body, and the tool sticks
    out further than the flange it is bolted to."""
    if not box:
        return None
    for k, ax in enumerate('XYZ'):
        lim = box.get(ax)
        if not lim:
            continue
        for p in (flange, tip):
            if p is not None and (p[k] < lim[0] or p[k] > lim[1]):
                return ax
    return None


def scan_arm(frames, hand, arm, cfg, q0, tool=None, axis_map=None,
             keepout=None, max_lag_mm=5.0, max_rot_lag_deg=3.0, resync=True,
             ik=None, delta_frame='ee'):
    """Label every control period of `frames` for one arm.

    :param frames: already resampled to cfg.CONTROL_HZ (drivers.umi_source.
                   resample) — the filter must see the same timeline the
                   control loop will.
    :param q0:     starting configuration. Feasibility is relative to it.
    :param tool:   algos.tool_frame.ToolFrame, or None to judge the flange.
    :param delta_frame: must match what the replay entry point will use, or
                   this predicts a trajectory nobody will run. Defaults to
                   'ee' because this filter exists for the recorded-UMI path
                   and scripts/umi_replay.py uses the body-frame delta; then
                   axis_map is unused. See algos/retarget.py's docstring.
    :param resync: after a fault, re-anchor on the next frame and keep
                   scanning (data curation: cut the bad span, keep the rest).
                   False stops at the first fault, which answers the other
                   question — how much of this demo survives in one take.
    :return: ArmScan
    """
    ik = ik or make_ik(arm, cfg)
    scan = ArmScan(arm, hand, q0, tool, cfg)
    period = 1.0 / cfg.CONTROL_HZ
    budget = cfg.MAX_JOINT_RATE_DEG_S

    def solve_joints(target, q_now, zsp_dir, arm_angle):
        with ik.probing():
            r = ik.solve_with_backoff(target, q_now, zsp_dir=zsp_dir,
                                      arm_angle=arm_angle)
        return r.joints if r.ok else None

    rt, gate = _make_channel(arm, cfg, axis_map, tool, ik, delta_frame)
    q_cmd = list(q0)
    consec = 0
    broken = False
    # ik may be shared across passes (recheck, speed sweep), so report this
    # pass's delta rather than the instance's running total.
    stats0 = dict(ik.stats)

    for i, f in enumerate(frames):
        if broken:
            scan.verdicts.append(FrameVerdict(BROKEN))
            continue
        if rt is None:              # re-anchor: fresh latch on this frame
            rt, gate = _make_channel(arm, cfg, axis_map, tool, ik,
                                     delta_frame)
            scan.breaks.append(i)

        if not rt.update(f, hand, q_cmd, ik.fk_xyzabc, ik.nsp_dir, dt=period,
                         solve=solve_joints):
            # Only reachable if the frame has no pose for this hand; a UMI
            # frame reports the clutch permanently pressed.
            scan.verdicts.append(FrameVerdict('no_pose'))
            continue

        lag, rot_lag = rt.cart_lag_mm, rt.rot_lag_deg
        r = ik.solve_with_backoff(rt.target_xyzabc, q_cmd,
                                  zsp_dir=rt.zsp_dir, arm_angle=rt.arm_angle)
        if not r.ok:
            scan.verdicts.append(FrameVerdict('ik:' + r.reason, lag_mm=lag,
                                              rot_lag_deg=rot_lag))
            consec += 1
            if consec >= cfg.MAX_CONSEC_REJECT:
                rt, consec, broken = None, 0, not resync
            continue

        # Required rate is measured BEFORE the gate clamps it -- that is the
        # "over the speed budget" number, and the clamped one never exceeds
        # the budget by construction.
        rate = max(abs(a - b) for a, b in zip(r.joints, q_cmd)) / period
        v = gate.check(r.joints, q_cmd, q_cmd, 0.0, period)
        if not v.ok:
            scan.verdicts.append(FrameVerdict('gate:' + v.reason,
                                              rate_ratio=rate / budget,
                                              lag_mm=lag, rot_lag_deg=rot_lag))
            consec += 1
            # ArmChannel treats joint_limit as an immediate fault, not as
            # backpressure -- mirror that or the filter is predicting a
            # pipeline that does not exist.
            if v.reason == 'joint_limit' or consec >= cfg.MAX_CONSEC_REJECT:
                rt, consec, broken = None, 0, not resync
            continue

        consec = 0
        q_cmd = v.joints
        x = ik.fk_xyzabc(q_cmd)
        flange = list(x[:3])
        tip = list(tool.tip_from_flange(x)[:3]) if tool else list(flange)

        ax = _keepout_violation(keepout, flange, tip)
        if ax is not None:
            code = 'keepout:' + ax
        elif lag > max_lag_mm or rot_lag > max_rot_lag_deg:
            code = SPEED
        elif v.clamped or rate > budget:
            code = SLOW
        else:
            code = OK
        scan.verdicts.append(FrameVerdict(
            code, margin_deg=r.min_margin_deg, rate_ratio=rate / budget,
            lag_mm=lag, rot_lag_deg=rot_lag, tip=tip, flange=flange))

        scan.max_rate_ratio = max(scan.max_rate_ratio, rate / budget)
        scan.max_lag_mm = max(scan.max_lag_mm, lag)
        scan.max_rot_lag_deg = max(scan.max_rot_lag_deg, rot_lag)
        if r.min_margin_deg is not None:
            scan.worst_margin = r.min_margin_deg if scan.worst_margin is None \
                else min(scan.worst_margin, r.min_margin_deg)

    scan.ik_stats = {k: v - stats0.get(k, 0) for k, v in ik.stats.items()
                     if v - stats0.get(k, 0)}
    return scan


# ---------- segmentation ----------

def segment(verdicts, min_len, bridge):
    """Cut a verdict stream into feasible / infeasible spans.

    :param min_len: drop feasible spans shorter than this many frames. A
        50ms island of "the robot could have done that" is not a usable demo
        segment, it is noise between two failures.
    :param bridge: swallow infeasible spans shorter than this many frames
        into the surrounding feasible one. The pipeline holds its last
        command through a rejection and usually recovers on the next frame
        (that is what solve_with_backoff is for), so a 2-frame hiccup is
        backpressure, not a break. Bridged frames are counted per segment and
        must be reported, not hidden.

    Order matters: bridge first, then drop short spans, or a long feasible
    span split by one hiccup gets thrown away as two short ones.
    """
    n = len(verdicts)
    ok = [v.ok for v in verdicts]
    bridged_at = [False] * n

    i = 0
    while i < n:
        if ok[i]:
            i += 1
            continue
        j = i
        while j < n and not ok[j]:
            j += 1
        # Only an *interior* gap can be bridged: a run at either end has no
        # feasible span on both sides to belong to.
        if 0 < i and j < n and (j - i) <= bridge:
            for k in range(i, j):
                ok[k], bridged_at[k] = True, True
        i = j

    out, i = [], 0
    while i < n:
        j = i
        while j < n and ok[j] == ok[i]:
            j += 1
        seg = Segment(i, j, ok[i])
        for k in range(i, j):
            v = verdicts[k]
            seg.reasons[v.code] = seg.reasons.get(v.code, 0) + 1
            if bridged_at[k]:
                seg.bridged += 1
            if v.margin_deg is not None:
                seg.worst_margin = v.margin_deg if seg.worst_margin is None \
                    else min(seg.worst_margin, v.margin_deg)
            seg.max_rate_ratio = max(seg.max_rate_ratio, v.rate_ratio)
            seg.max_lag_mm = max(seg.max_lag_mm, v.lag_mm)
            if v.tip is not None:
                if seg.lo is None:
                    seg.lo, seg.hi = list(v.tip), list(v.tip)
                else:
                    for a in range(3):
                        seg.lo[a] = min(seg.lo[a], v.tip[a])
                        seg.hi[a] = max(seg.hi[a], v.tip[a])
        out.append(seg)
        i = j

    # Demote too-short feasible spans, then merge neighbours so the caller
    # never sees two adjacent infeasible segments.
    for seg in out:
        if seg.ok and seg.n < min_len:
            seg.ok = False
            n_ok = sum(c for k, c in seg.reasons.items() if k in FEASIBLE)
            for k in FEASIBLE:
                seg.reasons.pop(k, None)    # 'ok: 57' inside a cut span reads
            seg.reasons['too_short'] = n_ok  # like a bug; say why it was cut
    return merge_cuts(out)


def merge_cuts(segments):
    """Fuse adjacent cut spans into one. Two cut spans in a row are an
    artifact of *how* they were cut (one demoted for length, one infeasible),
    not something a reader needs to see -- and callers that demote a span
    later (umi_filter's recheck) reintroduce exactly that."""
    merged = []
    for seg in segments:
        if merged and not merged[-1].ok and not seg.ok:
            prev = merged[-1]
            prev.end = seg.end
            prev.bridged += seg.bridged
            prev.max_rate_ratio = max(prev.max_rate_ratio, seg.max_rate_ratio)
            prev.max_lag_mm = max(prev.max_lag_mm, seg.max_lag_mm)
            for k, c in seg.reasons.items():
                prev.reasons[k] = prev.reasons.get(k, 0) + c
            continue
        merged.append(seg)
    return merged


def stamp_times(segments, sim_frames, raw_frames):
    """Fill in each segment's wall-clock span and its **recorded** frame range.

    Everything downstream of this repo indexes episodes by recorded frame
    (30Hz), not by control period (250Hz), so a mask expressed in control
    periods would be unusable. Recorded indices are found by searching the
    raw timestamps rather than dividing by fps: a dropped recording frame
    would silently shift every index the arithmetic way.
    """
    if not sim_frames or not raw_frames:
        return segments
    ts = [f.host_ns for f in raw_frames]
    t0 = ts[0]

    def rec_index(host_ns):
        j = bisect.bisect_right(ts, host_ns) - 1
        if j < 0:
            return 0.0
        if j >= len(ts) - 1:
            return float(len(ts) - 1)
        span = ts[j + 1] - ts[j]
        u = 0.0 if span <= 0 else (host_ns - ts[j]) / span
        return j + u

    for s in segments:
        a = sim_frames[min(s.start, len(sim_frames) - 1)].host_ns
        b = sim_frames[min(s.end - 1, len(sim_frames) - 1)].host_ns
        s.t_start = (a - t0) / 1e9
        s.t_end = (b - t0) / 1e9
        # Round inward: a kept span must not claim a recorded frame that is
        # only partly inside it.
        s.rec_start = int(math.ceil(rec_index(a) - 1e-9))
        s.rec_end = min(len(ts), int(math.floor(rec_index(b) + 1e-9)) + 1)
    return segments


def intersect(keeps, n):
    """Frame ranges feasible for EVERY arm.

    A bimanual episode is only usable where both arms can follow: one arm
    stalling mid-reach makes the other arm's frames unusable too, because the
    recorded action for that instant no longer describes what the robot did.

    :param keeps: one list of (start, end) per arm, control-period indices.
    """
    if not keeps:
        return []
    mask = [True] * n
    for spans in keeps:
        hit = [False] * n
        for a, b in spans:
            for i in range(max(0, a), min(n, b)):
                hit[i] = True
        mask = [m and h for m, h in zip(mask, hit)]
    out, i = [], 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        out.append((i, j))
        i = j
    return out


if __name__ == '__main__':
    # Segmentation logic only -- no SDK, no robot, no dataset.
    def V(code):
        return FrameVerdict(code)

    def codes(s):
        return [V(OK if c == '.' else 'ik:ik_failed') for c in s]

    segs = segment(codes('..........' + 'XX' + '..........'),
                   min_len=3, bridge=4)
    assert len(segs) == 1 and segs[0].ok and segs[0].bridged == 2, segs
    print('✓ 短暂拒解被桥接：22 帧合成 1 段，其中 2 帧被记为 bridged')

    segs = segment(codes('..........' + 'X' * 20 + '..........'),
                   min_len=3, bridge=4)
    assert [(s.start, s.end, s.ok) for s in segs] == \
        [(0, 10, True), (10, 30, False), (30, 40, True)], segs
    print('✓ 长故障不被桥接：切成 可用/剔除/可用 三段')

    segs = segment(codes('..' + 'X' * 20 + '..........'), min_len=3, bridge=4)
    assert [(s.start, s.end, s.ok) for s in segs] == \
        [(0, 22, False), (22, 32, True)], segs
    print('✓ 过短的可用段被降级，并与相邻剔除段合并（不会留下两段相邻的剔除段）')

    # bridge-then-drop ordering: one hiccup inside a long span must not turn
    # it into two short spans that both get dropped
    segs = segment(codes('.....X.....'), min_len=8, bridge=2)
    assert len(segs) == 1 and segs[0].ok, segs
    print('✓ 先桥接后剔短：一次打嗝不会把长段拆成两段短的然后全丢掉')

    assert segment(codes('X....X'), min_len=1, bridge=4)[0].ok is False
    print('✓ 首尾的故障不桥接（没有两侧可用段可归属）')

    n = 20
    assert intersect([[(0, 10)], [(5, 20)]], n) == [(5, 10)]
    assert intersect([[(0, 5), (10, 15)], [(0, 20)]], n) == [(0, 5), (10, 15)]
    print('✓ 双臂取交集：只有两条臂都能跟的帧才算可用')
