#!/usr/bin/env python3
"""Safety clamping layer — small pure functions/classes, unit-testable
without hardware or the SDK.

Every "should this frame be sent" decision belongs here, not scattered
across arm_driver. Design principle: reject is safer than allow — every
function defaults to "don't pass" when uncertain.

See docs/algos.md for the isotropic-vs-per-axis clamping data and the
rate-vs-per-frame-constant rationale.
"""
import math


class Verdict:
    """Result of one safety check. If ok=False, joints is unusable and the
    caller should hold the previous command."""

    __slots__ = ('ok', 'joints', 'reason', 'detail', 'clamped')

    def __init__(self, ok, joints=None, reason='ok', detail=None,
                 clamped=False):
        self.ok = ok
        self.joints = joints
        self.reason = reason
        self.detail = detail or {}
        self.clamped = clamped      # passed, but was clamped — worth logging

    def __repr__(self):
        return '<Verdict %s %s%s>' % ('ok' if self.ok else 'BLOCK',
                                      self.reason,
                                      ' (clamped)' if self.clamped else '')


# ---------- individual checks ----------

def clamp_joint_step(q_new, q_prev, max_step_deg):
    """Scale the per-frame joint delta isotropically to fit max_step_deg.
    Returns (scaled joints, whether scaling occurred).

    A clamp, not a rejection — small overshoot just means "follow slower."
    Must scale all axes by the same factor, not clamp each independently,
    or the end effector moves in a direction IK never solved for. See
    docs/algos.md for the measured per-axis vs. isotropic direction error.
    """
    d = [a - b for a, b in zip(q_new, q_prev)]
    if not d:
        return [], False
    m = max(abs(x) for x in d)
    if m <= max_step_deg:
        return list(q_new), False
    f = max_step_deg / m            # max_step_deg=0 (dt<=0) => f=0, no motion
    return [b + x * f for b, x in zip(q_prev, d)], True


def check_limits(q, lim_lo, lim_hi, margin_deg):
    """Limit-margin check. Returns (passed, min margin, worst joint index)."""
    worst, worst_j = float('inf'), None
    for i, v in enumerate(q):
        m = min(lim_hi[i] - v, v - lim_lo[i])
        if m < worst:
            worst, worst_j = m, i
    return worst >= margin_deg, worst, worst_j


def check_tracking_error(q_cmd, q_meas, max_err_deg):
    """Command-vs-measured deviation. Excess usually means the servo fell
    behind, a collision, or an e-stop."""
    if q_meas is None:
        return True, 0.0, None
    errs = [abs(a - b) for a, b in zip(q_cmd, q_meas)]
    m = max(errs)
    return m <= max_err_deg, m, errs.index(m)


def merge_limit_override(lim_lo, lim_hi, override):
    """Merge config's manual limit override into the config-file limits,
    keeping whichever is stricter (e.g. a hand-tuned table for arm B's J6
    that's more conservative than the config file)."""
    lo, hi = list(lim_lo), list(lim_hi)
    for j, (a, b) in (override or {}).items():
        lo[j] = max(lo[j], a)
        hi[j] = min(hi[j], b)
    return lo, hi


# ---------- stateful guards ----------

class Watchdog:
    """XR data freshness watchdog.

    Feed it age (seconds). Only trips on CONSECUTIVE staleness — a single
    stale frame is normal at 90Hz and shouldn't make following stutter.
    """

    def __init__(self, stale_s, consec_required=3):
        self.stale_s = stale_s
        self.consec_required = consec_required
        self.consec = 0

    def feed(self, age_s):
        if age_s > self.stale_s:
            self.consec += 1
        else:
            self.consec = 0
        return self.consec < self.consec_required

    @property
    def tripped(self):
        return self.consec >= self.consec_required

    def reset(self):
        self.consec = 0


class RejectCounter:
    """Consecutive-rejection counter. Repeated rejection at a workspace
    boundary is normal; too many in a row should exit following instead of
    leaving the operator wondering why the robot feels stuck."""

    def __init__(self, max_consec):
        self.max_consec = max_consec
        self.consec = 0
        self.total = 0
        self.by_reason = {}

    def hit(self, reason):
        self.consec += 1
        self.total += 1
        self.by_reason[reason] = self.by_reason.get(reason, 0) + 1
        return self.consec < self.max_consec

    def clear(self):
        self.consec = 0

    @property
    def tripped(self):
        return self.consec >= self.max_consec


class LowPass:
    """First-order low-pass filter for controller position debounce.

    dt is passed per call (the control thread doesn't guarantee a fixed
    period), so alpha is recomputed every call. cutoff_hz<=0 disables filtering.
    """

    def __init__(self, cutoff_hz, dim=3):
        self.cutoff_hz = cutoff_hz
        self.y = None
        self.dim = dim

    def __call__(self, x, dt):
        if self.cutoff_hz <= 0 or dt <= 0:
            return list(x)
        if self.y is None:
            self.y = list(x)
            return list(self.y)
        tau = 1.0 / (2 * math.pi * self.cutoff_hz)
        a = dt / (tau + dt)
        self.y = [yi + a * (xi - yi) for yi, xi in zip(self.y, x)]
        return list(self.y)

    def reset(self, x=None):
        self.y = list(x) if x is not None else None


class SafetyGate:
    """Chains all the checks above into one gate; the control loop calls
    gate.check() once per frame.

    The step limit is a RATE (deg/s), not a fixed per-frame amount — per-
    frame budget = rate x dt, so loop jitter no longer affects end-effector
    speed. See docs/algos.md for why a fixed-per-frame constant was wrong.
    The rate must be strictly tighter than what the controller can actually
    execute, or q_cmd outruns q_meas and tracking error accumulates until
    the gate latches.
    """

    def __init__(self, lim_lo, lim_hi, max_rate_deg_s, limit_margin_deg,
                 max_tracking_err_deg, max_consec_reject, stale_s,
                 dt_max=0.016, max_tracking_err_s=0.5):
        """:param max_rate_deg_s: per-joint speed limit (deg/s)
        :param dt_max: cap on how much elapsed time one frame can redeem, so
                       a stalled loop doesn't cash in a huge jump; default
                       16ms ~= 4 periods at 250Hz.
        :param max_tracking_err_s: how long the error must stay over limit
                       before it counts as a real fault, vs. a transient
                       "just skip this frame."
        """
        self.lim_lo = lim_lo
        self.lim_hi = lim_hi
        self.max_rate_deg_s = max_rate_deg_s
        self.dt_max = dt_max
        self.limit_margin_deg = limit_margin_deg
        self.max_tracking_err_deg = max_tracking_err_deg
        self.max_tracking_err_s = max_tracking_err_s
        self.watchdog = Watchdog(stale_s)
        self.rejects = RejectCounter(max_consec_reject)
        self.stats = {'pass': 0, 'clamped': 0, 'track_block': 0}
        self.max_step_used = 0.0        # observed: largest per-frame delta actually used
        self.track_err_s = 0.0          # accumulated duration of sustained excess

    def step_budget(self, dt):
        """Per-joint delta allowed this frame (deg). dt<=0 => 0 (no motion),
        not "unlimited" — no elapsed time should mean no displacement."""
        if dt is None or dt <= 0:
            return 0.0
        return self.max_rate_deg_s * min(dt, self.dt_max)

    def check(self, q_new, q_prev, q_meas, xr_age_s, dt):
        # 1) XR link freshness
        if not self.watchdog.feed(xr_age_s):
            return Verdict(False, reason='xr_stale',
                           detail={'age_s': round(xr_age_s, 4)})
        # 2) servo tracking (vs. the previous command; this frame not sent yet)
        ok, err, j = check_tracking_error(q_prev, q_meas,
                                          self.max_tracking_err_deg)
        if not ok:
            self.track_err_s += max(dt, 0.0) if dt else 0.0
            self.stats['track_block'] += 1
            return Verdict(False, reason='tracking_error',
                           detail={'max_err_deg': round(err, 3), 'joint': j,
                                   'held_s': round(self.track_err_s, 3),
                                   'sustained':
                                       self.track_err_s
                                       >= self.max_tracking_err_s})
        self.track_err_s = 0.0
        # 3) step clamp (budget redeemed against actual dt)
        q, clamped = clamp_joint_step(q_new, q_prev, self.step_budget(dt))
        # 4) limit margin (checked on the clamped value)
        ok, margin, j = check_limits(q, self.lim_lo, self.lim_hi,
                                     self.limit_margin_deg)
        if not ok:
            return Verdict(False, reason='joint_limit',
                           detail={'margin_deg': round(margin, 2), 'joint': j})
        self.stats['pass'] += 1
        if clamped:
            self.stats['clamped'] += 1
        self.max_step_used = max(self.max_step_used,
                                 max(abs(a - b) for a, b in zip(q, q_prev)))
        return Verdict(True, joints=q, clamped=clamped)


if __name__ == '__main__':
    # Self-test: no robot, no SDK required
    lo, hi = [-170] * 7, [170] * 7
    q_prev = [0.0] * 7

    q, c = clamp_joint_step([10] + [0] * 6, q_prev, 0.6)
    assert c and abs(q[0] - 0.6) < 1e-9, q
    print('✓ step clamp: a 10° jump clamped to 0.6°')

    # isotropic scaling must preserve the ratio across all 7 axes
    want = [0.05, 0.10, 0.80, 0.0, 0.0, 0.0, 0.0]
    q, c = clamp_joint_step(want, [0.0] * 7, 0.202)
    assert c and abs(q[2] - 0.202) < 1e-9
    assert abs(q[0] / q[2] - want[0] / want[2]) < 1e-12
    assert abs(q[1] / q[2] - want[1] / want[2]) < 1e-12
    print('✓ isotropic scaling: 1:2:16 ratio preserved (per-axis clamp would give 1:2:4)')

    # under budget: passes through unscaled
    q, c = clamp_joint_step([0.1, -0.2] + [0.0] * 5, [0.0] * 7, 0.6)
    assert not c and q == [0.1, -0.2] + [0.0] * 5
    print('✓ under budget: no scaling applied')

    lo2, hi2 = merge_limit_override(lo, hi, {5: (-58.0, 58.0)})
    assert lo2[5] == -58 and hi2[5] == 58
    print('✓ limit override: stricter ±58 wins')

    ok, m, j = check_limits([0, 0, 0, 0, 0, 165, 0], lo, hi, 8.0)
    assert not ok and j == 5
    print('✓ limit margin: 165° leaves only 5° of 170°, rejected')

    wd = Watchdog(0.1, consec_required=3)
    assert wd.feed(0.2) and wd.feed(0.2) and not wd.feed(0.2)
    print('✓ watchdog: trips only after 3 consecutive stale frames')

    lp = LowPass(8.0)
    lp([0, 0, 0], 0.004)
    y = lp([100, 0, 0], 0.004)
    assert 0 < y[0] < 100
    print('✓ low-pass: step input smoothed to %.1f' % y[0])

    # 150°/s @ 250Hz nominal period = 0.6°/frame, matches the old constant
    DT = 1.0 / 250
    gate = SafetyGate(lo, hi, 150.0, 8.0, 5.0, 15, 0.1)
    assert abs(gate.step_budget(DT) - 0.6) < 1e-9
    print('✓ budget redeemed by dt: 150°/s x 4ms = %.2f°' % gate.step_budget(DT))

    # loop running 2x slower => budget doubles (end-effector speed unchanged) —
    # the whole point of switching to a rate
    assert abs(gate.step_budget(2 * DT) - 1.2) < 1e-9
    # but a stall can't redeem unboundedly: dt_max caps it
    assert abs(gate.step_budget(1.0) - 150.0 * gate.dt_max) < 1e-9
    print('✓ stall capped: dt=1s still only allows %.2f° (dt_max=%.0fms)'
          % (gate.step_budget(1.0), gate.dt_max * 1e3))

    # dt<=0 => budget 0 (no motion), not "unlimited"
    assert gate.step_budget(0.0) == 0.0 and gate.step_budget(-1) == 0.0
    v = gate.check([9.0] * 7, q_prev, q_prev, 0.01, dt=0.0)
    assert v.ok and v.clamped and v.joints == q_prev, v.joints
    print('✓ dt=0: budget is 0, command holds in place (not a big jump)')

    v = gate.check([0.3] * 7, q_prev, q_prev, 0.01, dt=DT)
    assert v.ok and not v.clamped
    v = gate.check([5.0] * 7, q_prev, q_prev, 0.01, dt=DT)
    assert v.ok and v.clamped and abs(v.joints[0] - 0.6) < 1e-9
    # tracking error: transient excess reports sustained=False (backpressure only)
    g2 = SafetyGate(lo, hi, 150.0, 8.0, 5.0, 15, 0.1, max_tracking_err_s=0.5)
    far = [9.0] * 7
    v = g2.check([0.1] * 7, q_prev, far, 0.01, dt=DT)
    assert not v.ok and v.reason == 'tracking_error'
    assert v.detail['sustained'] is False, 'transient excess must not fault'
    for _ in range(int(0.5 / DT)):
        v = g2.check([0.1] * 7, q_prev, far, 0.01, dt=DT)
    assert v.detail['sustained'] is True, 'must fault after 0.5s sustained'
    print('✓ tracking error: transient -> sustained=False, after %.1fs -> True'
          % g2.max_tracking_err_s)
    g2.check([0.1] * 7, q_prev, q_prev, 0.01, dt=DT)     # error recovers
    assert g2.track_err_s == 0.0, 'timer must reset once error recovers'
    print('✓ timer resets on recovery, transients cannot accumulate into a fault')

    v = gate.check([0.1] * 7, q_prev, q_prev, 0.5, dt=DT)
    assert v.ok, 'a single stale frame must not trip the gate'
    gate.check([0.1] * 7, q_prev, q_prev, 0.5, dt=DT)
    v = gate.check([0.1] * 7, q_prev, q_prev, 0.5, dt=DT)
    assert not v.ok and v.reason == 'xr_stale', v
    print('✓ gate chain works end to end; verdict after 3 stale frames =', v)

    # P0-2 invariant: command-stream rate limit must be stricter than the controller's
    import config
    ctrl = config.JOINT_VMAX_DEG_S * config.VEL_RATIO / 100.0
    assert config.MAX_JOINT_RATE_DEG_S < ctrl, 'bottleneck inverted'
    print('✓ speed budget not inverted: command %.0f°/s < controller %.0f°/s'
          % (config.MAX_JOINT_RATE_DEG_S, ctrl))
    print('\nall passed')
