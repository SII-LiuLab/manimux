#!/usr/bin/env python3
"""Null-space control for the redundant 7-DoF arm.

Exposes the arm's redundancy as a single scalar (arm angle) about a
reference plane latched at clutch-press; sliding it moves the elbow along
the self-motion manifold while the end effector stays fixed.

The caller supplies an `evaluate(angle) -> joints` callback so the SDK
stays out of this module (same pattern as safety.py).

See docs/algos.md for the redundancy-geometry measurements and the
probe-threshold data behind NullSpaceController.
"""
import math


class NullSpaceObjective:
    """Scores a configuration. Lower cost is better.

    Subclass this to add a new secondary task (manipulability, obstacle
    clearance, ...). Costs are combined by weighted sum, so keep them
    dimensionless and roughly O(1) over the working range.
    """

    name = 'objective'

    def __init__(self, weight=1.0):
        self.weight = weight

    def cost(self, q):
        raise NotImplementedError

    def reset(self):
        """Called on engage. Override only if the objective is stateful."""


class JointLimitAvoidance(NullSpaceObjective):
    """Liegeois (1977) joint-range-availability criterion.

    activation_deg=None: classic form, always pulls every joint toward the
        middle of its travel:
            H = 1/(2n) * sum_i w_i * ((q_i - mid_i) / span_i)^2
    activation_deg=d: practical form, a joint costs nothing until within d
        of its limit:
            H = 1/(2n) * sum_i w_i * max(0, (d - margin_i) / d)^2

    Prefer the activated form — see docs/algos.md for the reach-tradeoff
    data. `weights` lets unevenly-ranged joints (e.g. J6, ±60°) cost more
    per degree lost.
    """

    name = 'joint_limit'

    def __init__(self, lim_lo, lim_hi, activation_deg=None, weights=None,
                 weight=1.0):
        super().__init__(weight)
        if len(lim_lo) != len(lim_hi):
            raise ValueError('lim_lo/lim_hi length mismatch')
        if activation_deg is not None and activation_deg <= 0:
            raise ValueError('activation_deg must be positive or None')
        self.lim_lo = list(lim_lo)
        self.lim_hi = list(lim_hi)
        self.activation_deg = activation_deg
        self.mid = [(a + b) / 2.0 for a, b in zip(lim_lo, lim_hi)]
        self.span = [max(b - a, 1e-9) for a, b in zip(lim_lo, lim_hi)]
        self.weights = list(weights) if weights else [1.0] * len(lim_lo)

    def margins(self, q):
        return [min(hi - v, v - lo) for v, lo, hi
                in zip(q, self.lim_lo, self.lim_hi)]

    def cost(self, q):
        n = len(q)
        if self.activation_deg is None:
            return sum(w * ((v - m) / s) ** 2 for v, m, s, w
                       in zip(q, self.mid, self.span, self.weights)) / (2.0 * n)
        d = self.activation_deg
        return sum(w * ((d - m) / d) ** 2 for m, w
                   in zip(self.margins(q), self.weights)
                   if m < d) / (2.0 * n)


class NullSpaceController:
    """Drives the arm angle down the weighted objective gradient.

    Samples the descent direction by probing ±probe_deg (two extra IK
    calls/frame) since the angle→configuration map isn't differentiable
    here. probe_deg must stay under the IK branch-jump threshold or probes
    collapse to a single configuration and the gradient vanishes silently
    (see n_blind). Rate-limited because the arm angle feeds IK's continuity
    check.

    See docs/algos.md for the probe-sensitivity measurements.
    """

    def __init__(self, objectives, rate_deg_s, limit_deg,
                 probe_deg=0.5, deadband=1e-6):
        self.objectives = list(objectives)
        self.rate_deg_s = rate_deg_s
        self.limit_deg = limit_deg
        self.probe_deg = probe_deg
        self.deadband = deadband
        self.angle = 0.0
        self.cost_now = None
        self.n_moved = 0
        self.n_blind = 0        # frames where both probes returned q(angle)

    def reset(self, angle=0.0):
        self.angle = max(-self.limit_deg, min(self.limit_deg, angle))
        self.cost_now = None
        self.n_moved = 0
        self.n_blind = 0
        for o in self.objectives:
            o.reset()

    def cost(self, q):
        return sum(o.weight * o.cost(q) for o in self.objectives)

    def update(self, evaluate, dt, bias_deg_s=0.0):
        """Return the arm angle to command this frame.

        :param evaluate:    fn(angle) -> joint list, or None when that angle
                            has no usable IK solution (it is then skipped)
        :param bias_deg_s:  operator override (joystick), added on top
        """
        step = self.rate_deg_s * max(dt, 0.0)
        here, q_here = self._probe(evaluate, self.angle)
        self.cost_now = here
        if here is not None and step > 0.0:
            best_cost, best_dir, blind = here, 0.0, True
            for direction in (-1.0, 1.0):
                cost, q = self._probe(evaluate,
                                      self.angle + direction * self.probe_deg)
                if q is None:
                    continue
                if any(abs(a - b) > 1e-9 for a, b in zip(q, q_here)):
                    blind = False
                if cost < best_cost - self.deadband:
                    best_cost, best_dir = cost, direction
            if blind:
                self.n_blind += 1
            if best_dir:
                self.angle += best_dir * step
                self.n_moved += 1
        self.angle += bias_deg_s * max(dt, 0.0)
        self.angle = max(-self.limit_deg, min(self.limit_deg, self.angle))
        return self.angle

    def _probe(self, evaluate, angle):
        if abs(angle) > self.limit_deg:
            return None, None
        q = evaluate(angle)
        return (None, None) if q is None else (self.cost(q), q)


if __name__ == '__main__':
    # No robot, no SDK: synthetic manifold trading J2 against J6, mimicking
    # the real arm's redundancy.
    LO = [-170, -120, -170, -145, -170, -60, -90]
    HI = [170, 120, 170, 60, 170, 60, 90]

    def manifold(angle):
        return [0.0, -90.0 + angle, 0.0, -90.0, 0.0, -angle * 0.5, 0.0]

    obj = JointLimitAvoidance(LO, HI)
    # J4 travels -145..+60, so "centred" is not all zeros
    assert obj.cost(obj.mid) == 0.0, 'centred config must cost nothing'
    assert obj.cost(manifold(30)) < obj.cost(manifold(0))
    print('✓ limit cost: minimal at midpoint, drops as J2 moves -90 -> -60')

    # Activated form: silent while everything is roomy, bites near a limit
    act = JointLimitAvoidance(LO, HI, activation_deg=25.0)
    assert act.cost(manifold(0)) == 0.0, 'J2 margin 30° > 25°, cost should be 0'
    assert act.cost(manifold(-20)) > 0.0, 'J2 margin 10° < 25°, cost should be >0'
    assert act.cost(manifold(-30)) > act.cost(manifold(-20))
    print('✓ activation threshold: zero cost beyond 25°, rises as margin shrinks')

    idle = NullSpaceController([act], rate_deg_s=20.0, limit_deg=45.0)
    idle.reset()
    for _ in range(200):
        idle.update(manifold, dt=0.004)
    assert idle.angle == 0.0 and idle.n_moved == 0, 'no axis in trouble -> no motion'
    print('✓ controller stays put when no axis nears a limit (why the '
          'unactivated form regresses on +Z)')

    ctl = NullSpaceController([obj], rate_deg_s=20.0, limit_deg=45.0)
    ctl.reset()
    for _ in range(1000):
        ctl.update(manifold, dt=0.004)
    q = manifold(ctl.angle)
    print('✓ converged to arm angle %+.1f°  J2 %.1f->%.1f  J6 %.1f->%.1f'
          % (ctl.angle, manifold(0)[1], q[1], manifold(0)[5], q[5]))
    assert abs(q[1]) < abs(manifold(0)[1]), 'J2 should move away from its limit'
    assert ctl.cost(q) < ctl.cost(manifold(0))

    # Rate limit: one frame moves at most rate*dt
    ctl.reset()
    a0 = ctl.angle
    ctl.update(manifold, dt=0.004)
    assert abs(ctl.angle - a0) <= 20.0 * 0.004 + 1e-12
    print('✓ rate limit: max %.3f° per frame' % (20.0 * 0.004))

    # Travel limit holds even when the objective keeps pulling
    ctl = NullSpaceController([obj], rate_deg_s=200.0, limit_deg=5.0)
    ctl.reset()
    for _ in range(500):
        ctl.update(manifold, dt=0.004)
    assert abs(ctl.angle) <= 5.0 + 1e-12
    print('✓ travel limit: angle clamped to ±5° despite sustained pull')

    # Infeasible angles are skipped, not crashed on
    ctl = NullSpaceController([obj], rate_deg_s=20.0, limit_deg=45.0)
    ctl.reset()
    ctl.update(lambda a: None, dt=0.004)
    assert ctl.angle == 0.0 and ctl.cost_now is None
    one_side = lambda a: None if a > 0 else manifold(a)   # noqa: E731
    ctl.reset()
    for _ in range(200):
        ctl.update(one_side, dt=0.004)
    assert ctl.angle <= 0.0
    print('✓ infeasible angles are skipped, not crashed on or drifted toward')

    # Operator bias composes with the automatic term
    ctl.reset()
    ctl.update(manifold, dt=0.004, bias_deg_s=100.0)
    assert ctl.angle > 0.0
    print('✓ joystick bias composes with the automatic term')

    # A probe the IK flattens (same q for every angle) must be reported, not
    # mistaken for "already optimal" -- this is what too-large a probe_deg does
    ctl.reset()
    for _ in range(10):
        ctl.update(lambda a: manifold(0.0), dt=0.004)
    assert ctl.n_moved == 0 and ctl.n_blind == 10
    print('✓ flattened probes counted as n_blind, not mistaken for optimal')

    # A second objective plugs in by weight alone
    class PreferStraightJ4(NullSpaceObjective):
        name = 'j4_straight'

        def cost(self, q):
            return (q[3] / 90.0) ** 2

    multi = NullSpaceController([obj, PreferStraightJ4(weight=0.0)],
                                rate_deg_s=20.0, limit_deg=45.0)
    multi.reset()
    assert math.isclose(multi.cost(manifold(0)), obj.cost(manifold(0)))
    print('✓ multi-objective: weighted sum, weight=0 opts an objective out')

    print('\nall passed')
