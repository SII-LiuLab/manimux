"""Causal lead for a stream of seven nominal Tianji position commands."""
from collections import deque
import math

import numpy as np


class JointPositionLead:
    max_ms = 100
    default_offset_deg = 2.0
    default_accel_cap_deg = 0.25
    accel_window_ns = 40_000_000
    velocity_slope_window_ns = 20_000_000

    @classmethod
    def validate_time(cls, lead_ms):
        if type(lead_ms) not in (float, int) or not 0 <= lead_ms <= cls.max_ms:
            raise ValueError(f"Position lead T must be between 0 and {cls.max_ms} ms")
        return float(lead_ms)

    @staticmethod
    def validate_offset(max_offset_deg):
        if type(max_offset_deg) not in (float, int) or not 0 <= max_offset_deg <= 10:
            raise ValueError("Position lead offset limit must be between 0 and 10 degrees")
        return float(max_offset_deg)

    @staticmethod
    def validate_ka(ka_s2):
        if type(ka_s2) not in (float, int) or not math.isfinite(ka_s2) or ka_s2 < 0:
            raise ValueError("Acceleration gain Ka must be finite and nonnegative (s²)")
        return float(ka_s2)

    def __init__(self, lead_ms, period_ns, max_offset_deg=default_offset_deg, *,
                 ka_s2=0, accel_cap_deg=default_accel_cap_deg, max_velocity_rad_s=None):
        self.lead_s = self.validate_time(lead_ms) / 1000
        self.max_offset_rad = np.deg2rad(self.validate_offset(max_offset_deg))
        self.ka_s2 = self.validate_ka(ka_s2)
        self.accel_cap_rad = np.deg2rad(self.validate_offset(accel_cap_deg))
        self.accel_history = deque()
        self.accel_offset = np.zeros(7)
        self.period_ns = period_ns
        self.max_recovery_step = (None if max_velocity_rad_s is None else
                                  np.asarray(max_velocity_rad_s) * (period_ns * 1e-9))
        self.recovery_start_ns = None
        self.recovery_offset = np.zeros(7)
        self.last_offset = np.zeros(7)
        # Use a short command history so high-rate timing jitter is not amplified.
        intervals = max(1, math.ceil(self.velocity_slope_window_ns / period_ns))
        self.history = deque(maxlen=intervals)
        self.velocity_history = deque(maxlen=intervals + 1)
        self.acceleration_rad_s2 = np.zeros(7)
        # Least-squares slope on the same nominal tick grid as the executor velocity.
        seconds = (np.arange(intervals + 1) - intervals / 2) * (period_ns * 1e-9)
        self.slope_weights = seconds / (seconds @ seconds)

    @property
    def enabled(self):
        return self.max_offset_rad > 0 and (self.lead_s > 0 or (self.ka_s2 > 0 and self.accel_cap_rad > 0))

    def _acceleration_offset(self, nominal, now_ns):
        history = self.accel_history
        history.append((now_ns, nominal.copy()))
        # Keep the first sample at/before the window edge, even with tick jitter.
        cutoff = now_ns - self.accel_window_ns
        while len(history) > 1 and history[1][0] <= cutoff:
            history.popleft()
        if len(history) < 3 or history[0][0] > cutoff:
            self.accel_offset.fill(0)
            return
        span_ns = now_ns - history[0][0]
        t = np.array([(ns - now_ns) / span_ns for ns, _ in history])
        design = np.column_stack((np.ones_like(t), t, t*t))
        positions = np.stack([q for _, q in history]) - nominal
        coefficients = np.linalg.lstsq(design, positions, rcond=None)[0]
        acceleration = 2 * coefficients[2] / (span_ns * 1e-9)**2
        self.accel_offset = np.clip(self.ka_s2 * acceleration, -self.accel_cap_rad, self.accel_cap_rad)

    def apply(self, nominal, now_ns, *, last_issued=None, velocity_rad_s=None, missed_tick=False):
        if not self.enabled and velocity_rad_s is None:
            return nominal
        result = nominal
        if self.history:
            previous_ns, previous_nominal = self.history[-1]
            if last_issued is None:
                last_issued = previous_nominal + self.last_offset
            elapsed = now_ns - previous_ns
            if elapsed <= 0:
                raise ValueError("Position lead requires increasing command timestamps")
            if missed_tick or elapsed > 1.5 * self.period_ns:
                if self.enabled:
                    self.recovery_start_ns = now_ns
                    self.recovery_offset = last_issued - previous_nominal
                self.history.clear()
                self.accel_history.clear()
                self.velocity_history.clear()
                self.accel_offset.fill(0)
        if velocity_rad_s is not None:
            velocity = np.asarray(velocity_rad_s, dtype=float)
            if velocity.shape != (7,) or not np.isfinite(velocity).all():
                raise ValueError("Expected seven finite nominal joint velocities in rad/s")
            self.velocity_history.append(velocity.copy())
            if len(self.velocity_history) == self.velocity_history.maxlen:
                self.acceleration_rad_s2 = self.slope_weights @ np.stack(self.velocity_history)
            else:
                self.acceleration_rad_s2.fill(0)
            self.accel_offset = np.clip(self.ka_s2 * self.acceleration_rad_s2,
                                        -self.accel_cap_rad, self.accel_cap_rad)
        else:
            velocity = None
        if velocity is None and self.ka_s2 and self.accel_cap_rad:
            self._acceleration_offset(nominal, now_ns)
        if velocity is None and self.history:
            first_ns, first_target = self.history[0]
            velocity = (nominal - first_target) / ((now_ns - first_ns) * 1e-9)
        if self.enabled and velocity is not None:
            offset = self.lead_s * velocity + self.accel_offset
            offset = np.clip(offset, -self.max_offset_rad, self.max_offset_rad)
            result = nominal + offset
        if self.recovery_start_ns is not None:
            # Carry the issued offset across the missed tick, then hand over
            # to fresh derivatives as their longest estimation window fills.
            weight = min(1., (now_ns - self.recovery_start_ns) / self.accel_window_ns)
            offset = (1 - weight) * self.recovery_offset + weight * (result - nominal)
            result = nominal + np.clip(offset, -self.max_offset_rad, self.max_offset_rad)
            step = result - last_issued
            if self.max_recovery_step is not None and np.any(np.abs(step) > self.max_recovery_step):
                result = last_issued + np.clip(step, -self.max_recovery_step, self.max_recovery_step)
                # An infeasible nominal move must still go through the usual rate check.
                result = nominal + np.clip(result - nominal, -self.max_offset_rad, self.max_offset_rad)
            elif weight == 1.:
                self.recovery_start_ns = None
        self.last_offset = result - nominal
        self.history.append((now_ns, nominal.copy()))
        return result


class DiffIKReferenceLead:
    """Shape dense arm references before the executor, on the solved IK clock."""

    def __init__(self, options, solver, lower, upper):
        self.lead_s = JointPositionLead.validate_time(options["lead_ms"]) * 1e-3
        self.ka = JointPositionLead.validate_ka(options["ka_s2"])
        self.accel_cap = np.radians(JointPositionLead.validate_offset(options["accel_cap_deg"]))
        self.cap = np.radians(JointPositionLead.validate_offset(options["max_offset_deg"]))
        ik_lo, ik_hi, self.wrist, self.wrist_bound = solver.protected_position_bounds()
        self.lower = np.maximum(ik_lo, lower)
        self.upper = np.minimum(ik_hi, upper)

    def shape(self, solved, dt_ns, positions, reference):
        # The active segment of the piecewise linear IK path supplies dq/dt.
        # All derivatives stay within one solve, including at chunk replacement.
        dt = dt_ns * 1e-9
        velocity = np.diff(solved[:, :7], axis=0) / dt
        acceleration = np.zeros_like(velocity)
        intervals = max(1, math.ceil(JointPositionLead.velocity_slope_window_ns / dt_ns))
        if len(velocity) > intervals:
            times = (np.arange(intervals + 1) - intervals / 2) * dt
            windows = np.lib.stride_tricks.sliding_window_view(velocity, intervals + 1, axis=0)
            acceleration[intervals:] = windows @ (times / (times @ times))
        index = np.floor(positions).astype(int)
        moving = (index >= 0) & (index < len(velocity))
        offset = np.zeros_like(reference[:, :7])
        accel_term = np.clip(self.ka * acceleration[index[moving]], -self.accel_cap, self.accel_cap)
        offset[moving] = np.clip(self.lead_s * velocity[index[moving]] + accel_term,
                                 -self.cap, self.cap)
        nominal = reference[:, :7]
        lo = np.maximum(self.lower, nominal - self.cap)
        hi = np.minimum(self.upper, nominal + self.cap)
        if np.any(lo > hi):
            raise ValueError("No protected pre-smooth target inside the lead cap")
        target = np.clip(nominal + offset, lo, hi)
        # Move only the wrist coordinates, within their correction box, toward
        # the feasible point nearest zero until every wrist halfplane is met.
        outside = np.any(target @ self.wrist.T > self.wrist_bound, axis=1)
        if np.any(outside):
            anchor = np.clip(0., lo[outside], hi[outside])
            budget = self.wrist_bound - anchor @ self.wrist.T
            if np.any(budget < -1e-12):
                raise ValueError("No protected wrist target inside the pre-smooth lead cap")
            rise = (target[outside] - anchor) @ self.wrist.T
            ratios = np.divide(budget, rise, out=np.ones_like(rise), where=rise > 0)
            weight = np.clip(ratios.min(axis=1), 0, 1)[:, None]
            target[outside, 5:7] = anchor[:, 5:7] + weight * (target[outside, 5:7] - anchor[:, 5:7])
        result = reference.copy()
        result[:, :7] = target
        return result

    def validate_output(self, joints):
        if (np.any(joints < self.lower - 1e-12) or np.any(joints > self.upper + 1e-12)
                or np.any(self.wrist @ joints > self.wrist_bound + 1e-12)):
            raise ValueError("Smoothed Tianji command exceeds protected joint or wrist limits")


class RolloutArmLead:
    """Protect the compensated stream, with history independent of nominal execution."""

    def __init__(self, options, period_ns, solver, *, lower, upper, vmax, amax, step_ik):
        self.options = dict(options)
        self.period_ns = period_ns
        self.dt = period_ns * 1e-9
        self.step_ik = step_ik
        self.solver = solver
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.vmax = np.asarray(vmax)
        self.amax = np.broadcast_to(amax, (7,))
        if step_ik:
            self.vmax = np.minimum(self.vmax, np.radians(solver.vmax))
            self.amax = np.minimum(self.amax, np.radians(solver.amax))
        # Reserve a feasible deceleration before every joint or wrist boundary.
        self.horizon = max(self.dt, float(np.max(self.vmax / self.amax)))
        if step_ik:
            self.horizon = max(self.horizon, solver.braking_time(self.dt))
        self.lower_ik, self.upper_ik, self.wrist, self.wrist_bound = solver.protected_position_bounds()
        self.lead = self._new_lead()
        self.tracking = False

    def _new_lead(self):
        return JointPositionLead(period_ns=self.period_ns, max_velocity_rad_s=self.vmax,
                                 **self.options)

    def reset(self, measured):
        self.previous = np.array(measured[:7], dtype=float)
        self.last_nominal = self.previous.copy()
        self.velocity = np.zeros(7)
        if self.tracking:
            self.lead = self._new_lead()
        self.tracking = False

    @staticmethod
    def _inverse_braking_sum(budget, quantum):
        # Invert x + max(0, x-b) + max(0, x-2b) + ... for x >= 0.
        n = np.floor((np.sqrt(1 + 8 * budget / quantum) - 1) / 2)
        return (budget + n * (n + 1) * quantum / 2) / (n + 1)

    def _offset_speed_limit(self, remaining, nominal_speed):
        """Maximum signed speed that can keep the offset cap if the target brakes.

        With acceleration quantum b=a*dt, a speed x above -vmax travels
        dt*sum(max(0,x-k*b), k>=1) beyond that speed during future braking.
        Reserve the difference between issued and nominal braking travel.
        """
        b = self.amax * self.dt
        finite = np.isfinite(b)
        if not np.any(finite):
            return remaining / self.dt
        quantum = b[finite]
        x = np.clip(nominal_speed[finite], -self.vmax[finite], self.vmax[finite]) + self.vmax[finite]
        n = np.floor(x / quantum)
        reserve = n * (x - (n + 1) * quantum / 2)
        step_budget = remaining[finite] / self.dt
        reversal_budget = np.maximum(0, step_budget + self.vmax[finite] + reserve)
        reversal = self._inverse_braking_sum(reversal_budget, quantum) - self.vmax[finite]
        # A chunk may end on any tick. Reserve enough distance to stop against
        # the now-fixed nominal target without drifting beyond the offset cap.
        stop = self._inverse_braking_sum(np.maximum(0, step_budget), quantum)
        speed_limit = remaining / self.dt
        speed_limit[finite] = np.minimum.reduce((step_budget, reversal, stop))
        return speed_limit

    def apply(self, nominal, now_ns, reference_velocity):
        prev = self.previous
        if self.step_ik and self.solver.elbow_side is None:
            self.solver.ensure_elbow_branch(prev)
        if self.step_ik:
            # A lifecycle reanchor can select the other J4 branch.
            self.lower_ik, self.upper_ik, _, _ = self.solver.protected_position_bounds()
        lower = np.maximum(self.lower, self.lower_ik)
        upper = np.minimum(self.upper, self.upper_ik)
        bounded = np.clip(nominal, lower, upper)
        wrist_use = np.max((self.wrist @ bounded) / self.wrist_bound)
        if wrist_use > 1:
            bounded[5:7] /= wrist_use
        if reference_velocity is None:
            if self.tracking:
                self.lead = self._new_lead()
            self.tracking = False
            if self.step_ik:
                desired = self.solver.brake(prev, self.velocity, self.dt)
            else:
                distance = bounded - prev
                speed = np.abs(distance) / self.dt
                finite = np.isfinite(self.amax)
                speed[finite] = self._inverse_braking_sum(speed[finite], self.amax[finite] * self.dt)
                desired = prev + np.sign(distance) * speed * self.dt
        else:
            self.tracking = True
            desired = self.lead.apply(nominal, now_ns, last_issued=prev,
                                      velocity_rad_s=reference_velocity)

        # Restrict the next point and its stopping endpoint using the actually
        # issued last point/velocity. The stopping box is centered at origin.
        fraction = self.dt / self.horizon
        origin = (1 - fraction) * prev
        lo = np.maximum(origin + fraction * lower,
                        prev + self.dt * np.maximum(-self.vmax, self.velocity - self.dt * self.amax))
        hi = np.minimum(origin + fraction * upper,
                        prev + self.dt * np.minimum(self.vmax, self.velocity + self.dt * self.amax))
        lo, hi = np.maximum(lo, lower), np.minimum(hi, upper)
        if self.tracking:
            displacement = bounded - prev
            v = (bounded - self.last_nominal) / self.dt
            cap = self.lead.max_offset_rad
            hi = np.minimum(hi, prev + self.dt * self._offset_speed_limit(cap + displacement, v))
            lo = np.maximum(lo, prev - self.dt * self._offset_speed_limit(cap - displacement, -v))
        else:
            lo = np.maximum(lo, bounded - self.lead.max_offset_rad)
            hi = np.minimum(hi, bounded + self.lead.max_offset_rad)
        if np.any(lo > hi + 1e-12):
            raise ValueError("No feasible Tianji lead command within position, rate and offset bounds")
        issued = np.clip(desired, lo, hi)
        # The centered wrist diamond is monotone in |J6|, |J7|. The point
        # closest to its center in the permitted box is a feasible line anchor.
        cap = fraction * self.wrist_bound
        rates = self.wrist @ (issued - origin)
        if np.any(rates > cap):
            base = np.clip(origin, lo, hi)
            start = self.wrist @ (base - origin)
            if np.any(start > cap + 1e-12):
                raise ValueError("No feasible Tianji lead command inside the protected wrist region")
            rising = rates > cap
            scale = np.min((cap[rising] - start[rising]) /
                           np.maximum(rates[rising] - start[rising], 1e-15))
            issued[5:7] = base[5:7] + np.clip(scale, 0, 1) * (issued[5:7] - base[5:7])
        if np.any(self.wrist @ issued > self.wrist_bound + 1e-12):
            raise ValueError("Tianji lead command outside the protected wrist region")
        if self.step_ik:
            reason = self.solver.command_violation(prev, issued, self.velocity, self.dt)
            if reason:
                raise ValueError(f"Unsafe compensated step IK command: {reason}")
        self.velocity = (issued - prev) / self.dt
        self.previous = issued.copy()
        self.last_nominal = bounded.copy()
        return issued
