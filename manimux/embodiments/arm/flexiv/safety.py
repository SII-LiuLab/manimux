"""Joint-space counterpart of collection's bounded differential-IK envelope.

Uses SDK bounds and measured feedback; does not solve IK or change posture in
the nullspace. Units are radians and seconds. No hardware dependencies.
"""

import numpy as np


class JointSafety:
    def __init__(self, *, slowdown_zone, tracking_limit, max_dt_s, tracking_timeout_s):
        values = [slowdown_zone, tracking_limit, max_dt_s, tracking_timeout_s]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Flexiv joint_safety values must be finite and positive")
        self.slowdown_zone = float(slowdown_zone)
        self.tracking_limit = float(tracking_limit)
        self.max_dt_s = float(max_dt_s)
        self.tracking_timeout_s = float(tracking_timeout_s)
        self.reset()

    def reset(self):
        self.previous = None
        self.sent_ns = None
        self.tracking_since_ns = None

    def target(self, requested, measured, lower, upper, velocity, now_ns):
        if self.previous is None:
            # Collection seeds its native generator from bounded measured q.
            return np.clip(measured, lower, upper)
        q = self.previous
        elapsed = (now_ns - self.sent_ns) / 1e9
        if elapsed < 0:
            raise RuntimeError("Flexiv command clock regressed")
        dt = min(elapsed, self.max_dt_s)
        tracking_bad = np.any(np.abs(q - measured) > self.tracking_limit)
        if not tracking_bad and dt > 0:
            # Same directional velocity dampers as FlexivDiffIK.solve().
            low_distance = np.minimum(q, measured) - lower
            high_distance = upper - np.maximum(q, measured)
            low_speed = -velocity * np.clip(low_distance / self.slowdown_zone, 0, 1)
            high_speed = velocity * np.clip(high_distance / self.slowdown_zone, 0, 1)
            low = np.maximum(q + low_speed * dt, np.maximum(lower, measured - self.tracking_limit))
            high = np.minimum(q + high_speed * dt, np.minimum(upper, measured + self.tracking_limit))
            tracking_bad = bool(np.any(low > high))
            if not tracking_bad:
                self.tracking_since_ns = None
                return np.clip(requested, low, high)
        if tracking_bad:
            if self.tracking_since_ns is None:
                self.tracking_since_ns = now_ns
            if now_ns - self.tracking_since_ns >= self.tracking_timeout_s * 1e9:
                raise RuntimeError("Flexiv joint tracking error persisted beyond joint_safety timeout")
        else:
            self.tracking_since_ns = None
        # Hold the complete arm when no feasible step exists. The shared runtime
        # stops on timeout; unlike teleop there is no clutch to disengage/rebase.
        return q.copy()

    def commit(self, submitted, now_ns):
        """Advance the reference only after successful SDK submission."""
        self.previous = submitted.copy()
        self.sent_ns = now_ns

    def metadata(self):
        return {key: getattr(self, key) for key in (
            "slowdown_zone", "tracking_limit", "max_dt_s", "tracking_timeout_s",
        )}
