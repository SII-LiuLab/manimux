"""Time-based position tracking with a budget on the complete MIT PD request.

Adapted from tianji-control/algos/gripper_force_position.py (collection driver).
Inspired by TacCap v0.4.2's force-position design; this is a DM4310 policy,
not its SDK or firmware safety envelope. The budget bounds a prediction at
the latest feedback, not physical fingertip force or inter-sample torque.
"""
import math
from dataclasses import dataclass


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def gain(value, maximum):
    # Match the transport's unsigned, floor-quantized MIT gain fields.
    bits = int(value / maximum * 4095)
    return bits, bits / 4095 * maximum


@dataclass(frozen=True)
class Command:
    position: float
    velocity: float
    kp: float
    kd: float
    predicted_torque: float
    limited: bool


class ForcePosition:
    def __init__(self, position, *, kp, kd, speed, acceleration, hz, grasp_torque):
        kp_bits, self.kp = gain(kp, 500.)
        kd_bits, self.kd = gain(kd, 5.)
        # Mid-bin inputs reliably encode back to the gains used in the law.
        self.wire_kp = min(500., (kp_bits + .5) * 500. / 4095)
        self.wire_kd = min(5., (kd_bits + .5) * 5. / 4095)
        self.speed = speed
        self.acceleration = acceleration
        self.max_dt = 2. / hz
        # Reserve a full encoding step for position, velocity and zero torque.
        # Gains are already quantized above. Includes feedforward torque's
        # signed-zero encoding offset even though this policy requests zero.
        self.reserve = self.kp * 25/65535 + self.kd * 60/4095 + 20/4095
        self.budget = grasp_torque - self.reserve
        if self.kp <= 0 or self.budget <= 0:
            raise ValueError('force-position gain or torque budget below protocol resolution')
        self.reference = position
        self.velocity = 0.

    def step(self, target, position, velocity, dt):
        if not all(math.isfinite(v) for v in (target, position, velocity, dt)) or dt <= 0:
            raise ValueError('invalid force-position input or period')
        dt = min(dt, self.max_dt)
        # Remove accumulated reference error after a reversal or disturbance.
        # This window is derived from torque, rather than a fixed travel margin.
        lead = self.budget / self.kp
        previous = clamp(self.reference, position-lead, position+lead)
        if (target-position) * (previous-position) < 0:
            previous, self.velocity = position, 0.
        gap = target-previous
        speed = min(self.speed, math.sqrt(2*self.acceleration*abs(gap)))
        desired = math.copysign(speed, gap) if gap else 0.
        ramp_velocity = clamp(desired, self.velocity-self.acceleration*dt,
                              self.velocity+self.acceleration*dt)
        candidate = previous + ramp_velocity*dt
        candidate = clamp(candidate, min(previous, target), max(previous, target))

        # tau = kp*(r-pos) + kd*((r-prev)/dt-vel). Solve |tau|<=budget
        # for r, so damping and velocity feedforward share the same budget.
        a = self.kp + self.kd/dt
        b = self.kp*position + self.kd*previous/dt + self.kd*velocity
        lower = max((b-self.budget)/a, previous-self.speed*dt, min(previous, target))
        upper = min((b+self.budget)/a, previous+self.speed*dt, max(previous, target))
        if lower > upper:
            # A disturbance can make speed/target and torque bounds incompatible.
            # Use bounded motor-side damping for this frame and re-seed.
            kd_bits, kd = gain(min(self.kd, self.budget/(abs(velocity)+1e-9)), 5.)
            self.reference, self.velocity = position, 0.
            return Command(position, 0., 0., (kd_bits+.5)*5./4095,
                           -kd*velocity, True)
        reference = clamp(candidate, lower, upper)
        ramp_velocity = (reference-previous)/dt
        torque = self.kp*(reference-position) + self.kd*(ramp_velocity-velocity)
        self.reference, self.velocity = reference, ramp_velocity
        return Command(reference, ramp_velocity, self.wire_kp, self.wire_kd,
                       torque, abs(reference-candidate) > 1e-10)
