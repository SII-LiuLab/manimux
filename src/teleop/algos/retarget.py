#!/usr/bin/env python3
"""Retargeting — maps controller pose to a robot end-effector target pose.

Relative-mapping + clutch design (not absolute mapping): clutch-press
latches (controller pose, robot pose); thereafter the target tracks the
controller's motion relative to that latch, scaled.

`delta_frame` picks WHICH FRAME that relative motion is expressed in, and
the two choices are not interchangeable:

* `'base'` (default, live PICO teleop) -- the delta is a *spatial* one,
  taken in the source's world frame and conjugated into the arm's base
  frame::

      dR = R_xr(t) . R_xr(0)^T          p_des = p_rob0 + R_map . dp_xr . s
      R_des = (R_map . dR . R_map^T) . R_rob0                (LEFT-multiply)

  Semantics: "the hand turned N degrees about a world axis, so the tool
  turns N degrees about the corresponding base axis". Coordinate
  conversion is the two-stage pipeline XR raw -> right-handed -> base
  frame, each stage independently verifiable, and `config.AXIS_MAP` is
  what makes the second stage meaningful. A human watches the tool and
  closes the loop, so a world-referenced delta is what feels right.

  Consequence: a tool frame's ROTATION cancels out of every commanded
  pose (it is applied at latch and undone at emit, with a left-multiply
  in between), so only `ToolFrame.t` affects this path.

* `'ee'` (recorded UMI replay) -- the delta is a *body* one, taken in the
  recorded gripper's own EE frame and applied in the robot gripper's own
  EE frame::

      dT = T_rec(0)^-1 . T_rec(t)       p_des = p_rob0 + R_rob0 . dp_ee . s
      R_des = R_rob0 . dR_ee                                (RIGHT-multiply)

  No axis map at all: the delta already lives in the gripper's own frame,
  which the recorded EE and the robot EE share by construction, so there
  is no world->base conversion left to do (`self.R_map` is None here and
  using it is a bug).

  This is the representation the policy pipeline trains on
  (`lerobot_image_dataset.convert_action_to_relative`: `T_cur^-1 @ T_tgt`,
  `output_pose_frame: current_ee`) and the one deployment executes, so a
  replay run this way rehearses what will actually run.

  Consequence: with the tip latched via `fk_tip`, the commanded flange
  works out to `T_f(0) . X . dT . X^-1` where `X` is the tool frame -- a
  CONJUGATION, so `ToolFrame.R` (`kine_offset`'s a/b/c) no longer cancels.
  This path is only as correct as `configs/tool/<name>.yaml`'s rotation.

See docs/algos.md for why absolute mapping is unsafe, the coordinate-frame
calibration procedure, and the rate-limiter design.
"""
import math

import numpy as np

# ---------- math primitives ----------


def quat_to_mat(q, order='xyzw'):
    """Quaternion -> 3x3 rotation matrix."""
    if order == 'wxyz':
        w, x, y, z = q
    else:
        x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_xyzabc(R, p):
    """3x3 rotation + position -> Marvin's XYZABC (Euler angles, degrees).

    Assumes fixed-axis X->Y->Z Euler angles — verify against
    kine.mat4x4_to_xyzabc() via verify_euler_convention() before trusting.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-9:
        a = math.atan2(R[2, 1], R[2, 2])
        b = math.atan2(-R[2, 0], sy)
        c = math.atan2(R[1, 0], R[0, 0])
    else:                                   # gimbal lock
        a = math.atan2(-R[1, 2], R[1, 1])
        b = math.atan2(-R[2, 0], sy)
        c = 0.0
    return [p[0], p[1], p[2],
            math.degrees(a), math.degrees(b), math.degrees(c)]


def build_axis_matrix(axis_map):
    """Build the 3x3 matrix mapping XR basis vectors to robot-base axes.

    Returns R such that v_base = R @ v_xr_rh. Keys are xr_x/xr_y/xr_z (not
    semantic names like right/up/forward) and det is checked to be +1, not
    -1 (a reflection) — see docs/algos.md for why both choices matter.
    """
    col_of = {'xr_x': 0, 'xr_y': 1, 'xr_z': 2}
    row_of = {'X': 0, 'Y': 1, 'Z': 2}
    R = np.zeros((3, 3))
    for key, spec in axis_map.items():
        if key not in col_of:
            raise ValueError('AXIS_MAP key must be xr_x/xr_y/xr_z, got %r.\n'
                             '(legacy xr_right/xr_up/xr_forward keys are '
                             'deprecated: PICO +Z is "backward", semantic '
                             'names mislead)' % key)
        s = spec.strip()
        sign = -1.0 if s[0] == '-' else 1.0
        axis = s[-1].upper()
        if axis not in row_of:
            raise ValueError('AXIS_MAP value must look like +X/-Y/+Z, got %r' % spec)
        R[row_of[axis], col_of[key]] = sign
    if len(axis_map) != 3:
        raise ValueError('AXIS_MAP must have exactly 3 entries')
    # must be a signed permutation: exactly one nonzero per row and column
    if not (np.count_nonzero(R, axis=0) == 1).all() or \
       not (np.count_nonzero(R, axis=1) == 1).all():
        raise ValueError('AXIS_MAP has a duplicated axis:\n%s' % R)
    det = np.linalg.det(R)
    if det < 0:
        raise ValueError(
            'AXIS_MAP determinant is %.0f (a reflection, not a rotation).\n'
            'This mirrors orientation while position still looks correct — '
            'very hard to catch by eye. Check for a flipped axis sign.' % det)
    return R


def rot_angle_deg(R):
    """Rotation angle (degrees) of a rotation matrix."""
    c = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def scale_rotation(R, f):
    """Scale rotation R to f times its angle about the same axis (f in [0,1])."""
    if f >= 1.0:
        return R
    ang = math.radians(rot_angle_deg(R))
    if ang < 1e-9:
        return np.eye(3)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return R                     # angle ~180°, axis degenerate (rare)
    axis = axis / n
    a = ang * f
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def flip_handedness(p, R, axis='z'):
    """Left-handed -> right-handed. axis=None: data is already right-handed
    (measured true for PICO 4U, see docs/algos.md), skip unchanged.
    """
    if axis is None:
        return np.asarray(p, dtype=float), np.asarray(R, dtype=float)
    i = {'x': 0, 'y': 1, 'z': 2}[axis]
    F = np.eye(3)
    F[i, i] = -1.0
    return F @ np.asarray(p, dtype=float), F @ R @ F


# ---------- clutch state machine ----------

DISENGAGED = 'disengaged'
ENGAGED = 'engaged'


class Retargeter:
    """Retargets one controller to one arm. Instantiate one per arm.

    Typical per-frame usage:
        r.update(frame, hand='right', q_now=..., fk=ik.fk_xyzabc, nsp=ik.nsp_dir)
        if r.state == ENGAGED:
            target = r.target_xyzabc     # feed to ArmIK.solve_xyzabc
    """

    def __init__(self, cfg, arm, lowpass=None, nullspace=None, axis_map=None,
                 tool=None, delta_frame='base'):
        """:param arm: 'A' or 'B' — axis mapping is per-arm, see config.AXIS_MAP.
        :param nullspace: optional nullspace.NullSpaceController; when given,
        it drives the arm angle (joystick becomes a bias term), else the
        joystick integrates the angle directly.
        :param axis_map: per-arm table to use instead of config.AXIS_MAP —
        the input device decides the source frame, not the robot, so a
        non-PICO source passes its own here rather than mutating config.
        Only meaningful for delta_frame='base'; ignored (and R_map left
        None) for 'ee', which has no world->base mapping to do.
        :param tool: optional algos.tool_frame.ToolFrame. None (default) =
        retarget the flange, i.e. every existing caller's behaviour, bit for
        bit. With one, the *tip* is what latches, what follows and what the
        Cartesian limiter bounds; only the final target_xyzabc is converted
        back to the flange for the IK. A hand-held demo records its
        fingertips, so following it at the flange rotates the real fingertips
        through an arc the demonstrator never made — see
        docs/algos.md#tool_framepy.
        :param delta_frame: 'base' (default) takes the relative motion in the
        source's world frame and conjugates it into the arm's base frame via
        axis_map — live PICO teleop. 'ee' takes it in the recorded gripper's
        own EE frame and applies it in the robot gripper's own EE frame,
        which is what the policy pipeline trains on and what deployment
        executes; axis_map is then unused. Full derivation and the
        consequences for ToolFrame in this module's docstring."""
        self.cfg = cfg
        self.arm = arm
        self.tool = tool
        if delta_frame not in ('base', 'ee'):
            raise ValueError("delta_frame must be 'base' or 'ee', got %r"
                             % (delta_frame,))
        self.delta_frame = delta_frame
        if delta_frame == 'ee':
            # No world->base conversion exists on this path: the delta is
            # already in the gripper's own frame. Left as None so an
            # accidental use raises instead of silently mapping twice.
            self.R_map = None
        else:
            table = cfg.AXIS_MAP if axis_map is None else axis_map
            if arm not in table:
                raise ValueError('axis map has no mapping for arm %s' % arm)
            self.R_map = build_axis_matrix(table[arm])
        self.scale = cfg.SCALE
        self.follow_rotation = cfg.FOLLOW_ROTATION
        self.lowpass = lowpass
        self.nullspace = nullspace

        self.state = DISENGAGED
        # latched at clutch-press
        self._p_xr0 = None
        self._R_xr0 = None
        self._p_rob0 = None
        self._R_rob0 = None
        self._q0 = None
        self._zsp_dir = None
        self.arm_angle = 0.0
        # rate-limited "current commanded pose" (not the desired pose)
        self._cmd_p = None
        self._cmd_R = None
        self.cart_lag_mm = 0.0
        self.rot_lag_deg = 0.0
        self.n_slewed = 0

        self.target_xyzabc = None
        self.target_tip_xyzabc = None
        self.engage_count = 0
        # latched after a fault; must release clutch before re-engaging.
        # See docs/algos.md for what happens without this latch.
        self._fault_latched = False
        self.fault_count = 0

    # ----- clutch -----

    def _clutch_pressed(self, frame, hand):
        v = frame.button(hand, self.cfg.CLUTCH_BUTTON, 0.0)
        try:
            return float(v) >= self.cfg.CLUTCH_THRESHOLD
        except (TypeError, ValueError):
            return bool(v)

    def fk_tip(self, fk_xyzabc, q):
        """FK, expressed at the tool tip when a ToolFrame is attached.

        Everything the retargeter latches, compares and rate-limits goes
        through here, so the whole clutch/limiter layer works in the frame
        that touches the object rather than at the flange."""
        x = fk_xyzabc(q)
        return x if self.tool is None else self.tool.tip_from_flange(x)

    def engage(self, p_xr, R_xr, q_now, fk_xyzabc, nsp_dir):
        """Latch current controller + robot state and start following."""
        self._p_xr0, self._R_xr0 = p_xr, R_xr
        x = self.fk_tip(fk_xyzabc, q_now)
        self._p_rob0 = np.array(x[:3], dtype=float)
        self._R_rob0 = xyzabc_to_rot(x)
        self._q0 = list(q_now)
        self._zsp_dir = nsp_dir(q_now)      # fixed for the whole engagement
        self.arm_angle = 0.0
        if self.nullspace is not None:
            self.nullspace.reset()
        # start the rate limiter from the robot's actual current pose so
        # clutch-press causes zero jump — see docs/algos.md
        self._cmd_p = np.array(self._p_rob0, dtype=float)
        self._cmd_R = np.array(self._R_rob0, dtype=float)
        self.cart_lag_mm = 0.0
        self.state = ENGAGED
        self.engage_count += 1
        if self.lowpass:
            self.lowpass.reset(p_xr)

    def disengage(self, fault=False):
        """Stop following. fault=True latches until the clutch is released
        (a plain release does not latch)."""
        self.state = DISENGAGED
        self.target_xyzabc = None
        self.target_tip_xyzabc = None
        if fault and not self._fault_latched:
            self._fault_latched = True
            self.fault_count += 1

    # ----- per frame -----

    def update(self, frame, hand, q_now, fk_xyzabc, nsp_dir, dt=0.004,
               solve=None):
        """Returns True iff self.target_xyzabc is valid this frame.

        :param solve: fn(target_xyzabc, q_now, zsp_dir, arm_angle) -> joints
                      or None. Only needed with a nullspace controller
                      attached, to probe candidate arm angles.
        """
        pose = frame.pose(hand)
        if pose is None:
            self.disengage()
            return False

        p_raw = np.array(pose[:3], dtype=float) * self.cfg.XR_POS_TO_MM
        R_raw = quat_to_mat(pose[3:], order=self.cfg.XR_QUAT_ORDER)
        p_rh, R_rh = flip_handedness(p_raw, R_raw,
                                     self.cfg.XR_HANDEDNESS_FLIP_AXIS)
        if self.lowpass:
            p_rh = np.array(self.lowpass(p_rh, dt))

        pressed = self._clutch_pressed(frame, hand)

        # fault latch: holding the clutch down does not re-engage; must release first
        if self._fault_latched:
            if not pressed:
                self._fault_latched = False
            else:
                if self.state == ENGAGED:
                    self.disengage()
                return False

        if pressed and self.state == DISENGAGED:
            self.engage(p_rh, R_rh, q_now, fk_xyzabc, nsp_dir)
        elif not pressed and self.state == ENGAGED:
            self.disengage()
            return False
        if self.state != ENGAGED:
            return False

        if self.delta_frame == 'ee':
            # Body-frame delta: dT = T_rec(latch)^-1 . T_rec(t), applied as
            # T_rob(latch) . dT. Both sides are the gripper's own EE frame
            # (the latch went through fk_tip, the recording is a fingertip
            # midpoint), so there is nothing to axis-map -- this is the same
            # representation the policy trains on and deployment executes.
            # See the module docstring.
            dp_ee = self._R_xr0.T @ (p_rh - self._p_xr0)
            p_des = self._p_rob0 + self._R_rob0 @ dp_ee * self.scale
            if self.follow_rotation:
                dR_ee = self._R_xr0.T @ R_rh
                R_des = self._R_rob0 @ dR_ee        # RIGHT-multiply
            else:
                R_des = self._R_rob0
        else:
            # translation: controller displacement from latch, mapped to base frame, scaled
            dp_xr = p_rh - self._p_xr0
            p_des = self._p_rob0 + self.R_map @ dp_xr * self.scale

            # orientation: controller relative rotation, conjugated to base frame
            if self.follow_rotation:
                dR_xr = R_rh @ self._R_xr0.T
                dR_base = self.R_map @ dR_xr @ self.R_map.T
                R_des = dR_base @ self._R_rob0      # LEFT-multiply
            else:
                R_des = self._R_rob0

        # Cartesian rate limiter: command tracks the desired pose at a
        # bounded speed instead of jumping to it. Origin MUST be the robot's
        # actual current pose (FK of q_now), not an accumulated command —
        # see docs/algos.md for the cascading-rejection failure mode this
        # avoids.
        x_now = self.fk_tip(fk_xyzabc, q_now)
        p_now = np.array(x_now[:3], dtype=float)
        R_now = xyzabc_to_rot(x_now)

        # dt<=0 (first frame / timing glitch) => budget is 0, i.e. no
        # motion — written as an explicit branch, not a chained comparison
        # (see docs/algos.md for the bug that pattern caused).
        dt_eff = min(max(dt, 0.0), self.cfg.MAX_STEP_DT_S)
        max_mm = self.cfg.CART_MAX_SPEED_MM_S * dt_eff
        dp = p_des - p_now
        dist = float(np.linalg.norm(dp))
        self.cart_lag_mm = dist                 # observability: how far behind
        if max_mm <= 0.0:
            self._cmd_p = np.array(p_now, dtype=float)
        elif dist > max_mm:
            self._cmd_p = p_now + dp * (max_mm / dist)
            self.n_slewed += 1
        else:
            self._cmd_p = p_des

        if self.follow_rotation:
            dR = R_des @ R_now.T
            ang = rot_angle_deg(dR)
            self.rot_lag_deg = ang              # observability, same as cart_lag_mm
            max_deg = self.cfg.CART_MAX_ROT_DEG_S * dt_eff
            if max_deg <= 0.0:
                self._cmd_R = np.array(R_now, dtype=float)
            elif ang > max_deg:
                self._cmd_R = scale_rotation(dR, max_deg / ang) @ R_now
                self.n_slewed += 1
            else:
                self._cmd_R = R_des
        else:
            self._cmd_R = R_des
            self.rot_lag_deg = 0.0

        # The command is decided at the tip; the IK targets the flange, so the
        # tool transform is undone exactly once, here, at the very end.
        self.target_tip_xyzabc = mat_to_xyzabc(self._cmd_R, self._cmd_p)
        self.target_xyzabc = (self.target_tip_xyzabc if self.tool is None
                              else self.tool.flange_from_tip(
                                  self.target_tip_xyzabc))

        # arm angle must be decided after the target pose is finalized:
        # the nullspace controller solves against it to probe candidates.
        bias = 0.0
        if self.cfg.ARM_ANGLE_RATE:
            ax = frame.button(hand, self.cfg.ARM_ANGLE_AXIS, 0.0) or 0.0
            if abs(ax) > 0.15:              # joystick deadband
                bias = ax * self.cfg.ARM_ANGLE_RATE
        if self.nullspace is not None and solve is not None:
            self.arm_angle = self.nullspace.update(
                lambda a: solve(self.target_xyzabc, q_now, self._zsp_dir, a),
                dt, bias_deg_s=bias)
        else:
            self.arm_angle = max(-self.cfg.ARM_ANGLE_LIMIT,
                                 min(self.cfg.ARM_ANGLE_LIMIT,
                                     self.arm_angle + bias * dt))
        return True

    @property
    def zsp_dir(self):
        return self._zsp_dir

    @property
    def q_latch(self):
        """Configuration at clutch-press — used as the IK's initial reference."""
        return self._q0


def xyzabc_to_rot(x):
    """XYZABC -> 3x3 rotation (inverse of mat_to_xyzabc); Euler convention
    must match mat_to_xyzabc."""
    a, b, c = (math.radians(v) for v in x[3:6])
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def verify_euler_convention(kine, samples=200, tol_deg=1e-3):
    """Verify this module's Euler convention matches the SDK's
    mat4x4_to_xyzabc. Needs an initial_kine'd Marvin_Kine, so this is only
    called manually where the SDK is available (see replay.py)."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(samples):
        q = rng.normal(size=4)
        R = quat_to_mat(q / np.linalg.norm(q))
        p = rng.uniform(-500, 500, 3)
        mat = [[R[0, 0], R[0, 1], R[0, 2], p[0]],
               [R[1, 0], R[1, 1], R[1, 2], p[1]],
               [R[2, 0], R[2, 1], R[2, 2], p[2]],
               [0, 0, 0, 1]]
        sdk = kine.mat4x4_to_xyzabc(pose_mat=mat)
        mine = mat_to_xyzabc(R, p)
        d = max(abs((a - b + 180) % 360 - 180) for a, b in zip(sdk[3:], mine[3:]))
        worst = max(worst, d)
    return worst <= tol_deg, worst


if __name__ == '__main__':
    import config

    for arm, m in sorted(config.AXIS_MAP.items()):
        R = build_axis_matrix(m)
        print('arm %s %s  det = %.0f' % (arm, m, np.linalg.det(R)))
        assert abs(np.linalg.det(R) - 1) < 1e-9

    try:
        # flip only z's sign -> det becomes -1
        build_axis_matrix({'xr_x': '-Z', 'xr_y': '-Y', 'xr_z': '+X'})
        print('✗ reflection matrix was not caught')
    except ValueError as e:
        print('✓ reflection matrix caught:', str(e).split('\n')[0])

    try:
        build_axis_matrix({'xr_x': '-Z', 'xr_y': '-Z', 'xr_z': '-X'})
        print('✗ duplicated axis was not caught')
    except ValueError as e:
        print('✓ duplicated axis caught')

    try:
        build_axis_matrix({'xr_right': '+X', 'xr_up': '-Y', 'xr_forward': '-Z'})
        print('✗ legacy key names were not caught')
    except ValueError as e:
        print('✓ legacy key names caught (avoids semantic-name mixups)')

    # sanity-check the three axis correspondences (arm A)
    R = build_axis_matrix(config.AXIS_MAP['A'])
    for desc, v_xr, expect in [
            ('hand forward (away from body, XR -Z)', [0, 0, -1], 'base +X = forward'),
            ('hand up (XR +Y)', [0, 1, 0], 'base -Y = up'),
            ('hand right (XR +X)', [1, 0, 0], 'base -Z = robot\'s own right')]:
        v_b = R @ np.array(v_xr, dtype=float)
        print('  %s -> base %s   expected %s'
              % (desc, np.array2string(v_b, precision=0, suppress_small=True),
                 expect))

    # handedness flip keeps det = +1
    Rq = quat_to_mat([0.2, 0.3, 0.1, 0.927], 'xyzw')
    _, Rf = flip_handedness([1, 2, 3], Rq, 'z')
    print('✓ det after handedness flip =', round(float(np.linalg.det(Rf)), 6))

    # Euler round-trip self-consistency
    x = mat_to_xyzabc(Rq, [10, 20, 30])
    back = xyzabc_to_rot(x)
    print('✓ max Euler round-trip error = %.2e' % float(np.abs(back - Rq).max()))
    print('\n⚠️ Euler convention vs. the SDK must be checked separately with '
          'verify_euler_convention() — see replay.py')
