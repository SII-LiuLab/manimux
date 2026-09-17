#!/usr/bin/env python3
"""Modified-DH forward kinematics and analytic Jacobian for the 7-DOF arm.

Pure numpy, no SDK/ctypes dependency -- takes an already-loaded DH table
(e.g. from Marvin_Kine.load_config) rather than touching the SDK itself.
Exists so the differential-IK QP (algos/diff_ik.py) and, later, WBC have a
Jacobian that isn't tied to Marvin_Kine's analytical ik()/ik_nsp() calls.

See docs/algos.md#kinematics for how the modified-DH convention and the
static row-7 flange offset were confirmed empirically against
Marvin_Kine.fk() and joints2JacobMatrix(), and for the deg-vs-rad unit
derivation used in jacobian().
"""
import math

import numpy as np


class ArmKinematics:
    """FK/Jacobian from an 8x4 modified-DH table: rows 0-6 are joints 1-7
    (the joint angle is added to each row's theta offset), row 7 is a
    static flange transform applied last, with no joint added."""

    N_JOINTS = 7

    def __init__(self, dh_table):
        self.dh = [list(row) for row in dh_table]
        if len(self.dh) != self.N_JOINTS + 1:
            raise ValueError('expected %d DH rows (%d joints + 1 static '
                             'flange offset), got %d'
                             % (self.N_JOINTS + 1, self.N_JOINTS, len(self.dh)))

    @staticmethod
    def _link_transform(alpha_deg, a_mm, d_mm, theta_deg):
        """One modified-DH link transform: Rx(alpha) Tx(a) Rz(theta) Tz(d)."""
        al, th = math.radians(alpha_deg), math.radians(theta_deg)
        ca, sa, ct, st = math.cos(al), math.sin(al), math.cos(th), math.sin(th)
        return np.array([
            [ct,       -st,       0.0,  a_mm],
            [st * ca,   ct * ca, -sa,   -sa * d_mm],
            [st * sa,   ct * sa,  ca,    ca * d_mm],
            [0.0,       0.0,      0.0,   1.0]])

    def _joint_frames(self, q_deg):
        """Cumulative transform/origin/z-axis at frames 0 (base) through 7
        (after joint 7, before the static flange offset)."""
        T = np.eye(4)
        origins = [T[:3, 3].copy()]
        zaxes = [T[:3, 2].copy()]
        for j in range(self.N_JOINTS):
            alpha, a, d, theta0 = self.dh[j]
            T = T @ self._link_transform(alpha, a, d, theta0 + q_deg[j])
            origins.append(T[:3, 3].copy())
            zaxes.append(T[:3, 2].copy())
        return T, origins, zaxes

    def fk(self, q_deg):
        """Flange pose, 4x4 homogeneous transform (mm). Matches
        Marvin_Kine.fk() to ~1e-8mm -- see docs/algos.md#kinematics."""
        T, _, _ = self._joint_frames(q_deg)
        return T @ self._link_transform(*self.dh[self.N_JOINTS])

    def fk_and_jacobian(self, q_deg):
        """Both, from ONE walk of the chain.

        fk() and jacobian() each call _joint_frames(); a caller that wants
        both at the same configuration -- diff_ik.solve() does, every frame
        -- otherwise walks the 7-link chain twice for identical results.
        Returns exactly what fk(q) and jacobian(q) would return.
        """
        T, origins, zaxes = self._joint_frames(q_deg)
        flange = T @ self._link_transform(*self.dh[self.N_JOINTS])
        return flange, self._jacobian_from_frames(flange[:3, 3], origins, zaxes)

    def jacobian(self, q_deg):
        """Geometric Jacobian, 6x7, column i = joint i+1's contribution.

        Rows 0-2 (linear): mm per deg of joint.
        Rows 3-5 (angular): deg/s of end-effector angular velocity (as an
        axis-angle rotation vector) per deg/s of joint rate -- exact for
        velocities. This is NOT the same as a per-axis xyzabc Euler-angle
        rate: naively feeding a raw xyzabc coordinate difference into these
        rows as if it were an angular-velocity vector is only valid near
        the identity rotation, and is measurably wrong (not just
        approximately so) at a general orientation -- see
        docs/algos.md#kinematics; diff_ik.py uses a proper axis-angle
        rotation error instead.

        Joint i's rotation axis is the z-axis of the frame *after* its own
        transform (frame i+1, not frame i) -- modified DH applies Rz(theta_i)
        before Tz(d_i), so frame i+1's z-axis is invariant under theta_i and
        equals the physical rotation axis. Using frame i instead is a classic
        off-by-one that silently produces a wrong-but-plausible-looking
        Jacobian; a finite-difference cross-check catches it, visual
        inspection does not.
        """
        T, origins, zaxes = self._joint_frames(q_deg)
        p_e = (T @ self._link_transform(*self.dh[self.N_JOINTS]))[:3, 3]
        return self._jacobian_from_frames(p_e, origins, zaxes)

    def _jacobian_from_frames(self, p_e, origins, zaxes):
        """jacobian()'s body, once the chain has already been walked."""
        J = np.zeros((6, self.N_JOINTS))
        deg2rad = math.pi / 180.0
        for i in range(self.N_JOINTS):
            z, p = zaxes[i + 1], origins[i + 1]
            # z x (p_e - p), written out. np.cross on 3-vectors spends more
            # than 7 us per call inside moveaxis/normalize_axis_tuple, which
            # at 250 Hz x 2 arms is a seventh of the whole diff-IK budget;
            # the arithmetic below is the same expression it evaluates.
            rx, ry, rz = p_e[0] - p[0], p_e[1] - p[1], p_e[2] - p[2]
            J[:3, i] = np.array([z[1] * rz - z[2] * ry,
                                 z[2] * rx - z[0] * rz,
                                 z[0] * ry - z[1] * rx]) * deg2rad  # mm / deg
            J[3:, i] = z                                # deg/s per deg/s (see docstring)
        return J


if __name__ == '__main__':
    # Self-test: cross-validate against Marvin_Kine three independent ways.
    # No robot needed, but the SDK is used here purely to load the DH table
    # and as a ground-truth oracle -- none of its solving is used.
    import os
    import random
    import sys

    _SDK = os.environ.get('MARVIN_SDK',
                          '/home/jw/Downloads/TJ_FX_ROBOT_CONTRL_SDK')
    if _SDK not in sys.path:
        sys.path.insert(0, _SDK)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from SDK_PYTHON.fx_kine import Marvin_Kine  # noqa: E402
    import config  # noqa: E402

    kine = Marvin_Kine()
    kine.log_switch(0)
    cfg = kine.load_config(arm_type=0, config_path=config.KINE_CFG)
    kine.initial_kine(robot_type=cfg['TYPE'][0], dh=cfg['DH'][0],
                      pnva=cfg['PNVA'][0], j67=[[0, 0, 0]] * 4)
    pnva = cfg['PNVA'][0]
    lim_p = [r[0] for r in pnva]
    lim_n = [r[1] for r in pnva]

    kin = ArmKinematics(cfg['DH'][0])

    random.seed(0)

    def rand_q():
        # Stay well inside the limits -- this validates the math, not
        # limit-adjacent singular configurations.
        return [random.uniform(lim_n[i] * 0.6, lim_p[i] * 0.6) for i in range(7)]

    # 1) FK vs Marvin_Kine.fk(), full 4x4 (position + orientation)
    max_fk_err = 0.0
    for _ in range(50):
        q = rand_q()
        T_mine = kin.fk(q)
        T_sdk = kine.fk(joints=q)
        max_fk_err = max(max_fk_err, float(np.max(np.abs(T_mine - T_sdk))))
    assert max_fk_err < 1e-5, 'FK mismatch vs Marvin_Kine.fk(): %g' % max_fk_err
    print('OK  FK matches Marvin_Kine.fk() over 50 random configs '
          '(max elementwise err %.2e)' % max_fk_err)

    # 2) Jacobian (linear part) vs finite-difference of our own FK
    def fk_pos(q):
        return kin.fk(q)[:3, 3]

    def fd_jacobian_linear(q, eps_deg=1e-4):
        Jfd = np.zeros((3, 7))
        for i in range(7):
            qp, qm = list(q), list(q)
            qp[i] += eps_deg
            qm[i] -= eps_deg
            Jfd[:, i] = (fk_pos(qp) - fk_pos(qm)) / (2 * eps_deg)
        return Jfd

    max_jac_fd_err = 0.0
    for _ in range(30):
        q = rand_q()
        Ja = kin.jacobian(q)[:3]
        Jfd = fd_jacobian_linear(q)
        max_jac_fd_err = max(max_jac_fd_err, float(np.max(np.abs(Ja - Jfd))))
    assert max_jac_fd_err < 1e-4, ('analytic vs finite-diff Jacobian '
                                   'mismatch: %g' % max_jac_fd_err)
    print('OK  analytic Jacobian matches finite-difference of our own FK '
          'over 30 random configs (max err %.2e mm/deg)' % max_jac_fd_err)

    # 3) Jacobian vs Marvin_Kine.joints2JacobMatrix() -- independent oracle.
    # SDK reports linear part in m/rad, angular in (unitless) rad/rad; ours
    # is mm/deg linear, deg/deg angular. Linear: mm/deg = (m/rad) * 1000 *
    # (pi/180). Angular: deg/deg == rad/rad exactly (both are angle-ratios,
    # the unit conversion cancels between numerator and denominator).
    deg2rad = math.pi / 180.0
    max_jac_sdk_err = 0.0
    for _ in range(30):
        q = rand_q()
        Ja = kin.jacobian(q)
        Jsdk = np.array(kine.joints2JacobMatrix(joints=q))
        Jsdk_converted = np.vstack([Jsdk[:3] * 1000.0 * deg2rad, Jsdk[3:]])
        max_jac_sdk_err = max(max_jac_sdk_err,
                              float(np.max(np.abs(Ja - Jsdk_converted))))
    assert max_jac_sdk_err < 1e-6, ('Jacobian mismatch vs SDK '
                                    'joints2JacobMatrix: %g' % max_jac_sdk_err)
    print('OK  Jacobian matches Marvin_Kine.joints2JacobMatrix() over 30 '
          'random configs after unit conversion (max err %.2e)'
          % max_jac_sdk_err)

    print('\nall passed')
