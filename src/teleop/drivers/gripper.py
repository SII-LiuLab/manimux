#!/usr/bin/env python3
"""Gripper control — OmniGripper (DM4310), over Marvin's end-effector CAN
channel 1 pass-through.

Protocol layer is reused from tools/km_gripper.py; this module only adds
connection reuse (shares the arm driver's Marvin_Robot) and a 50Hz
GripperThread for slew-limited, force-limited position control.

See docs/drivers.md for design rationale, empirical data, and known pitfalls.
"""
import os
import sys
import threading
import time

_SDK = os.environ.get('MARVIN_SDK',
                      '/home/jw/Downloads/TJ_FX_ROBOT_CONTRL_SDK')
for _p in (_SDK, os.path.join(_SDK, 'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# protocol constants and packing functions come from km_gripper.py
from km_gripper import (ARM_SLAVE, CH_CAN, Q_MAX, DQ_MAX, TAU_MAX,  # noqa: E402
                        STATUS, f2u, u2f)

import ctypes  # noqa: E402


class Gripper:
    """Gripper controller over an already-connected Marvin_Robot."""

    def __init__(self, robot):
        """:param robot: an already-connected Marvin_Robot (from RobotConnection)"""
        self.robot = robot
        lib = robot.robot
        for n in ('OnGetChDataA', 'OnGetChDataB'):
            f = getattr(lib, n)
            f.argtypes = [ctypes.POINTER(ctypes.c_ubyte),
                          ctypes.POINTER(ctypes.c_long)]
            f.restype = ctypes.c_long
        for n in ('OnSetChDataA', 'OnSetChDataB'):
            f = getattr(lib, n)
            f.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_long,
                          ctypes.c_long]
            f.restype = ctypes.c_bool
        self.lib = lib
        self._tx = (ctypes.c_ubyte * 256)()
        self._rx = (ctypes.c_ubyte * 256)()
        self._enabled = set()

    # ---------- low level (mirrors km_gripper) ----------

    def send(self, arm, can_id, data8):
        payload = int(can_id).to_bytes(4, 'little') + bytes(data8).ljust(8, b'\x00')
        for i, b in enumerate(payload):
            self._tx[i] = b
        fn = self.lib.OnSetChDataA if arm == 'A' else self.lib.OnSetChDataB
        for _ in range(30):             # previous frame not yet drained -> retry
            if fn(self._tx, len(payload), CH_CAN):
                return True
            time.sleep(0.001)
        return False

    def _raw(self, arm):
        fn = self.lib.OnGetChDataA if arm == 'A' else self.lib.OnGetChDataB
        rc = ctypes.c_long(CH_CAN)      # shipped .so treats ret_ch as an input; must be 1~3
        n = fn(self._rx, ctypes.byref(rc))
        return bytes(self._rx)[:n] if n and n > 0 else None

    def state(self, arm, timeout=0.05):
        last, t0 = None, time.time()
        while time.time() - t0 < timeout:
            f = self._raw(arm)
            if f and len(f) >= 12:
                d = f[4:12]
                last = {
                    'status': (d[0] & 0xF0) >> 4,
                    'motor_id': d[0] & 0x0F,
                    'q': u2f((d[1] << 8) | d[2], -Q_MAX, Q_MAX, 16),
                    'dq': u2f((d[3] << 4) | (d[4] >> 4), -DQ_MAX, DQ_MAX, 12),
                    'tau': u2f(((d[4] & 0x0F) << 8) | d[5],
                               -TAU_MAX, TAU_MAX, 12),
                    't_mos': d[6], 't_rotor': d[7],
                }
            else:
                time.sleep(0.001)
        return last

    def enable(self, arm):
        ok = self.send(arm, ARM_SLAVE[arm], [0xFF] * 7 + [0xFC])
        if ok:
            self._enabled.add(arm)
        return ok

    def disable(self, arm):
        self._enabled.discard(arm)
        return self.send(arm, ARM_SLAVE[arm], [0xFF] * 7 + [0xFD])

    def disable_all(self):
        for arm in list(self._enabled) or ['A', 'B']:
            try:
                self.disable(arm)
            except Exception:
                pass

    def mit(self, arm, kp, kd, q, dq=0.0, tau=0.0):
        kp_u, kd_u = f2u(kp, 0, 500, 12), f2u(kd, 0, 5, 12)
        q_u = f2u(q, -Q_MAX, Q_MAX, 16)
        dq_u = f2u(dq, -DQ_MAX, DQ_MAX, 12)
        t_u = f2u(tau, -TAU_MAX, TAU_MAX, 12)
        return self.send(arm, ARM_SLAVE[arm], [
            (q_u >> 8) & 0xFF, q_u & 0xFF,
            dq_u >> 4, ((dq_u & 0xF) << 4) | ((kp_u >> 8) & 0xF), kp_u & 0xFF,
            kd_u >> 4, ((kd_u & 0xF) << 4) | ((t_u >> 8) & 0xF), t_u & 0xFF])


class GripperThread:
    """50Hz thread converting a [0,1] open/close target into slew-limited MIT commands.

    Main control loop only calls set_target(arm, 0~1); non-blocking.
    """

    def __init__(self, gripper, cfg, arms):
        self.g = gripper
        self.cfg = cfg
        self.arms = list(arms)
        self._target = {a: 0.0 for a in self.arms}
        self._cur = {a: None for a in self.arms}
        self._q_meas = {a: None for a in self.arms}     # last measured position, for force limiting
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.fault = None               # set on protection trip; main loop should check

    def set_target(self, arm, x):
        """x in [0,1], 0=open 1=closed. Typically driven directly by the trigger axis."""
        with self._lock:
            self._target[arm] = max(0.0, min(1.0, float(x)))

    def _rad(self, x):
        lo, hi = self.cfg.GRIPPER_OPEN_RAD, self.cfg.GRIPPER_CLOSE_RAD
        return lo + (hi - lo) * x

    def _inv_rad(self, q):
        """rad -> [0,1], inverse of _rad(). Used to seed the target at the current position."""
        lo, hi = self.cfg.GRIPPER_OPEN_RAD, self.cfg.GRIPPER_CLOSE_RAD
        if hi == lo:
            return 0.0
        return max(0.0, min(1.0, (q - lo) / (hi - lo)))

    def start(self):
        """Enable and seed commands at the measured position. See docs/drivers.md for
        the enable -> sleep -> state ordering requirement."""
        for a in self.arms:
            self.g.enable(a)                # ① enable first; the enable frame itself triggers a reply
            time.sleep(0.15)                # ② give it time to respond
            st = self.g.state(a, timeout=0.3)   # ③ then read; don't use the default 0.05 timeout
            if st is None:
                self.g.disable_all()        # don't leave it energized if we can't read it back
                raise RuntimeError(
                    '夹爪 %s 使能后仍无应答。依次查：\n'
                    '  1) 控制器刚重启？CAN 透传还不通 → 先跑 tools/wait_ready.py\n'
                    '  2) 夹爪供电、末端连接线是否插紧\n'
                    '  3) tools/probe_gripper_alive.py 是否还能通过' % a)
            self._cur[a] = st['q']
            self._q_meas[a] = st['q']
            # seed target at current position too, else the thread would
            # immediately slew toward the (uncalibrated) target default
            self._target[a] = self._inv_rad(st['q'])
            self.g.mit(a, self.cfg.GRIPPER_KP, self.cfg.GRIPPER_KD, st['q'])
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        period = 1.0 / self.cfg.GRIPPER_HZ
        max_step = self.cfg.GRIPPER_SLEW_RAD_S * period
        try:
            while not self._stop.is_set():
                for a in self.arms:
                    with self._lock:
                        tgt = self._rad(self._target[a])
                    cur = self._cur[a]
                    d = max(-max_step, min(max_step, tgt - cur))
                    cur += d

                    # force-limited position control: clamp command near the
                    # measured position; see docs/drivers.md
                    q_meas = self._q_meas[a]
                    if q_meas is not None:
                        lim = self.cfg.GRIPPER_MAX_OVERSHOOT_RAD
                        cur = max(q_meas - lim, min(q_meas + lim, cur))

                    self._cur[a] = cur
                    self.g.mit(a, self.cfg.GRIPPER_KP, self.cfg.GRIPPER_KD, cur)

                    st = self.g.state(a, timeout=0.005)
                    if st:
                        self._q_meas[a] = st['q']
                        if abs(st['tau']) > self.cfg.GRIPPER_MAX_TAU:
                            self.fault = ('%s 力矩 %.2fN·m 超限'
                                          % (a, st['tau']))
                            return
                        if abs(st['q'] - cur) > self.cfg.GRIPPER_MAX_ERR_RAD:
                            self.fault = ('%s 位置偏差 %.2frad 超限'
                                          % (a, st['q'] - cur))
                            return
                time.sleep(period)
        finally:
            self.g.disable_all()        # protection 4: unconditional disable

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.g.disable_all()


if __name__ == '__main__':
    import argparse

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    import config                                    # noqa: E402
    from drivers.arm_driver import RobotConnection   # noqa: E402

    ap = argparse.ArgumentParser(
        description='Standalone repeated open/close test for the gripper '
                    '-- only uses the gripper CAN pass-through, does not '
                    'touch any joints.')
    ap.add_argument('--arm', default='A', choices=['A', 'B'])
    ap.add_argument('--num', type=int, default=1,
                    help='Number of open/close cycles (e.g. --num 5).')
    ap.add_argument('--hold', type=float, default=3.0,
                    help='Seconds to keep monitoring after each open/close '
                         'reaches its target.')
    ap.add_argument('--pause', type=float, default=2.0,
                    help='Seconds to pause between open/close actions.')
    args = ap.parse_args()

    print('连接 %s…' % config.ROBOT_IP)
    conn = RobotConnection(config.ROBOT_IP)
    print('✅ 控制器版本 %s' % conn.version)

    g = Gripper(conn.robot)
    th = GripperThread(g, config, [args.arm])

    REACH_TOL_RAD = 0.03

    def move_to(label, target):
        """Set target, watch it for --hold seconds. Returns False on fault."""
        print('>>> %s…' % label)
        target_rad = th._rad(target)
        th.set_target(args.arm, target)
        t0 = time.monotonic()
        reach_t = None
        while time.monotonic() - t0 < args.hold:
            if th.fault:
                print('❌ 保护触发：%s' % th.fault)
                return False
            st = g.state(args.arm, timeout=0.05)
            if st:
                print('   q=%+.4f rad  tau=%+.3f N·m' % (st['q'], st['tau']))
                if reach_t is None and abs(st['q'] - target_rad) < REACH_TOL_RAD:
                    reach_t = time.monotonic() - t0
                    print('   ⏱ 到位用时 %.3fs' % reach_t)
            time.sleep(0.2)
        if reach_t is None:
            print('   ⚠️ %.1fs 内未到位（tol=%.2frad）' % (args.hold, REACH_TOL_RAD))
        return True

    try:
        th.start()
        print('起始位置 %.4f rad' % th._cur[args.arm])

        ok = True
        t_start = time.monotonic()
        for i in range(1, args.num + 1):
            print('\n=== 第 %d/%d 次开合 ===' % (i, args.num))
            if not move_to('闭合夹爪 %s' % args.arm, 1.0):
                ok = False
                break
            time.sleep(args.pause)
            if not move_to('张开夹爪 %s' % args.arm, 0.0):
                ok = False
                break
            if i < args.num:
                time.sleep(args.pause)
        total = time.monotonic() - t_start

        if ok:
            print('\n✅ %d 次开合全部完成，无保护触发，总耗时 %.1fs' % (args.num, total))
        else:
            print('\n耗时 %.1fs 后中断' % total)
    finally:
        th.stop()            # calls disable_all() internally
        conn.close()
        print('夹爪已失能，连接已断开')
