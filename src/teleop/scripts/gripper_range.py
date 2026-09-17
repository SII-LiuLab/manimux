#!/usr/bin/env python3
"""Gripper travel measurement — zero-torque free mode, operator moves the
jaw by hand through its full range, script logs min/max position.

See docs/scripts.md for the MIT zero-torque principle and why this beats
probing end-stops step by step with gripper_calib.py.

Usage:
    python3 scripts/gripper_range.py                # 40s by default
    python3 scripts/gripper_range.py --arm A --sec 60
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                        # noqa: E402
from drivers.arm_driver import RobotConnection       # noqa: E402
from drivers.gripper import Gripper                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--arm', default='A', choices=['A', 'B'])
ap.add_argument('--sec', type=float, default=40.0)
ap.add_argument('--rate', type=float, default=100.0)
args = ap.parse_args()

DT = 1.0 / args.rate
conn = RobotConnection(config.ROBOT_IP)
g = Gripper(conn.robot)
arm = args.arm

try:
    g.enable(arm)
    time.sleep(0.15)
    s = g.state(arm, timeout=0.3)
    if not s:
        raise SystemExit('使能后无反馈 —— 先跑 tools/probe_gripper_alive.py')
    print('起始位置 %+.4f rad\n' % s['q'])
    print('>>> 现在夹爪是【零力矩自由态】，可以徒手掰动。')
    print('>>> 请依次掰到 ①完全张开 ②完全闭合，各停一下。%.0f 秒后自动结束。\n'
          % args.sec)

    qmin, qmax, tmax = s['q'], s['q'], 0.0
    t0 = time.monotonic()
    last = 0.0
    while time.monotonic() - t0 < args.sec:
        g.mit(arm, 0.0, 0.0, 0.0, 0.0, 0.0)     # kp=kd=tau=0 -> zero torque
        st = g.state(arm, timeout=0.005)
        if st:
            qmin = min(qmin, st['q'])
            qmax = max(qmax, st['q'])
            tmax = max(tmax, abs(st['tau']))
            el = time.monotonic() - t0
            if el - last >= 1.0:
                last = el
                print('  t=%2.0fs  q=%+.4f   已测区间 [%+.4f, %+.4f]'
                      % (el, st['q'], qmin, qmax))
        time.sleep(DT)

    span = qmax - qmin
    print('\n=== 实测行程 ===')
    print('最小 q = %+.4f rad   ← 张开端（q 减小 = 向外张开）' % qmin)
    print('最大 q = %+.4f rad   ← 闭合端' % qmax)
    print('行程   = %.4f rad' % span)
    print('自由态最大力矩 %.3f N·m（应接近 0，否则说明没真正卸力）' % tmax)

    if span < 0.05:
        print('\n⚠️ 行程太小，可能没掰动 —— 确认夹爪确实处于自由态再试一次')
    else:
        m = 0.02        # margin at each end to avoid commanding into the hard stop
        print('\n建议填入 config.py（两端各留 %.2f rad 余量，防止顶限位）：' % m)
        print('    GRIPPER_OPEN_RAD  = %.3f' % (qmin + m))
        print('    GRIPPER_CLOSE_RAD = %.3f' % (qmax - m))
finally:
    try:
        g.disable_all()
    finally:
        conn.close()
    print('\n夹爪已失能，连接已断开')
