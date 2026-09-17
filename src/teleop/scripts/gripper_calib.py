#!/usr/bin/env python3
"""Gripper travel calibration — step through target positions, pausing at
each so a human can visually confirm open/close direction and end-stops.

Stays enabled for the whole sequence; see docs/scripts.md for why and
for the abort thresholds.

Usage:
    python3 scripts/gripper_calib.py --arm A --steps 0.55,0.45,0.35,0.25 --hold 20
    python3 scripts/gripper_calib.py --arm A --steps 0.75,0.85,0.95 --hold 20   # opposite direction
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
ap.add_argument('--steps', required=True, help='Comma-separated target positions (rad).')
ap.add_argument('--hold', type=float, default=20.0, help='Seconds to hold at each step.')
ap.add_argument('--ramp', type=float, default=0.3, help='Ramp speed, rad/s.')
ap.add_argument('--rate', type=float, default=100.0)
args = ap.parse_args()

STEPS = [float(x) for x in args.steps.split(',')]
KP, KD = config.GRIPPER_KP, config.GRIPPER_KD
TAU_ABORT, DEV_ABORT = config.GRIPPER_MAX_TAU, config.GRIPPER_MAX_ERR_RAD
DT = 1.0 / args.rate

conn = RobotConnection(config.ROBOT_IP)
g = Gripper(conn.robot)
arm = args.arm
aborted = None
q_cmd = None


def check(tag):
    """Read state and check abort thresholds. Returns state dict or None."""
    global aborted
    s = g.state(arm, timeout=0.01)
    if not s:
        return None
    if abs(s['tau']) > TAU_ABORT:
        aborted = '力矩 %.2f N·m 超限（顶到机械限位或夹住东西）' % s['tau']
    elif abs(q_cmd - s['q']) > DEV_ABORT:
        aborted = '位置偏差 %.2f rad 超限' % (q_cmd - s['q'])
    return s


try:
    g.enable(arm)
    time.sleep(0.15)
    s = g.state(arm, timeout=0.3)
    if not s:
        raise SystemExit('使能后无反馈 —— 先跑 tools/probe_gripper_alive.py')
    q_cmd = s['q']
    print('起始位置 %+.4f rad   tau=%+.3f\n' % (q_cmd, s['tau']))

    # hold at zero error for 0.3s; gripper should not move
    for _ in range(int(0.3 * args.rate)):
        g.mit(arm, KP, KD, q_cmd)
        time.sleep(DT)

    step_rad = args.ramp * DT
    for i, target in enumerate(STEPS, 1):
        print('--- 第 %d 步 → %+.3f rad ---' % (i, target))
        while abs(q_cmd - target) > 1e-3 and not aborted:
            q_cmd += step_rad if target > q_cmd else -step_rad
            if abs(q_cmd - target) < step_rad:
                q_cmd = target
            g.mit(arm, KP, KD, q_cmd)
            check('move')
            time.sleep(DT)
        if aborted:
            break

        s = check('hold')
        print('    到位 cmd=%+.3f  fb=%+.3f  tau=%+.3f'
              % (q_cmd, s['q'] if s else float('nan'),
                 s['tau'] if s else float('nan')))
        print('    ⏸  保持 %.0f 秒 —— 请观察夹爪' % args.hold)

        t0 = time.monotonic()
        last_print = 0.0
        while time.monotonic() - t0 < args.hold and not aborted:
            g.mit(arm, KP, KD, q_cmd)
            s = check('hold')
            el = time.monotonic() - t0
            if el - last_print >= 5.0:
                last_print = el
                print('      t=%2.0fs  fb=%+.3f  tau=%+.3f'
                      % (el, s['q'] if s else float('nan'),
                         s['tau'] if s else float('nan')))
            time.sleep(DT)
        if aborted:
            break

    if aborted:
        print('\n❌ 保护触发：%s' % aborted)
        print('   → 这一步的位置就是该方向的端点附近，别再往前推')
    else:
        print('\n✅ 全部 %d 步完成，无保护触发' % len(STEPS))
finally:
    try:
        g.disable_all()
    finally:
        conn.close()
    print('夹爪已失能，连接已断开')
