#!/usr/bin/env python3
"""Gripper startup smoke test — verifies only GripperThread.start(), no arms involved.

See docs/scripts.md for why this is isolated from run_teleop and what
bug this timing check guards against.

Usage:
    python3 scripts/gripper_smoke.py            # arm A by default
    python3 scripts/gripper_smoke.py --arms AB

⚠️ Gripper will move once --run is used — see docs/scripts.md.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                        # noqa: E402
from drivers.arm_driver import RobotConnection       # noqa: E402
from drivers.gripper import Gripper, GripperThread   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--arms', default='A')
ap.add_argument('--run', type=float, default=0.0,
                help='Seconds to run the ramp thread (0 = only verify start()).')
ap.add_argument('--target', type=float, default=None,
                help='Target fraction 0-1 (0=open, 1=closed); omit to hold current position.')
ap.add_argument('--open-first', action='store_true',
                help='Open fully, wait --wait seconds, then close to --target.')
ap.add_argument('--wait', type=float, default=20.0,
                help='Seconds to wait after opening (placing an object).')
args = ap.parse_args()
arms = list(args.arms.upper())

print('连接 %s…' % config.ROBOT_IP)
conn = RobotConnection(config.ROBOT_IP)
print('✅ 控制器版本 %s' % conn.version)

g = Gripper(conn.robot)
th = GripperThread(g, config, arms)
try:
    print('\n--- start()：enable → sleep(0.15) → state(timeout=0.3) ---')
    th.start()
    print('✅ start() 通过 —— 使能后读到了状态，时序正确')
    for a in arms:
        st = g.state(a, timeout=0.3)
        if st:
            print('   夹爪%s  q=%.4f rad  tau=%.3f N·m  温度 %s/%s ℃'
                  % (a, st['q'], st['tau'], st['t_mos'], st['t_rotor']))
        print('   零误差起步位置 _cur[%s] = %.4f rad' % (a, th._cur[a]))

    if args.open_first:
        for a in arms:
            th.set_target(a, 0.0)
        print('\n>>> 正在张开到全开…')
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6.0 and not th.fault:
            time.sleep(0.5)
        for a in arms:
            st = g.state(a, timeout=0.05)
            print('    夹爪%s 已到 q=%+.4f rad' % (a, st['q'] if st else float('nan')))

        print('\n>>> ⏸  请把物体放入爪口 —— %.0f 秒后开始夹' % args.wait)
        t0 = time.monotonic()
        nxt = args.wait
        while True:
            left = args.wait - (time.monotonic() - t0)
            if left <= 0:
                break
            if left <= nxt:
                print('    倒计时 %2.0f s' % left)
                nxt -= 5
            time.sleep(0.5)
        print('\n>>> 开始夹\n')

    if args.target is not None:
        for a in arms:
            th.set_target(a, args.target)
        print('\n>>> 目标设为 %.2f（%.4f rad）—— 力限超前量 %.2f rad，'
              '预期夹持力矩 ≈ %.1f N·m'
              % (args.target, th._rad(args.target),
                 config.GRIPPER_MAX_OVERSHOOT_RAD,
                 config.GRIPPER_KP * config.GRIPPER_MAX_OVERSHOOT_RAD))
        print('>>> 想验证力限：用手挡住夹爪，力矩应稳定在上面这个值附近，'
              '而不是一路涨到 %.1f 触发保护' % config.GRIPPER_MAX_TAU)

    if args.run > 0:
        print('\n--- 斜坡线程跑 %.1fs ---' % args.run)
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.run:
            time.sleep(0.5)
            if th.fault:
                print('❌ 保护触发：%s' % th.fault)
                break
            for a in arms:
                st = g.state(a, timeout=0.05)
                if st:
                    print('   %s q=%.4f tau=%.3f' % (a, st['q'], st['tau']))
        if not th.fault:
            print('✅ 斜坡运行无保护触发')
    else:
        print('\n（未加 --run，线程不跑，夹爪不动）')
finally:
    th.stop()                 # calls disable_all() internally
    conn.close()
    print('\n夹爪已失能，连接已断开')
