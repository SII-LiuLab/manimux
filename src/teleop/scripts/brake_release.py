#!/usr/bin/env python3
"""Force-release one arm's brakes, then force-apply them again on Enter.

Controller params BRAK0 (arm A) / BRAK1 (arm B): 2 = force release,
1 = force brake -- DEMO_PYTHON/showcase_apply-brake_release-brake.py. For
an arm twisted into a pose it cannot enable from (after a crash or e-stop,
or a torque request the controller keeps rejecting), move it by hand and
lock it again before switching to a real control mode.

Two deviations from the vendor demo:
    1. One arm per run, not A-then-B back to back.
    2. Brakes stay released until Enter instead of a fixed 30 s, and are
       re-applied in a `finally`, so Ctrl+C or a crash also brakes.

No servo and no gravity compensation while released: the arm drops the
moment BRAKx=2 lands. Hold the arm before confirming. This does not clear
err=6 either -- entering drag afterwards still needs `--tool`.

Usage:
    python3 scripts/brake_release.py A
    python3 scripts/brake_release.py B
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                                    # noqa: E402
from drivers.arm_driver import RobotConnection                   # noqa: E402

BRAKE_PARAM = {'A': 'BRAK0', 'B': 'BRAK1'}
RELEASE, APPLY = 2, 1

ap = argparse.ArgumentParser()
ap.add_argument('arm', type=str.upper, choices=['A', 'B'])
args = ap.parse_args()
param = BRAKE_PARAM[args.arm]


def set_brake(value):
    """Write BRAKx and report the readback."""
    conn.robot.set_param('int', param, value)
    time.sleep(0.2)
    _, got = conn.robot.get_param('int', param)
    return got


print('连接 %s…' % config.ROBOT_IP)
conn = RobotConnection(config.ROBOT_IP)
print('✅ 控制器版本 %s' % conn.version)
try:
    conn.check_and_clear_errors()
    input('⚠️  臂%s 即将强制松闸 —— 不上伺服、没有重力补偿，臂会立刻下坠。\n'
          '    确认有人已经托住臂%s、另一人守在急停旁，按 Enter 松闸'
          '（Ctrl+C 取消）…' % (args.arm, args.arm))
    try:
        got = set_brake(RELEASE)
        print('🔓 臂%s 已松闸（%s=%s 回读 %s）。调整好姿态后按 Enter 抱闸…'
             % (args.arm, param, RELEASE, got))
        input()
    finally:
        got = set_brake(APPLY)
        time.sleep(3)
        print('🔒 臂%s 已抱闸（%s=%s 回读 %s）' % (args.arm, param, APPLY, got))
finally:
    # never enabled a servo -- just drop the connection so port 4730 is free
    conn.close()
    print('连接已断开')
