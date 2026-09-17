#!/usr/bin/env python3
"""Print both arms' current joint positions -- read-only, no command ever
sent (just ArmDriver.state(), a subscribe()). Safe to run any time,
including alongside another script or run_teleop.py.

Usage:
    python3 scripts/get_current_pos.py            # reads all arms in config.ARMS
    python3 scripts/get_current_pos.py --arms A
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                                    # noqa: E402
from drivers.arm_driver import ArmDriver, RobotConnection, ERR_CODE_CN  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--arms', default=None, help='Defaults to config.ARMS.')
args = ap.parse_args()
arms = list(args.arms.upper()) if args.arms else list(config.ARMS)

conn = RobotConnection(config.ROBOT_IP)
try:
    for a in arms:
        st = ArmDriver(conn, a, config).state()
        fault = ' ⚠️ err=%s(%s)' % (st['err'], ERR_CODE_CN.get(st['err'], '未知')) \
            if st['err'] else ''
        print('臂%s cur_state=%s%s' % (a, st['cur'], fault))
        print('  q     = %s' % [round(v, 2) for v in st['q']])
        print('  q_cmd = %s' % [round(v, 2) for v in st['q_cmd']])
finally:
    conn.close()
