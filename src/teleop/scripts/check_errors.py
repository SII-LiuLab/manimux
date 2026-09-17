#!/usr/bin/env python3
"""Check and clear both arms' latched faults -- nothing else. Connects,
reads cur_state/err_code per arm, and runs drivers.arm_driver.ensure_clear()
(clear_error + re-read until confirmed clean, see docs/drivers.md) on any
arm that is faulted.

No servo is enabled, no mode switch is attempted, and no joint command is
ever sent -- the arm does not move. That is the whole point: every other
entry point (run_teleop.py, home_now.py, goto_joints.py, set_state.py)
clears errors only as a prelude to *moving*, so after an e-stop there was
no way to just drop the latched err_code and walk away. Safe to run
before deciding what to do next.

Typical use: an e-stop leaves err=13 (Emcy) latched even after the button
is twisted back out, and the servo stays disabled until it is cleared --
run this, confirm ✅ on both arms, then start the real script.

Exit code 0 only if every requested arm ended clean; 1 otherwise (so it
can gate a shell `&&`).

Usage:
    python3 scripts/check_errors.py             # both arms (config.ARMS)
    python3 scripts/check_errors.py --arms A
    python3 scripts/check_errors.py --retries 10 --wait 0.5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                                    # noqa: E402
from drivers.arm_driver import (ArmDriver, RobotConnection,      # noqa: E402
                                ERR_CODE_CN, STATE_ERROR, ensure_clear)

ap = argparse.ArgumentParser()
ap.add_argument('--arms', default=None, help='Defaults to config.ARMS.')
ap.add_argument('--retries', type=int, default=5,
                help='clear_error attempts per arm before giving up (default 5).')
ap.add_argument('--wait', type=float, default=0.3,
                help='Seconds to wait after each clear_error before re-reading '
                     '(default 0.3).')
args = ap.parse_args()
arms = list(args.arms.upper()) if args.arms else list(config.ARMS)

bad = [a for a in arms if a not in ('A', 'B')]
if bad:
    raise SystemExit("❌ 未知的臂 %s（只支持 A / B）" % bad)

print('连接 %s…' % config.ROBOT_IP)
conn = RobotConnection(config.ROBOT_IP)
print('✅ 控制器版本 %s' % conn.version)

failed = []
try:
    for a in arms:
        drv = ArmDriver(conn, a, config)
        st = drv.state()
        if not st['err'] and st['cur'] != STATE_ERROR:
            # already clean -- don't fire a pointless clear_error
            print('臂%s ✅ 无故障 cur=%s err=0 q=%s'
                 % (a, st['cur'], [round(v, 2) for v in st['q']]))
            continue

        print('臂%s ⚠️  cur=%s err=%s(%s) —— 开始清错…'
             % (a, st['cur'], st['err'], ERR_CODE_CN.get(st['err'], '未知')))
        ok, st = ensure_clear(conn, drv, a, retries=args.retries, wait=args.wait)
        if ok:
            print('臂%s ✅ 已清错 cur=%s err=0 q=%s'
                 % (a, st['cur'], [round(v, 2) for v in st['q']]))
        else:
            failed.append(a)
            print('臂%s ❌ 清错 %d 次后仍有故障 cur=%s err=%s(%s)'
                 % (a, args.retries, st['cur'], st['err'],
                    ERR_CODE_CN.get(st['err'], '未知')))
finally:
    # never engaged a servo, so there is nothing to disable -- just drop the
    # connection (leaving it open would hold port 4730 against the GUI).
    conn.close()
    print('连接已断开')

if failed:
    raise SystemExit(
        '❌ 臂%s 仍有故障 —— 急停按钮是否还按着？总线/伺服报警需要用 '
        'MarvinPlatform GUI 或 tools/wait_ready.py 排查，清错清不掉。'
        % '、'.join(failed))
print('✅ %s 全部状态干净' % '、'.join(arms))
