#!/usr/bin/env python3
"""Point-to-point joint move -- smoothly drive one arm to a target configuration.

Recovers an arm left in an odd pose after teleop or a disable.
See docs/scripts.md for safety design and hardware notes.

Usage:
    python3 scripts/goto_joints.py --arm A --to 86.43,-76.25,-90.94,-84.22,-14.53,1.65,-10.12
    python3 scripts/goto_joints.py --arm A --to ... --speed 5      # slower
    python3 scripts/goto_joints.py --arm A --to ... --release      # disable on arrival (locks the pose)
    python3 scripts/goto_joints.py --arm A --to ... --control-mode impedance  # compliant tracking, docs/algos.md#impedanceconfig
    python3 scripts/goto_joints.py --arm A --to ... --control-mode impedance --impedance-type cartesian
    python3 scripts/goto_joints.py --arm A --to ... --viewer        # mirror to universal_viewer, see docs/viewer.md

Under impedance mode, larger tracking error is expected (compliance, not a collision).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                              # noqa: E402
from algos.solver_config import load_impedance_config       # noqa: E402
from drivers.arm_driver import (ArmDriver, RobotConnection,  # noqa: E402
                                ERR_CODE_CN, STATE_POSITION, STATE_TORQUE,
                                ensure_clear, move_to_joints, send_joint_commands)
from viewer_bridge import ViewerBridge                       # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument('--arm', default='A', choices=['A', 'B'])
ap.add_argument('--to', required=True, help='7 target joint angles, comma-separated (degrees).')
ap.add_argument('--speed', type=float, default=8.0, help='Max joint speed, deg/s.')
ap.add_argument('--hz', type=float, default=250.0)
ap.add_argument('--release', action='store_true',
                help='Disable and exit on arrival (arm locks its pose).')
ap.add_argument('--control-mode', choices=['position', 'impedance'],
                default='position',
                help='position (default) or impedance (compliant tracking).')
ap.add_argument('--impedance-type', choices=['joint', 'cartesian'],
                default='joint',
                help='joint (default) or cartesian.')
ap.add_argument('--impedance-config', default=None,
                help='Impedance parameter YAML path. Defaults to '
                     'configs/solver/impedance_<impedance-type>.yaml.')
ap.add_argument('--viewer', action='store_true',
                help='Mirror live joint state to universal_viewer (observe '
                     'only -- see docs/viewer.md). Off by default.')
ap.add_argument('--viewer-endpoint', default='tcp://127.0.0.1:5568')
args = ap.parse_args()

TARGET = [float(x) for x in args.to.split(',')]
if len(TARGET) != 7:
    raise SystemExit('--to 需要正好 7 个角度，收到 %d 个' % len(TARGET))

DT = 1.0 / args.hz

impedance_cfg = None
if args.control_mode == 'impedance':
    try:
        impedance_cfg, cfg_path = load_impedance_config(
            args.impedance_type, args.impedance_config, _REPO_ROOT)
    except ValueError as e:
        raise SystemExit('❌ %s' % e)
    print('阻抗(%s)配置: %s\n%s\n'
         % (args.impedance_type, cfg_path, impedance_cfg.model_dump()))
    config.ARM_STATE = STATE_TORQUE
else:
    config.ARM_STATE = STATE_POSITION

conn = RobotConnection(config.ROBOT_IP)
print('✅ 控制器版本 %s' % conn.version)

viewer = ViewerBridge(args.viewer, args.viewer_endpoint)
if args.viewer:
    print('✅ 查看器镜像已开启 -> %s（仅观察，不接收 pause/home）' % args.viewer_endpoint)

drv = ArmDriver(conn, args.arm, config, impedance_config=impedance_cfg)
try:
    print('臂%s 检查状态/清错…' % args.arm)
    ok, st = ensure_clear(conn, drv, args.arm)
    print('臂%s %s cur=%s err=%s q=%s'
         % (args.arm, '✅ 状态干净' if ok else '❌ 清错后仍有故障',
            st['cur'], st['err'], [round(v, 2) for v in st['q']]))
    if not ok:
        raise RuntimeError(
            '臂%s 清错未成功（cur=%s err=%s(%s)）—— 不去尝试切模式/移动，'
            '先用 MarvinPlatform GUI 或 tools/wait_ready.py 排查'
            % (args.arm, st['cur'], st['err'], ERR_CODE_CN.get(st['err'], '未知')))

    q = drv.joints()
    viewer.publish_state(conn.subscribe())
    d = [t - c for t, c in zip(TARGET, q)]
    dmax = max(abs(x) for x in d)
    print('\n当前 %s' % [round(v, 2) for v in q])
    print('目标 %s' % [round(v, 2) for v in TARGET])
    print('差值 %s' % [round(v, 2) for v in d])
    print('最大单轴 %.2f°，按 %.1f°/s 约需 %.1f 秒\n'
          % (dmax, args.speed, dmax / args.speed))

    drv.prepare()                           # switch to configured mode (sets vel/acc, reads back state)
    print('已进入%s模式，开始移动…（Ctrl+C 可随时停）\n'
         % ('位置' if args.control_mode == 'position' else '阻抗'))

    # interpolation/rate-limit/tracking live in arm_driver.move_to_joints (shared with auto-home)
    # -- it has no per-frame hook, so the viewer only sees before/after/hold
    # snapshots here, not the transit itself. See docs/viewer.md.
    ok, why, q_end = move_to_joints(conn, {args.arm: drv},
                                    {args.arm: TARGET},
                                    speed_deg_s=args.speed, hz=args.hz,
                                    max_err_deg=config.MAX_TRACKING_ERR_DEG)
    if not ok:
        print('\n❌ 未走到位：%s' % why)

    # hold at q_end (actual stop point), not TARGET -- avoids snapping if interrupted
    q_cmd = list(q_end[args.arm])
    q_meas = drv.joints()
    viewer.publish_state(conn.subscribe())
    print('\n到位：%s' % [round(v, 2) for v in q_meas])
    print('与目标偏差 %s'
          % [round(t - m, 2) for t, m in zip(TARGET, q_meas)])

    if args.release:
        print('\n--release：失能退出，手臂会抱死保持姿态')
    else:
        print('\n>>> 保持使能中，手臂停在此姿势。按 Ctrl+C 退出'
              '（退出后会抱死保持姿态，不会下垂）')
        try:
            while True:
                send_joint_commands(conn, {args.arm: q_cmd})
                if viewer.due():
                    viewer.publish_state(conn.subscribe())
                time.sleep(DT)
        except KeyboardInterrupt:
            print('\n[Ctrl+C] 退出')
except KeyboardInterrupt:
    print('\n[Ctrl+C] 中断，停在当前位置')
finally:
    drv.disable()
    conn.close()
    viewer.close()
    print('已下伺服并断开连接')
