#!/usr/bin/env python3
"""Emergency/standalone homing -- move configured arms to config.HOME_JOINTS
without starting a full run_teleop.py session. Same abort-on-tracking-error
path as core/homing.py's end-of-run homing, at constant config.HOME_SPEED_DEG_S.

Usage:
    python3 scripts/home_now.py            # homes all arms in config.ARMS
    python3 scripts/home_now.py --arms A
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                                    # noqa: E402
from drivers.arm_driver import (ArmDriver, RobotConnection,      # noqa: E402
                                ERR_CODE_CN, ensure_clear, move_to_joints)


ap = argparse.ArgumentParser()
ap.add_argument('--arms', default=None, help='Defaults to config.ARMS.')
args = ap.parse_args()
arms = list(args.arms.upper()) if args.arms else list(config.ARMS)

print('连接 %s…' % config.ROBOT_IP)
conn = RobotConnection(config.ROBOT_IP)
print('✅ 控制器版本 %s' % conn.version)

drivers = {}
try:
    for a in arms:
        d = ArmDriver(conn, a, config)
        drivers[a] = d
        print('臂%s 检查状态/清错…' % a)
        ok, st = ensure_clear(conn, d, a)
        print('臂%s %s cur=%s err=%s q=%s'
             % (a, '✅ 状态干净' if ok else '❌ 清错后仍有故障',
                st['cur'], st['err'], [round(v, 2) for v in st['q']]))
        if not ok:
            raise RuntimeError(
                '臂%s 清错未成功（cur=%s err=%s(%s)）—— 不去尝试切模式/归位，'
                '先用 MarvinPlatform GUI 或 tools/wait_ready.py 排查'
                % (a, st['cur'], st['err'], ERR_CODE_CN.get(st['err'], '未知')))

    for a, d in drivers.items():
        print('\n臂%s 切换到位置模式…' % a)
        d.prepare()

    targets = {a: list(config.HOME_JOINTS[a]) for a in arms
              if a in config.HOME_JOINTS}
    missing = [a for a in arms if a not in config.HOME_JOINTS]
    if missing:
        print('⚠️  %s 没有配 HOME_JOINTS，跳过' % missing)

    if targets:
        waypoints = getattr(config, 'HOME_WAYPOINTS', {})
        n_stages = max((len(waypoints.get(a, [])) for a in targets), default=0)
        stages = []
        for i in range(n_stages):
            stage = {a: list(waypoints[a][i]) for a in targets
                     if i < len(waypoints.get(a, []))}
            if stage:
                stages.append(('Medium %d/%d (%s)' % (i + 1, n_stages,
                                                       ','.join(stage)), stage))
        stages.append(('Final', targets))

        print('\n=== 归位 (峰值 %.0f°/s) ===' % config.HOME_SPEED_DEG_S)
        for label, stage_targets in stages:
            print('\n-- %s --' % label)
            ok, why, q_end = move_to_joints(conn, drivers, stage_targets,
                                            speed_deg_s=config.HOME_SPEED_DEG_S if label!="Final" else 2*config.HOME_SPEED_DEG_S,
                                            hz=config.CONTROL_HZ,
                                            max_err_deg=config.MAX_TRACKING_ERR_DEG)
            for a in stage_targets:
                q = drivers[a].joints()
                print('臂%s 现构型 %s' % (a, [round(v, 2) for v in q]))
            if not ok:
                print('⚠️  归位未完成（%s，卡在"%s"这一段）—— 已停在当前位置，未强行继续'
                     % (why, label))
                break
        else:
            print('✅ 已回到归位构型')
    else:
        print('没有可归位的臂')

finally:
    for d in drivers.values():
        d.disable()
    conn.close()
    print('\n已下伺服，连接已断开')
