#!/usr/bin/env python3
"""Enter joint-space drag on both Tianji arms through one SDK connection.

This is the two-arm equivalent of ``set_state.py --state drag``.  A single
RobotConnection is intentional: the Marvin UDP port is exclusive.
Ctrl+C exits drag space and disables both arms.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from drivers.arm_driver import (
    ArmDriver, RobotConnection, ensure_clear, send_joint_commands,
    STATE_TORQUE,
)
from drivers.tool_config import load_tool_config


ARMS = ('A', 'B')
DT = 1.0 / 250.0
DRAG_TRACK_RATE = 15.0
DRAG_STIFFNESS = 1.0
DRAG_DAMPING = 0.3
DRAG_TYPE = 1  # joint-space drag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', default='umi',
                        help='tool YAML name under configs/tool/ (default: umi)')
    args = parser.parse_args()
    tool_cfg, _ = load_tool_config(args.tool, 'A',
                                   '/home/jw/Desktop/project/teleop')
    tool_cfg_b, _ = load_tool_config(args.tool, 'B',
                                     '/home/jw/Desktop/project/teleop')
    conn = RobotConnection(config.ROBOT_IP)
    drivers = {arm: ArmDriver(conn, arm, config) for arm in ARMS}
    engaged = []
    drag_enabled = []
    try:
        for arm in ARMS:
            ok, st = ensure_clear(conn, drivers[arm], arm)
            print('臂%s %s cur=%s err=%s q=%s'
                  % (arm, '✅ 状态干净' if ok else '❌ 清错失败',
                     st['cur'], st['err'], [round(v, 2) for v in st['q']]))
            if not ok:
                raise RuntimeError('臂%s 清错未成功，拒绝进入 drag' % arm)

        # Configure both arms in one clear_set/send_cmd batch.
        conn.robot.clear_set()
        for arm, cfg_tool in (('A', tool_cfg), ('B', tool_cfg_b)):
            conn.robot.set_state(arm=arm, state=STATE_TORQUE)
            conn.robot.set_impedance_type(arm=arm, type=1)
            conn.robot.set_tool(arm=arm,
                                kineParams=cfg_tool.kine_params(),
                                dynamicParams=cfg_tool.dynamic_params())
            conn.robot.set_joint_kd_params(
                arm=arm,
                K=[DRAG_STIFFNESS] * 7,
                D=[DRAG_DAMPING] * 7,
            )
        conn.robot.send_cmd()
        time.sleep(0.5)
        for arm in ARMS:
            st = drivers[arm].state()
            if st['cur'] != STATE_TORQUE or st['err']:
                raise RuntimeError('臂%s 切扭矩模式失败：cur=%s err=%s'
                                   % (arm, st['cur'], st['err']))
            drivers[arm].engaged = True
            engaged.append(arm)

        conn.robot.clear_set()
        for arm in ARMS:
            conn.robot.set_drag_space(arm=arm, dgType=DRAG_TYPE)
        conn.robot.send_cmd()
        time.sleep(0.2)
        sub = conn.subscribe()
        for arm in ARMS:
            got = sub['inputs'][drivers[arm].idx]['drag_sp_type']
            if got != DRAG_TYPE:
                raise RuntimeError('臂%s drag 类型回读失败：%s' % (arm, got))
            drag_enabled.append(arm)
        print('✅ A/B 均已进入 joint drag（dgType=1，tool=%s）。按 Ctrl+C 退出.'
              % args.tool)

        max_step = DRAG_TRACK_RATE * DT
        q_cmd = {arm: list(drivers[arm].joints()) for arm in ARMS}
        while True:
            measured = {arm: drivers[arm].joints() for arm in ARMS}
            q_cmd = {
                arm: [c + max(-max_step, min(max_step, m - c))
                      for c, m in zip(q_cmd[arm], measured[arm])]
                for arm in ARMS
            }
            send_joint_commands(conn, q_cmd)
            time.sleep(DT)
    except KeyboardInterrupt:
        print('\n[Ctrl+C] 退出 drag')
    finally:
        if drag_enabled:
            try:
                conn.robot.clear_set()
                for arm in drag_enabled:
                    conn.robot.set_drag_space(arm=arm, dgType=0)
                conn.robot.send_cmd()
                time.sleep(0.5)
            except Exception as exc:
                print('⚠️ 退出 drag 空间失败：%s' % exc)
        for arm in engaged:
            drivers[arm].disable()
        conn.close()
        print('已退出 drag、下伺服并断开连接')


if __name__ == '__main__':
    main()
