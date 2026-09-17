#!/usr/bin/env python3
"""Restore a recorded diffusion-policy start pose from a YAML in data/.

Same multi-arm move path as scripts/home_now.py (one move_to_joints() call,
both arms interpolated together, abort on tracking error), except the target
comes from a pose file instead of config.HOME_JOINTS. Unlike home_now.py
there are no intermediate waypoints -- the move is a straight joint-space
line from wherever the arms are now, so read the printed per-joint deltas
before confirming.

Pose file format (see data/dp_start_20260910.yaml):
    name: dp_start_20260910
    arms:
      A:
        joints_deg: [7 angles, degrees, project A/B joint order]
        state_at_record: disabled | error | ...   # optional, informational
        error_at_record: 0                        # optional, informational
        error_description: servo_fault            # optional, informational

`state_at_record`/`error_at_record` describe the arm when the pose was
sampled, NOT a precondition -- a pose recorded off a faulted arm is still a
valid target, but it is worth knowing about, so it is echoed as a warning.

Usage:
    python3 scripts/goto_dp_start.py                      # data/dp_start_20260910.yaml, both arms
    python3 scripts/goto_dp_start.py --arms A
    python3 scripts/goto_dp_start.py --pose dp_start_20260910
    python3 scripts/goto_dp_start.py --pose data/other.yaml --speed 3
    python3 scripts/goto_dp_start.py --yes                # skip the confirmation prompt
    python3 scripts/goto_dp_start.py --hold               # stay enabled until Ctrl+C
"""
import argparse
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                                    # noqa: E402
from drivers.arm_driver import (ArmDriver, RobotConnection,      # noqa: E402
                                ERR_CODE_CN, ensure_clear, move_to_joints,
                                send_joint_commands)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_POSE = 'dp_start_20260910'


def resolve_pose_path(pose):
    """Accept a bare name (-> data/<name>.yaml) or an explicit path."""
    if os.sep in pose or pose.endswith(('.yaml', '.yml')):
        return pose if os.path.isabs(pose) else os.path.join(_REPO_ROOT, pose)
    return os.path.join(_REPO_ROOT, 'data', '%s.yaml' % pose)


def load_pose(path):
    """:return: (name, {arm: {'joints_deg': [...7...], ...}}) -- validated."""
    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        raise SystemExit('❌ 起始构型文件不存在：%s' % path)
    arms = raw.get('arms')
    if not isinstance(arms, dict) or not arms:
        raise SystemExit('❌ %s 里没有 arms: 段落' % path)
    for a, entry in arms.items():
        if not isinstance(entry, dict) or 'joints_deg' not in entry:
            raise SystemExit('❌ %s 的臂%s 缺少 joints_deg' % (path, a))
        q = entry['joints_deg']
        if not isinstance(q, list) or len(q) != 7:
            raise SystemExit('❌ %s 的臂%s joints_deg 需要正好 7 个角度，收到 %s'
                             % (path, a, len(q) if isinstance(q, list) else type(q).__name__))
        try:
            entry['joints_deg'] = [float(v) for v in q]
        except (TypeError, ValueError):
            raise SystemExit('❌ %s 的臂%s joints_deg 含非数值：%s' % (path, a, q))
    return raw.get('name') or os.path.basename(path), arms


ap = argparse.ArgumentParser()
ap.add_argument('--pose', default=DEFAULT_POSE,
                help='data/ 下的构型名，或 yaml 路径。默认 %s' % DEFAULT_POSE)
ap.add_argument('--arms', default=None,
                help='默认取构型文件里有、且在 config.ARMS 里的臂。')
ap.add_argument('--speed', type=float, default=config.HOME_SPEED_DEG_S,
                help='最大关节速度 deg/s，默认 config.HOME_SPEED_DEG_S=%.1f'
                     % config.HOME_SPEED_DEG_S)
ap.add_argument('--yes', action='store_true', help='跳过确认，直接移动。')
ap.add_argument('--hold', action='store_true',
                help='到位后保持使能，按 Ctrl+C 退出（默认到位即下伺服，'
                     '手臂抱死保持姿态）。')
args = ap.parse_args()

pose_path = resolve_pose_path(args.pose)
pose_name, pose_arms = load_pose(pose_path)

if args.arms:
    arms = list(args.arms.upper())
    missing = [a for a in arms if a not in pose_arms]
    if missing:
        raise SystemExit('❌ %s 里没有臂 %s（有：%s）'
                         % (pose_path, missing, sorted(pose_arms)))
else:
    arms = [a for a in config.ARMS if a in pose_arms]
    if not arms:
        raise SystemExit('❌ %s 里的臂 %s 都不在 config.ARMS=%s 里'
                         % (pose_path, sorted(pose_arms), config.ARMS))

targets = {a: list(pose_arms[a]['joints_deg']) for a in arms}

print('起始构型 %s（%s）' % (pose_name, pose_path))
for a in arms:
    entry = pose_arms[a]
    print('  臂%s -> %s' % (a, targets[a]))
    if entry.get('state_at_record') not in (None, 'disabled'):
        print('     ⚠️  记录时该臂状态 state_at_record=%s err=%s%s —— 只是记录，'
              '不影响目标本身，但说明这个姿势是在异常状态下采的'
              % (entry.get('state_at_record'), entry.get('error_at_record'),
                 '(%s)' % entry['error_description']
                 if entry.get('error_description') else ''))

print('\n连接 %s…' % config.ROBOT_IP)
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
                '臂%s 清错未成功（cur=%s err=%s(%s)）—— 不去尝试切模式/移动，'
                '先用 MarvinPlatform GUI 或 tools/wait_ready.py 排查'
                % (a, st['cur'], st['err'], ERR_CODE_CN.get(st['err'], '未知')))

    # No waypoints here (unlike home_now.py): show every delta so the operator
    # can judge the straight-line path themselves before it starts moving.
    dmax = 0.0
    print()
    for a in arms:
        q = drivers[a].joints()
        d = [t - c for t, c in zip(targets[a], q)]
        dmax = max(dmax, max(abs(x) for x in d))
        print('臂%s 当前 %s' % (a, [round(v, 2) for v in q]))
        print('臂%s 目标 %s' % (a, [round(v, 2) for v in targets[a]]))
        print('臂%s 差值 %s' % (a, [round(v, 2) for v in d]))
    print('\n最大单轴 %.2f°，按 %.1f°/s 约需 %.1f 秒（直线插补，无中间路点）'
          % (dmax, args.speed, dmax / args.speed if args.speed > 0 else float('inf')))

    if not args.yes:
        try:
            if input('回车确认，其它键取消：').strip():
                raise SystemExit('已取消')
        except EOFError:
            raise SystemExit('❌ 非交互环境且未加 --yes，取消')

    for a in arms:
        print('\n臂%s 切换到位置模式…' % a)
        drivers[a].prepare()

    print('\n=== 移动到 %s (峰值 %.1f°/s) ===' % (pose_name, args.speed))
    ok, why, q_end = move_to_joints(conn, drivers, targets,
                                    speed_deg_s=args.speed,
                                    hz=config.CONTROL_HZ,
                                    max_err_deg=config.MAX_TRACKING_ERR_DEG)
    for a in arms:
        q = drivers[a].joints()
        print('臂%s 现构型 %s' % (a, [round(v, 2) for v in q]))
        print('臂%s 与目标偏差 %s'
              % (a, [round(t - m, 2) for t, m in zip(targets[a], q)]))
    if not ok:
        print('\n⚠️  未走到位（%s）—— 已停在当前位置，未强行继续' % why)
    else:
        print('\n✅ 已到达 %s' % pose_name)

    if args.hold:
        # hold at q_end (actual stop point), not targets -- avoids snapping
        # forward if the move aborted partway.
        q_cmd = {a: list(q_end[a]) for a in arms}
        print('\n>>> 保持使能中。按 Ctrl+C 退出'
              '（退出后会抱死保持姿态，不会下垂）')
        dt = 1.0 / config.CONTROL_HZ
        try:
            while True:
                send_joint_commands(conn, q_cmd)
                time.sleep(dt)
        except KeyboardInterrupt:
            print('\n[Ctrl+C] 退出')
except KeyboardInterrupt:
    print('\n[Ctrl+C] 中断，停在当前位置')
finally:
    for d in drivers.values():
        d.disable()
    conn.close()
    print('\n已下伺服，连接已断开')
