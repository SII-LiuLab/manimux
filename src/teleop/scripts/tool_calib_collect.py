#!/usr/bin/env python3
"""Collect no-load / load PVT identification data for
Marvin_Kine.identify_tool_dyn() -- steps 1+2 of the 3-step tool-parameter
calibration pipeline described in docs/scripts.md's gravity-compensation
section. Step 3 (offline identify -> configs/tool/<name>.yaml numbers) is
scripts/tool_calib_identify.py.

Mirrors DEMO_PYTHON/showcase_identy_tool_dynamic_CCS_B.py's
collect_identy_data(), run twice (no-load, then load), wrapped in this
project's ArmDriver/RobotConnection instead of the demo's bare globals,
plus two deliberate deviations:
  - moves to all-zero joints before EACH pass -- a direct point-to-point
    move (goto_joints.py's move_to_joints() call, own --speed, own
    STATE_POSITION prepare()), deliberately NOT core/homing.py's "home"
    concept (config.HOME_JOINTS parking pose / config.HOME_SPEED_DEG_S):
    the two are unrelated poses/speeds that happen to share a verb in
    English. Same, trivially-reproducible all-zero reference configuration
    for both passes -- an explicit requirement for this pipeline, not
    something the vendor demo itself enforces (its own IdenTraj .fmv files
    start mid-air, e.g. [0,0,-90,-10,0,0,0], not zero).
  - runs both passes in one invocation with a pause in between to
    mount/remove the tool, instead of the demo's "uncomment one block, run
    it, re-comment, uncomment the next" workflow across 3 separate manual
    launches.

Also washes collect_data()'s raw '$'-delimited dump into the plain numeric
CSV identify_tool_dyn() expects (same wash as the demo's
collect_identy_data()), and copies the SDK's shipped identification config
file alongside the two CSVs, in the exact layout identify_tool_dyn() was
strace'd to expect (see docs/scripts.md): <dir>/LoadData.csv,
<dir>/NoLoadData.csv, <dir>/CfgFile/LoadIdenCfg_Marvin_CCS.txt.

First time driving PVT mode from this project's software -- same elevated-
risk category as scripts/set_state.py --state drag/release: the arm runs a
~1-minute trajectory under its own PVT-internal speed/accel, not this
script's rate limiting. Watch the arm for the whole pass; Ctrl+C aborts the
current pass and disables (does not clean up a mid-flight PVT run
server-side beyond stop_collect_data()).

Usage:
    python3 scripts/tool_calib_collect.py --arm B --tool omnigripper
    python3 scripts/tool_calib_collect.py --arm A --tool omnigripper --pvt-id 5 --duration 75

Output:
    data/tool_calib/<tool>/<arm>/{NoLoadData.csv, LoadData.csv,
    CfgFile/LoadIdenCfg_Marvin_CCS.txt}
    -> feed that directory to scripts/tool_calib_identify.py.
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                                     # noqa: E402
from drivers.arm_driver import (ArmDriver, RobotConnection,       # noqa: E402
                                ERR_CODE_CN, STATE_POSITION, STATE_PVT,
                                ensure_clear, move_to_joints)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SDK = os.environ.get('MARVIN_SDK', '/home/jw/Downloads/TJ_FX_ROBOT_CONTRL_SDK')
_IDEN_TRAJ_DIR = os.path.join(_SDK, 'CommonConfig', 'LoadData_ccs', 'LoadData',
                              'IdenTraj')
_CFG_FILE_SRC = os.path.join(_SDK, 'CommonConfig', 'LoadData_ccs', 'LoadData',
                             'CfgFile', 'LoadIdenCfg_Marvin_CCS.txt')

# robot_type=1 (CCS) is this project's only machine type -- config.KINE_CFG
# points at ccs_m6_40.MvKDCfg. robot_type=2 (SRS) is a different machine,
# out of scope here (see docs/scripts.md).
_PVT_FILE = {
    'A': os.path.join(_IDEN_TRAJ_DIR, 'LoadIdenTraj_MarvinCCS_Left.fmv'),
    'B': os.path.join(_IDEN_TRAJ_DIR, 'LoadIdenTraj_MarvinCCS_Right.fmv'),
}

# collect_data() targetID layout: 7 joint positions + 7 joint-sensor torques
# (NM) + 1 tag column, zero-padded to the fixed 35-slot array (targetNum=15
# tells the controller only the first 15 slots are real). Matches
# DEMO_PYTHON/showcase_identy_tool_dynamic_CCS_B.py's collect_identy_data().
_COLS = 15
_TAG_IDX_LEGACY = 76
_TAG_IDX_NEW = 66
_TAG_VERSION_THRESHOLD = 100343007


def _tag_idx(robot):
    version = robot.SDK_version()
    return _TAG_IDX_NEW if version > _TAG_VERSION_THRESHOLD else _TAG_IDX_LEGACY


def _target_ids(arm, tag):
    if arm == 'A':
        pos, trq = range(0, 7), range(50, 57)
    else:
        pos, trq = range(100, 107), range(150, 157)
        tag += 100
    ids = list(pos) + list(trq) + [tag]
    return ids + [0] * (35 - len(ids))


def _wash_csv(path):
    """In-place: raw '$'-delimited collect_data() dump -> plain numeric CSV.
    Drops the header line and the first two (non-payload) columns per row --
    same wash as the vendor demo's collect_identy_data()."""
    with open(path) as fh:
        lines = fh.readlines()[1:]
    rows = []
    for line in lines:
        numbers = [part.split()[-1] for part in line.strip().split('$') if part]
        if len(numbers) >= 2:
            numbers = numbers[2:]
        rows.append(numbers)
    with open(path, 'w') as fh:
        for row in rows:
            fh.write(','.join(row) + '\n')


def _goto_zero(conn, drv, arm, speed_deg_s):
    """Direct point-to-point move to all-zero joints -- same primitive as
    goto_joints.py --to 0,0,0,0,0,0,0, not core/homing.py's staged "home to
    config.HOME_JOINTS" concept (unrelated pose, unrelated speed constant)."""
    print('\n臂%s 移动到零位（位置模式，7 关节 -> 0）…' % arm)
    config.ARM_STATE = STATE_POSITION
    drv.prepare()
    ok, why, _ = move_to_joints(conn, {arm: drv}, {arm: [0.0] * 7},
                                speed_deg_s=speed_deg_s,
                                hz=config.CONTROL_HZ,
                                max_err_deg=config.MAX_TRACKING_ERR_DEG)
    if not ok:
        raise RuntimeError('臂%s 移动到零位失败：%s' % (arm, why))
    print('✅ 臂%s 已到零位：%s' % (arm, [round(v, 3) for v in drv.joints()]))


def _collect_pass(conn, drv, arm, pvt_file, pvt_id, duration, out_path):
    """One PVT identification pass -> raw dump at out_path, washed into
    plain numeric CSV in place. Caller must have already moved to zero
    (_goto_zero) and confirmed the tool state (mounted/removed) for this
    pass."""
    robot = conn.robot

    # raw set_state() call, same as set_state.py's drag/release branches --
    # without drv.engaged=True, the outer `finally: drv.disable()` silently
    # no-ops.
    drv.engaged = True
    # set_state() is asynchronous and its UDP command can go unapplied --
    # ArmDriver.disable() hits the same thing and resends up to 3x with a
    # 0.3s wait + readback (see docs/drivers.md), not just one send + a
    # longer wait. Same pattern here, one extra retry since PVT has more
    # riding on it (skipping straight to disable() abandons a ~60s pass).
    st = None
    for attempt in range(5):
        robot.clear_set()
        robot.set_state(arm=arm, state=STATE_PVT)
        robot.send_cmd()
        time.sleep(0.3)
        st = drv.state()
        if st['cur'] == STATE_PVT:
            break
        print('  臂%s 切 PVT 未生效（cur=%s err=%s）—— 重发 (第%d次)'
             % (arm, st['cur'], st['err'], attempt + 1))
    if st['cur'] != STATE_PVT:
        raise RuntimeError('臂%s 切 PVT 模式失败：期望 %s 实得 %s（err=%s）'
                           % (arm, STATE_PVT, st['cur'], st['err']))

    robot.send_pvt_file(arm, pvt_file, pvt_id)
    print('已上传 PVT 轨迹：%s (id=%d)' % (pvt_file, pvt_id))
    time.sleep(0.5)

    robot.clear_set()
    robot.collect_data(targetNum=_COLS, targetID=_target_ids(arm, _tag_idx(robot)),
                       recordNum=1_000_000)
    robot.send_cmd()
    time.sleep(0.5)

    robot.clear_set()
    robot.set_pvt_id(arm, pvt_id)
    robot.send_cmd()
    print('⚠️  轨迹运行中（约 %.0fs，机械臂自行运动，不受本脚本限速）——'
         '全程观察，确认没有异常' % duration)
    time.sleep(duration)

    robot.clear_set()
    robot.stop_collect_data()
    robot.send_cmd()
    time.sleep(0.5)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    robot.save_collected_data_to_path(out_path)
    time.sleep(1.0)
    _wash_csv(out_path)
    print('✅ 已保存并清洗：%s' % out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['A', 'B'])
    ap.add_argument('--tool', required=True,
                    help='工具名（决定输出目录 data/tool_calib/<tool>/<arm>/，'
                         '应与之后 configs/tool/<tool>.yaml 的 name 对应）')
    ap.add_argument('--pvt-id', type=int, default=3,
                    help='PVT 轨迹 id，1-99（默认 3，同 vendor demo）')
    ap.add_argument('--duration', type=float, default=60.0,
                    help='单次轨迹采集等待时长，秒（默认 60，同 vendor demo '
                         '注释里的轨迹时长；轨迹尚未走完就停止采集/切模式会'
                         '导致辨识失败或数据不全，宁可偏大）')
    ap.add_argument('--out-dir', default=None,
                    help='默认 data/tool_calib/<tool>/<arm>/')
    ap.add_argument('--speed', type=float, default=8.0,
                    help='两次采集前，直线点到点移动到零位的速度，deg/s（默认 8，'
                         '同 goto_joints.py 默认值；与 config.HOME_SPEED_DEG_S 无关，'
                         '零位不是 config.HOME_JOINTS 的归位姿态）')
    args = ap.parse_args()

    pvt_file = _PVT_FILE[args.arm]
    if not os.path.exists(pvt_file):
        raise SystemExit('❌ 找不到 PVT 轨迹文件：%s（检查 MARVIN_SDK 环境变量）'
                         % pvt_file)
    if not os.path.exists(_CFG_FILE_SRC):
        raise SystemExit('❌ 找不到 SDK 自带的辨识配置文件：%s' % _CFG_FILE_SRC)

    out_dir = args.out_dir or os.path.join(_REPO_ROOT, 'data', 'tool_calib',
                                           args.tool, args.arm)
    no_load_path = os.path.join(out_dir, 'NoLoadData.csv')
    load_path = os.path.join(out_dir, 'LoadData.csv')
    cfg_dst = os.path.join(out_dir, 'CfgFile', 'LoadIdenCfg_Marvin_CCS.txt')

    print('连接 %s…' % config.ROBOT_IP)
    conn = RobotConnection(config.ROBOT_IP)
    print('✅ 控制器版本 %s' % conn.version)

    drv = ArmDriver(conn, args.arm, config)
    try:
        print('臂%s 检查状态/清错…' % args.arm)
        ok, st = ensure_clear(conn, drv, args.arm)
        print('臂%s %s cur=%s err=%s q=%s'
             % (args.arm, '✅ 状态干净' if ok else '❌ 清错后仍有故障',
                st['cur'], st['err'], [round(v, 2) for v in st['q']]))
        if not ok:
            raise RuntimeError(
                '臂%s 清错未成功（cur=%s err=%s(%s)）—— 不去尝试切模式/采集，'
                '先用 MarvinPlatform GUI 或 tools/wait_ready.py 排查'
                % (args.arm, st['cur'], st['err'],
                   ERR_CODE_CN.get(st['err'], '未知')))

        input('\n=== 第 1/2 步：空载数据 ===\n'
             '确认工具 (%s) 已从法兰卸下，机械臂周围空旷。回车继续…' % args.tool)
        _goto_zero(conn, drv, args.arm, args.speed)
        _collect_pass(conn, drv, args.arm, pvt_file, args.pvt_id, args.duration,
                     no_load_path)

        input('\n=== 第 2/2 步：带载数据 ===\n'
             '确认工具 (%s) 已装回法兰，机械臂周围空旷。回车继续…' % args.tool)
        _goto_zero(conn, drv, args.arm, args.speed)
        _collect_pass(conn, drv, args.arm, pvt_file, args.pvt_id, args.duration,
                     load_path)

        os.makedirs(os.path.dirname(cfg_dst), exist_ok=True)
        shutil.copyfile(_CFG_FILE_SRC, cfg_dst)

        print('\n✅ 采集完成：%s' % out_dir)
        print('下一步（离线辨识，无需连接机器人）：\n'
             '  python3 scripts/tool_calib_identify.py --dir %s --arm %s --tool %s'
             % (out_dir, args.arm, args.tool))
    except KeyboardInterrupt:
        print('\n[Ctrl+C] 中断')
    finally:
        drv.disable()
        conn.close()
        print('已下伺服并断开连接')


if __name__ == '__main__':
    main()
