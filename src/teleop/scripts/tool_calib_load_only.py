#!/usr/bin/env python3
"""Collect one mounted-tool identification pass using a saved no-load pass.

This is for the case where the tool cannot be removed now.  The supplied
NoLoadData.csv is copied unchanged; only the mounted-tool PVT pass is
collected live.  The arm is moved to all-zero joints immediately before the
new pass, matching tool_calib_collect.py's zero-point requirement.
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from drivers.arm_driver import (  # noqa: E402
    ArmDriver, RobotConnection, ERR_CODE_CN, ensure_clear,
)
from scripts.tool_calib_collect import (  # noqa: E402
    _CFG_FILE_SRC, _PVT_FILE, _collect_pass, _goto_zero,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['A', 'B'])
    ap.add_argument('--tool', required=True)
    ap.add_argument('--no-load', required=True,
                    help='已有的历史 NoLoadData.csv')
    ap.add_argument('--out-dir', required=True,
                    help='新目录，写入 NoLoadData.csv/LoadData.csv')
    ap.add_argument('--pvt-id', type=int, default=3)
    ap.add_argument('--duration', type=float, default=60.0)
    ap.add_argument('--speed', type=float, default=8.0)
    args = ap.parse_args()

    if not os.path.isfile(args.no_load):
        raise SystemExit('找不到历史空载数据：%s' % args.no_load)
    pvt_file = _PVT_FILE[args.arm]
    if not os.path.isfile(pvt_file):
        raise SystemExit('找不到 PVT 轨迹：%s' % pvt_file)
    if not os.path.isfile(_CFG_FILE_SRC):
        raise SystemExit('找不到辨识配置：%s' % _CFG_FILE_SRC)

    os.makedirs(args.out_dir, exist_ok=True)
    no_load_path = os.path.join(args.out_dir, 'NoLoadData.csv')
    load_path = os.path.join(args.out_dir, 'LoadData.csv')
    cfg_dst = os.path.join(args.out_dir, 'CfgFile',
                           'LoadIdenCfg_Marvin_CCS.txt')
    os.makedirs(os.path.dirname(cfg_dst), exist_ok=True)
    shutil.copyfile(args.no_load, no_load_path)
    shutil.copyfile(_CFG_FILE_SRC, cfg_dst)

    print('连接 %s…' % config.ROBOT_IP)
    conn = RobotConnection(config.ROBOT_IP)
    print('✅ 控制器版本 %s' % conn.version)
    drv = ArmDriver(conn, args.arm, config)
    try:
        print('臂%s 检查状态/清错…' % args.arm)
        ok, st = ensure_clear(conn, drv, args.arm)
        print('臂%s %s cur=%s err=%s q=%s' % (
            args.arm, '✅ 状态干净' if ok else '❌ 清错后仍有故障',
            st['cur'], st['err'], [round(v, 2) for v in st['q']]))
        if not ok:
            raise RuntimeError('臂%s 清错未成功（cur=%s err=%s(%s)）' % (
                args.arm, st['cur'], st['err'],
                ERR_CODE_CN.get(st['err'], '未知')))
        input('\n=== 新带载采集 ===\n'
              '确认 UMI 工具已安装、机械臂周围空旷；回车后先回全零再开始约 %.0fs PVT 轨迹…'
              % args.duration)
        _goto_zero(conn, drv, args.arm, args.speed)
        _collect_pass(conn, drv, args.arm, pvt_file, args.pvt_id,
                      args.duration, load_path)
        print('\n✅ 新带载数据已保存：%s' % args.out_dir)
    except KeyboardInterrupt:
        print('\n[Ctrl+C] 中断')
    finally:
        drv.disable()
        conn.close()
        print('已下伺服并断开连接')


if __name__ == '__main__':
    main()
