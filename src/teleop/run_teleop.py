#!/usr/bin/env python3
"""Teleop main program -- wires xr_source / retarget / ik_solver / safety / arm_driver together.

Threads (do not collapse into one loop):
    SDK callback   90Hz   XR data -> slot (xr_source)
    Control loop  250Hz   frame -> retarget -> IK -> gate -> send
    Gripper        50Hz   slew-rate limit + protection (gripper.GripperThread)

Layout:
    config.py             global config
    algos/                 IK / nullspace / retarget / safety gate / diff_ik
    algos/solver_config.py  typed yaml config for diff_ik, impedance
    configs/solver/        yaml tuning profiles
    drivers/               arm SDK, gripper, XR source
    core/                  ArmChannel, home_arms
    scripts/               standalone calibration/debug tools
    replay.py              offline trajectory replay
    bench/compare_ik.py    analytic vs diff_ik comparison
    viewer_bridge.py        optional universal_viewer mirror (--viewer)

See docs/core.md for design rationale.

Usage:
    python3 replay.py data/traj_b.jsonl               # no robot, sanity check first
    python3 run_teleop.py --dry-run                    # robot connected, nothing sent
    python3 run_teleop.py --arms A --scale 0.2 --duration 60
    python3 run_teleop.py --dry-run --solver diff       # diff-IK prototype, see docs/algos.md#diff_ikpy
    python3 run_teleop.py --arms A --solver diff --control-mode impedance \
        --impedance-type cartesian                      # docs/algos.md#impedanceconfig
    python3 run_teleop.py --arms A --viewer              # mirror to universal_viewer, see docs/viewer.md

Refuses to leave --dry-run while config.CALIBRATED=False.
"""
import argparse
import os
import sys
import time

import config
from algos.retarget import ENGAGED
from algos.solver_config import (DiffIKConfig, ImpedanceConfig,
                                 load_solver_config)
from core.arm_channel import ArmChannel
from core.homing import home_arms
from drivers.arm_driver import ControlLoop, RobotConnection, STATE_TORQUE
from drivers.xr_source import XRSource
from viewer_bridge import ViewerBridge

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# --solver (how q is computed) and --control-mode (how q is executed) are orthogonal.
_SOLVER_CONFIG_MODELS = {'diff': DiffIKConfig}
_CONTROL_MODE_CONFIG_MODELS = {'impedance': ImpedanceConfig}
_IMPEDANCE_TYPE_VALUE = {'joint': 1, 'cartesian': 2}  # -> ImpedanceConfig.type


def _load_typed_config(label, model_cls, cfg_path):
    """Load+validate a solver YAML config; prints and returns None on failure."""
    try:
        cfg = load_solver_config(model_cls, cfg_path)
    except FileNotFoundError:
        print('❌ %s配置文件不存在：%s' % (label, cfg_path))
        return None
    except Exception as e:   # pydantic.ValidationError et al.
        print('❌ %s配置校验失败（%s）：\n%s' % (label, cfg_path, e))
        return None
    print('%s配置: %s\n%s\n' % (label, cfg_path, cfg.model_dump()))
    return cfg


def _duration(s):
    """--duration type: float seconds, or 'forever'."""
    if s.strip().lower() == 'forever':
        return None
    return float(s)


def main():
    ap = argparse.ArgumentParser(description='PICO -> Tianji dual-arm teleop')
    ap.add_argument('--dry-run', action='store_true',
                    help='Run without sending commands.')
    ap.add_argument('--arms', default=None, help='Arms to control, e.g. A or AB.')
    ap.add_argument('--scale', type=float, default=None)
    ap.add_argument('--vel-ratio', type=int, default=None,
                    help='Velocity percent; rescales derived limits too. '
                         'Ramp up gradually.')
    ap.add_argument('--acc-ratio', type=int, default=None,
                    help='Acceleration percent (independent of --vel-ratio).')
    ap.add_argument('--duration', type=_duration, default=None,
                    help="Seconds to run, or 'forever' (default).")
    ap.add_argument('--gripper', action='store_true', help='Enable the gripper.')
    ap.add_argument('--no-home', action='store_true',
                    help='Skip homing after the run.')
    ap.add_argument('--force-uncalibrated', action='store_true',
                    help='Allow live commands while uncalibrated (dangerous).')
    ap.add_argument('--solver', choices=['analytic', 'diff'], default='analytic',
                    help='analytic (default) or diff (QP prototype, '
                         'docs/algos.md#diff_ikpy).')
    ap.add_argument('--solver-config', default=None,
                    help='Solver YAML path (default configs/solver/<solver>.yaml).')
    ap.add_argument('--control-mode', choices=['position', 'impedance'],
                    default='position',
                    help='position (default) or impedance '
                         '(docs/algos.md#impedanceconfig; --dry-run skips '
                         'its SDK calls).')
    ap.add_argument('--impedance-type', choices=['joint', 'cartesian'],
                    default='joint',
                    help='joint (default) or cartesian; selects '
                         '--impedance-config default path only.')
    ap.add_argument('--impedance-config', default=None,
                    help='Impedance YAML path (default '
                         'configs/solver/impedance_<impedance-type>.yaml).')
    ap.add_argument('--viewer', action='store_true',
                    help='Mirror live joint state to universal_viewer (observe '
                         'only -- see docs/viewer.md). Off by default.')
    ap.add_argument('--viewer-endpoint', default='tcp://127.0.0.1:5568')
    args = ap.parse_args()

    if args.arms:
        config.ARMS = list(args.arms.upper())
    if args.scale is not None:
        config.SCALE = args.scale
    if args.vel_ratio is not None or args.acc_ratio is not None:
        # recomputes 3 derived limits together
        config.apply_vel_ratio(args.vel_ratio or config.VEL_RATIO,
                               args.acc_ratio)
    if args.gripper:
        config.GRIPPER_ENABLED = True

    solver_cfg = None
    if args.solver != 'analytic':
        model_cls = _SOLVER_CONFIG_MODELS[args.solver]
        cfg_path = args.solver_config or os.path.join(
            _REPO_ROOT, 'configs', 'solver', '%s.yaml' % args.solver)
        solver_cfg = _load_typed_config('求解器(%s)' % args.solver,
                                        model_cls, cfg_path)
        if solver_cfg is None:
            return 2

    impedance_cfg = None
    if args.control_mode == 'impedance':
        model_cls = _CONTROL_MODE_CONFIG_MODELS[args.control_mode]
        cfg_path = args.impedance_config or os.path.join(
            _REPO_ROOT, 'configs', 'solver',
            'impedance_%s.yaml' % args.impedance_type)
        impedance_cfg = _load_typed_config('阻抗(%s)' % args.impedance_type,
                                           model_cls, cfg_path)
        if impedance_cfg is None:
            return 2
        # catches stale/hand-edited yaml before it reaches hardware
        expect_type = _IMPEDANCE_TYPE_VALUE[args.impedance_type]
        if impedance_cfg.type != expect_type:
            print('❌ --impedance-type %s 要求 yaml 里 type=%d，实际读到 %d'
                  '（%s）—— 文件和命令行选的不是同一种阻抗，拒绝启动'
                  % (args.impedance_type, expect_type, impedance_cfg.type,
                     cfg_path))
            return 2
        # ArmDriver.prepare() checks config.ARM_STATE, same pattern as --vel-ratio.
        config.ARM_STATE = STATE_TORQUE

    dry = args.dry_run
    if not dry and not config.CALIBRATED and not args.force_uncalibrated:
        print('❌ config.CALIBRATED=False —— 坐标系还没标定，拒绝真实下发。\n'
              '   未标定就上机，机器人会往你没预期的方向走。\n'
              '   先完成 Session B（XR 侧轴向）和 Session C（基座侧轴向），\n'
              '   把结果填进 config.AXIS_MAP 并置 CALIBRATED=True。\n'
              '   只想验证逻辑就加 --dry-run。')
        return 2

    config.summary()
    print('模式: %s\n' % ('dry-run（不下发）' if dry else '⚠️ 真实下发'))

    print('连接 XR…')
    src = XRSource()
    src.start()
    if not src.wait_for_data(15.0):
        print('❌ 收不到 XR 数据。检查 runService.sh 是否在跑、头显应用是否打开 Send')
        src.stop()
        return 1
    print('✅ XR 数据正常')

    print('连接机器人 %s…' % config.ROBOT_IP)
    conn = RobotConnection(config.ROBOT_IP, dry_run=dry)
    print('✅ 控制器版本 %s' % conn.version)
    conn.check_and_clear_errors()

    viewer = ViewerBridge(args.viewer, args.viewer_endpoint)
    if args.viewer:
        print('✅ 查看器镜像已开启 -> %s（仅观察，不接收 pause/home）' % args.viewer_endpoint)

    channels = {}
    hand_of_arm = {v: k for k, v in config.ARM_OF_HAND.items()}
    for arm in config.ARMS:
        channels[arm] = ArmChannel(arm, hand_of_arm[arm], conn, config,
                                   solver=args.solver,
                                   diff_ik_config=solver_cfg,
                                   impedance_config=impedance_cfg)

    grip_thread = None
    try:
        for ch in channels.values():
            ch.prepare()

        if config.GRIPPER_ENABLED:
            from drivers.gripper import Gripper, GripperThread
            grip_thread = GripperThread(Gripper(conn.robot), config,
                                        config.ARMS)
            grip_thread.start()
            print('✅ 夹爪线程已启动')

        drivers = {a: c.driver for a, c in channels.items()}
        loop = ControlLoop(conn, drivers, config.CONTROL_HZ)
        last_report = [time.monotonic()]
        viewer_step = [0]

        def step(dt):
            frame = src.latest()
            age = src.age_s()
            state = conn.subscribe()
            # already fetched above for q_meas -- publish_state() throttles
            # its own send rate, so this costs nothing extra on off-ticks.
            viewer.publish_state(state, step=viewer_step[0])
            viewer_step[0] += 1
            cmds = {}
            for arm, ch in channels.items():
                idx = 0 if arm == 'A' else 1
                q_meas = list(state['outputs'][idx]['fb_joint_pos'])
                q = ch.step(frame, age, dt, q_meas)
                if q is not None:
                    cmds[arm] = q
                if (config.GRIPPER_ENABLED and grip_thread and frame
                        and ch.rt.state == ENGAGED):
                    grip_thread.set_target(
                        arm, frame.button(ch.hand, config.GRIPPER_AXIS, 0.0))

            if grip_thread and grip_thread.fault:
                print('❌ 夹爪保护触发：%s' % grip_thread.fault)
                loop.running = False

            now = time.monotonic()
            if now - last_report[0] >= 2.0:
                last_report[0] = now
                bits = ['XR %.0fms' % (age * 1e3)]
                for arm, ch in channels.items():
                    idx = 0 if arm == 'A' else 1
                    vr = state['inputs'][idx]['joint_vel_ratio']  # actual controller vel%
                    bits.append('%s:%s err %.2f°/%.1f°'
                                % (arm, ch.blocked_reason or '跟随中',
                                   ch.track_err, config.MAX_TRACKING_ERR_DEG))
                    if 1 <= vr <= 100 and vr != config.VEL_RATIO:
                        bits.append('⚠️vel 回读 %d%%≠%d%%'
                                    % (vr, config.VEL_RATIO))
                print('  ' + '  '.join(bits))
            return cmds

        if args.no_home:
            before_disable = None
        else:
            # Gripper and arm commands share one UDP channel; stop it before homing.
            def before_disable():
                if grip_thread:
                    grip_thread.stop()
                home_arms(conn, channels)

        print('\n开始。按住 %s 进入跟随，松开保持。Ctrl+C 退出。\n'
              % config.CLUTCH_BUTTON)
        stats = loop.run(step, duration=args.duration,
                         before_disable=before_disable)

        print('\n=== 统计 ===')
        print('循环 %d 次，下发 %d 次，超时 %d 次，最大抖动 %.1f ms'
              % (stats['iters'], stats['sent'], stats['overruns'],
                 stats['max_jitter_ms']))
        for arm, ch in channels.items():
            # diff solver never calls ch.ik.solve*(); report diff_solver's own stats.
            if ch.solver == 'diff':
                print('臂%s: 通过安全闸 %d 帧，离合 %d 次，保护闭锁 %d 次，'
                      'diff-IK 统计 %s'
                      % (arm, ch.n_sent, ch.rt.engage_count, ch.rt.fault_count,
                         {k: v for k, v in ch.diff_solver.stats.items() if v}))
                print('     跟踪误差峰值 %.2f° / 上限 %.1f°   '
                      '实际最大帧增量 %.3f° / 额度 %.3f°   钳位 %d 帧   '
                      '背压 %d 帧   单帧求解最大耗时 %.3fms（对照 '
                      'bench/compare_ik.py 离线结果，4ms 预算内正常）'
                      % (ch.track_err_max, config.MAX_TRACKING_ERR_DEG,
                         ch.gate.max_step_used, config.MAX_JOINT_STEP_DEG,
                         ch.gate.stats['clamped'], ch.gate.stats['track_block'],
                         ch.diff_solve_ms_max))
            else:
                print('臂%s: 通过安全闸 %d 帧，离合 %d 次，保护闭锁 %d 次，IK 统计 %s'
                      % (arm, ch.n_sent, ch.rt.engage_count, ch.rt.fault_count,
                         {k: v for k, v in ch.ik.stats.items() if v}))
                print('     跟踪误差峰值 %.2f° / 上限 %.1f°   '
                      '实际最大帧增量 %.3f° / 额度 %.3f°   钳位 %d 帧   '
                      '退让 %d 次   投影 %d 次   背压 %d 帧'
                      % (ch.track_err_max, config.MAX_TRACKING_ERR_DEG,
                         ch.gate.max_step_used, config.MAX_JOINT_STEP_DEG,
                         ch.gate.stats['clamped'], ch.ik.n_backoff,
                         ch.ik.n_projected, ch.gate.stats['track_block']))
            if ch.ns:
                print('     零空间: 介入 %d 帧，探测被抹平 %d 帧，'
                      '最终臂角 %+.1f° / 行程 ±%.0f°'
                      % (ch.ns.n_moved, ch.ns.n_blind, ch.ns.angle,
                         config.ARM_ANGLE_LIMIT))
        # acceptance criterion: zero latched faults
        latched = sum(ch.rt.fault_count for ch in channels.values())
        if latched == 0:
            print('\n✅ 本档（vel %d%%）无保护闭锁 —— 可以升到下一档或加大 scale'
                  % config.VEL_RATIO)
        else:
            print('\n❌ 仍有 %d 次保护闭锁。看上面的「跟踪误差峰值」：\n'
                  '   接近上限 → 指令流还是比控制器快，降 CMD_RATE_MARGIN 或降 scale\n'
                  '   远低于上限 → 是别的原因踢出的（看 IK 统计里的 ik_failed）'
                  % latched)
    finally:
        if grip_thread:
            grip_thread.stop()
        for ch in channels.values():
            ch.driver.disable()
        conn.close()
        src.stop()
        viewer.close()
        print('已下伺服并断开连接')
    return 0


if __name__ == '__main__':
    sys.exit(main())
