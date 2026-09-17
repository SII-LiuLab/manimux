#!/usr/bin/env python3
"""Offline replay — run a recorded XR trajectory through the full pipeline, no robot.

See docs/core.md for why this is the project's most valuable debugging tool
and for the P2 acceptance criteria.

Recording format: `drivers/xr_source.py --record traj.jsonl` (Session B).
    Each line: {"host_ns":..., "dev":..., "state":{raw stateJson}}

Usage:
    python3 replay.py data/traj_b.jsonl                # full replay + report
    python3 replay.py data/traj_b.jsonl --hand right --arm A
    python3 replay.py data/traj_b.jsonl --scale 0.3    # try a different scale
    python3 replay.py --synth                          # self-test with synthetic data

UMI demos (LeRobot v3 episodes from the handheld TacCap rig) go through the
same pipeline, with the source frame and axis map swapped -- see
drivers/umi_source.py:
    python3 replay.py --umi ~/Downloads/replay_test --hand left
    python3 replay.py --umi ~/Downloads/replay_test --hand right --episode 0

--solver picks the IK solver and defaults to whichever one the entry point
being rehearsed for uses: diff (algos/diff_ik.py) with --umi, matching
scripts/umi_replay.py; analytic (algos/ik_solver.py) otherwise, matching
run_teleop.py. Rehearsing one solver and deploying the other would validate
a trajectory nobody is going to execute -- see docs/scripts.md#umi_replaypy.
"""
import argparse
import json
import math
import os
import sys

import config
from algos.diff_ik import build_from_config as build_diff_solver
from algos.ik_solver import ArmIK
from algos.nullspace import JointLimitAvoidance, NullSpaceController
from algos.retarget import Retargeter, ENGAGED, verify_euler_convention
from algos.safety import SafetyGate, LowPass, merge_limit_override
from algos.solver_config import load_diff_ik_config
from drivers.xr_source import XRFrame


def load_traj(path):
    frames = []
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                frames.append(XRFrame(rec.get('dev', ''), rec['state'],
                                      rec['host_ns']))
            except (ValueError, KeyError) as e:
                print('⚠️  第 %d 行解析失败，跳过：%s' % (ln, e))
    return frames


def synth_traj(n=900, hz=90.0):
    """Synthetic controller trajectory for self-testing without a real recording."""
    out = []
    for i in range(n):
        t = i / hz
        # Clutch released for the first/last 10% to exercise the clutch state machine.
        grip = 0.0 if (i < n * 0.1 or i > n * 0.9) else 1.0
        p = (0.30 + 0.10 * math.sin(2 * math.pi * 0.3 * t),
             1.10 + 0.06 * math.cos(2 * math.pi * 0.3 * t),
             -0.25 + 0.04 * math.sin(2 * math.pi * 0.15 * t))
        raw = {
            'timeStampNs': int(t * 1e9),
            'Controller': {
                'right': {'pose': '%f,%f,%f,0,0,0,1' % p,
                          'grip': grip, 'trigger': 0.5 * (1 + math.sin(t)),
                          'axisX': 0.0},
                'left': {'pose': '%f,%f,%f,0,0,0,1' % p,
                         'grip': grip, 'trigger': 0.0, 'axisX': 0.0},
            },
        }
        out.append(XRFrame('SYNTH', raw, int(t * 1e9)))
    return out


def resample_to_control_rate(frames, hz):
    """Zero-order-hold resample a recorded frame stream to the real control
    rate, not the recording's frame rate. See docs/core.md for why
    iterating by recorded frame would be misleading. Shared with
    bench/compare_ik.py so both use the identical resampled timeline."""
    period = 1.0 / hz
    t0, t1 = frames[0].host_ns, frames[-1].host_ns
    steps = int((t1 - t0) / 1e9 / period)
    fi = 0
    sim = []
    for k in range(steps):
        t = t0 + k * period * 1e9
        while fi + 1 < len(frames) and frames[fi + 1].host_ns <= t:
            fi += 1
        sim.append(frames[fi])
    return sim


def replay(frames, hand, arm, cfg, verbose=False, force_clutch=False,
           axis_map=None, resampled=False, tool=None, solver='analytic',
           diff_ik_config=None, delta_frame='base'):
    arm_type = 0 if arm == 'A' else 1
    ik = ArmIK(arm_type=arm_type, config_path=cfg.KINE_CFG,
               pos_tol_mm=cfg.IK_POS_TOL_MM, rot_tol_deg=cfg.IK_ROT_TOL_DEG,
               max_step_deg=cfg.IK_MAX_STEP_DEG,   # branch-jump detection; rate limiting is safety's job
               limit_margin_deg=cfg.LIMIT_MARGIN_DEG,
               limit_override=cfg.JOINT_LIMIT_OVERRIDE.get(arm))

    ok, worst = verify_euler_convention(ik.kine)
    print('欧拉约定与 SDK 一致: %s（最大偏差 %.2e 度）' % (ok, worst))
    if not ok:
        print('❌ 欧拉约定不一致，姿态会算错，先修 retarget.mat_to_xyzabc')
        return None

    lo, hi = ik.lim_n, ik.lim_p      # single source of truth for joint limits
    gate = SafetyGate(lo, hi, cfg.MAX_JOINT_RATE_DEG_S, cfg.LIMIT_MARGIN_DEG,
                      cfg.MAX_TRACKING_ERR_DEG, cfg.MAX_CONSEC_REJECT,
                      cfg.XR_STALE_S, dt_max=cfg.MAX_STEP_DT_S,
                      max_tracking_err_s=cfg.MAX_TRACKING_ERR_S)
    # Same branch, same builder, same order as core/arm_channel.py -- an
    # offline rehearsal solved by a differently-configured solver than the
    # one that runs on hardware would not be a rehearsal of anything.
    diff_solver = None
    ns = None
    if solver == 'diff':
        if diff_ik_config is None:
            raise ValueError("solver='diff' requires diff_ik_config "
                             '(an algos.solver_config.DiffIKConfig)')
        if cfg.NULLSPACE_ENABLED:
            print('⚠️  solver=diff 不支持 NullSpaceController，已关闭'
                  '（零空间规避请配置 diff.yaml 的 mu_nullspace）')
        diff_solver = build_diff_solver(ik, arm_type, cfg, diff_ik_config)
    elif cfg.NULLSPACE_ENABLED:
        ns = NullSpaceController(
            [JointLimitAvoidance(lo, hi,
                                 activation_deg=cfg.NULLSPACE_ACTIVATION_DEG)],
            rate_deg_s=cfg.NULLSPACE_RATE_DEG_S,
            limit_deg=cfg.ARM_ANGLE_LIMIT,
            probe_deg=cfg.NULLSPACE_PROBE_DEG)

    def solve_joints(target, q_now, zsp_dir, arm_angle):
        with ik.probing():          # exclude trial solves from IK stats
            r = ik.solve_with_backoff(target, q_now, zsp_dir=zsp_dir,
                                      arm_angle=arm_angle)
        return r.joints if r.ok else None

    rt = Retargeter(cfg, arm, lowpass=LowPass(cfg.LOWPASS_HZ), nullspace=ns,
                    axis_map=axis_map, tool=tool, delta_frame=delta_frame)
    if force_clutch:
        # Diagnostic only: bypasses the clutch state machine, doesn't validate it.
        rt._clutch_pressed = lambda frame, hand: True
        print('⚠️  --force-clutch：强制视为一直握住离合，clutch 状态机未被验证')

    # Starting config: vendor demo's "desk task" pose. Use measured joints on real hardware.
    q = [44.04, -62.57, -8.92, -57.21, 1.45, -4.39, 2.1]
    q_cmd = list(q)

    n_engaged = n_ik_ok = n_gate_ok = n_clamped = 0
    ik_reasons, gate_reasons = {}, {}
    max_step = 0.0
    max_pos_err = 0.0
    max_cart_lag = 0.0
    max_solve_ms = 0.0
    prev_ns = None

    if resampled:
        sim = frames
        print('回放 %d 个控制周期 @ %.0fHz（源已按控制率插值）'
              % (len(sim), cfg.CONTROL_HZ))
    else:
        sim = resample_to_control_rate(frames, cfg.CONTROL_HZ)
        print('回放 %d 帧录制 → %d 个控制周期 @ %.0fHz（零阶保持）'
              % (len(frames), len(sim), cfg.CONTROL_HZ))

    period = 1.0 / cfg.CONTROL_HZ
    for i, f in enumerate(sim):
        dt = period
        prev_ns = f.host_ns

        if not rt.update(f, hand, q_cmd, ik.fk_xyzabc, ik.nsp_dir, dt=dt,
                         solve=solve_joints):
            continue
        n_engaged += 1
        max_cart_lag = max(max_cart_lag, rt.cart_lag_mm)

        if diff_solver is not None:
            # diff-IK is a rate controller: it takes one bounded step
            # toward the target per frame, so pos_err_mm below is tracking
            # lag, not a convergence residual (docs/algos.md#diff_ikpy).
            r = diff_solver.solve(rt.target_xyzabc, q_cmd, dt)
            max_solve_ms = max(max_solve_ms, r.solve_time_ms or 0.0)
        else:
            r = ik.solve_with_backoff(rt.target_xyzabc, q_cmd,
                                      zsp_dir=rt.zsp_dir,
                                      arm_angle=rt.arm_angle)
        if not r.ok:
            ik_reasons[r.reason] = ik_reasons.get(r.reason, 0) + 1
            if verbose and i % 100 == 0:
                print('  帧%4d IK 拒解 %s %s' % (i, r.reason, r.detail))
            continue
        n_ik_ok += 1
        max_pos_err = max(max_pos_err, r.pos_err_mm)

        v = gate.check(r.joints, q_cmd, q, xr_age_s=0.0, dt=dt)
        if not v.ok:
            gate_reasons[v.reason] = gate_reasons.get(v.reason, 0) + 1
            continue
        n_gate_ok += 1
        if v.clamped:
            n_clamped += 1
        step = max(abs(a - b) for a, b in zip(v.joints, q_cmd))
        max_step = max(max_step, step)
        q_cmd = v.joints
        q = list(q_cmd)     # replay assumes perfect servo tracking

    total = len(sim)
    print('\n=== 回放报告 ===')
    print('总帧数           : %d' % total)
    print('clutch 按下帧数  : %d (%.1f%%)  离合次数 %d'
          % (n_engaged, 100.0 * n_engaged / max(total, 1), rt.engage_count))
    if n_engaged:
        print('IK 成功          : %d/%d = %.2f%%'
              % (n_ik_ok, n_engaged, 100.0 * n_ik_ok / n_engaged))
        if ik_reasons:
            print('  IK 拒解分布    : %s' % ik_reasons)
        print('安全闸放行       : %d/%d = %.2f%%'
              % (n_gate_ok, n_ik_ok, 100.0 * n_gate_ok / max(n_ik_ok, 1)))
        if gate_reasons:
            print('  拦截分布       : %s' % gate_reasons)
        print('被钳位的帧       : %d (%.1f%%)'
              % (n_clamped, 100.0 * n_clamped / max(n_gate_ok, 1)))
        print('最大单帧增量     : %.4f° (上限 %.2f°)'
              % (max_step, cfg.MAX_JOINT_STEP_DEG))
        print('笛卡尔限速介入   : %d 帧 (%.1f%%)  最大滞后 %.1f mm'
              % (rt.n_slewed, 100.0 * rt.n_slewed / max(n_engaged, 1),
                 max_cart_lag))
        if diff_solver is not None:
            print('diff-IK 跟踪滞后  : %.3f mm (峰值；不是收敛残差)'
                  % max_pos_err)
            print('diff-IK 求解耗时  : 峰值 %.3f ms (周期预算 %.1f ms)'
                  % (max_solve_ms, 1000.0 / cfg.CONTROL_HZ))
        else:
            print('IK 最大 FK 误差  : %.6f mm (容差 %.2f)'
                  % (max_pos_err, cfg.IK_POS_TOL_MM))
        if ns:
            print('零空间           : 介入 %d 帧，探测被抹平 %d 帧，'
                  '最终臂角 %+.1f° / 行程 ±%.0f°'
                  % (ns.n_moved, ns.n_blind, ns.angle, cfg.ARM_ANGLE_LIMIT))

    verdict = (n_engaged > 0
               and n_ik_ok / max(n_engaged, 1) > 0.99
               and max_step <= cfg.MAX_JOINT_STEP_DEG + 1e-9)
    print('\nP2 验收: %s' % ('✅ 通过' if verdict else '❌ 不通过'))
    if not verdict and n_engaged:
        print('  → IK 成功率不足多半是坐标标定错了或起始构型离工作空间边缘太近')
    return verdict


def main():
    ap = argparse.ArgumentParser(
        description='Offline replay of an XR trajectory through the full pipeline')
    ap.add_argument('traj', nargs='?', help='jsonl produced by xr_source.py --record')
    ap.add_argument('--synth', action='store_true',
                    help='Self-test with a synthetic trajectory.')
    ap.add_argument('--umi', metavar='DIR',
                    help='Replay a UMI/LeRobot v3 episode directory instead '
                         '(drivers/umi_source.py). Retargets with the '
                         "body-frame delta (delta_frame='ee'), the same "
                         'representation the policy trains on -- no axis '
                         'map is involved.')
    ap.add_argument('--episode', type=int, default=0,
                    help='--umi only: which episode_index to replay.')
    ap.add_argument('--frames', default=None, metavar='A:B',
                    help='--umi only: replay just this half-open range of '
                         'RECORDED frames, e.g. the spans scripts/'
                         'umi_filter.py prints. Same indices as the dataset, '
                         'not control periods.')
    ap.add_argument('--tool', default='umi',
                    help="--umi only: configs/tool/<name>.yaml to take the "
                         "flange->fingertip offset from ('none' = judge the "
                         'flange, i.e. pre-tool_frame.py behaviour).')
    ap.add_argument('--tip', default=None,
                    help='--umi only: override the tool offset, '
                         'x,y,z[,a,b,c] mm/deg.')
    ap.add_argument('--solver', choices=['analytic', 'diff'], default=None,
                    help='IK solver. Default follows the entry point it is '
                         'rehearsing for: diff with --umi (matching '
                         'scripts/umi_replay.py), analytic otherwise '
                         '(matching run_teleop.py).')
    ap.add_argument('--solver-config', default=None,
                    help='--solver diff only: YAML path '
                         '(default configs/solver/diff.yaml).')
    ap.add_argument('--hand', default='right', choices=['left', 'right'])
    ap.add_argument('--arm', default=None, choices=['A', 'B'])
    ap.add_argument('--scale', type=float, default=None)
    ap.add_argument('--vel-ratio', type=int, default=None,
                    help='Try a different velocity percent (also rescales '
                         'upper-layer limits). Compare here before running '
                         'on hardware.')
    ap.add_argument('--rotation', action='store_true',
                    help='Force orientation tracking.')
    ap.add_argument('--force-clutch', action='store_true',
                    help='Ignore the clutch button, forcing continuous '
                         'engagement (diagnostic only).')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    if args.scale is not None:
        config.SCALE = args.scale
    if args.vel_ratio is not None:
        config.apply_vel_ratio(args.vel_ratio)
    if args.rotation:
        config.FOLLOW_ROTATION = True
    arm = args.arm or config.ARM_OF_HAND[args.hand]

    # The point of this script is to rehearse what will actually run, so
    # the solver defaults to whatever the matching hardware entry point
    # defaults to -- rehearsing the analytic solver and then deploying
    # diff-IK would validate a trajectory nobody is going to execute.
    solver = args.solver or ('diff' if args.umi else 'analytic')
    solver_cfg = None
    if solver == 'diff':
        try:
            solver_cfg, solver_cfg_path = load_diff_ik_config(
                args.solver_config,
                os.path.dirname(os.path.abspath(__file__)))
        except ValueError as e:
            print('❌ %s' % e)
            return 2
        print('求解器: diff-IK  配置 %s\n%s'
              % (solver_cfg_path, solver_cfg.model_dump()))
    else:
        print('求解器: analytic (ik_solver.ArmIK)')

    axis_map = None
    resampled = False
    tool = None
    delta_frame = 'base'
    if args.umi:
        from algos.tool_frame import ToolFrame
        from drivers.umi_source import (load_episode, resample,
                                        slice_frames)
        raw, info = load_episode(args.umi, args.episode)
        if raw[0].pose(args.hand) is None:
            print('❌ episode %d 里没有 %s 手的数据' % (args.episode, args.hand))
            return 1
        print('载入 %d 帧 @ %sHz ← %s (episode %d, %s)'
              % (len(raw), info.get('fps'), args.umi, args.episode,
                 info.get('robot_type')))
        if args.frames:
            raw = slice_frames(raw, args.frames)
            if len(raw) < 2:
                print('❌ --frames %s 只剩 %d 帧，没法回放'
                      % (args.frames, len(raw)))
                return 1
            print('  截取 --frames %s -> %d 帧' % (args.frames, len(raw)))
        # A recorded UMI pose is the leader gripper's fingertip midpoint, not
        # a flange pose -- follow it at the flange and the real fingertips
        # swing through an arc nobody demonstrated. See
        # docs/algos.md#tool_framepy.
        if args.tip:
            tool = ToolFrame([float(v) for v in args.tip.split(',')],
                             label='--tip')
        elif args.tool.lower() not in ('none', 'no', 'flange'):
            tool, tool_path = ToolFrame.load(
                args.tool, arm, os.path.dirname(os.path.abspath(__file__)))
            if tool is None:
                print('⚠️  %s 的 kine_offset 全零 —— 指尖偏移未标定，'
                      '下面判的是法兰轨迹而不是指尖轨迹'
                      '（scripts/tip_calib.py 可标定）' % tool_path)
        if tool is not None:
            print('  指尖偏移 %s' % tool)
        # The UMI world origin is the headset pose at VR-app start and moves
        # every restart, so these are only ever followed relatively -- the
        # clutch reads as permanently pressed, latching onto the robot's
        # actual pose at frame 0. See drivers/umi_source.py.
        frames = resample(raw, config.CONTROL_HZ)
        resampled = True
        # Body-frame delta, like scripts/umi_replay.py and like the policy
        # pipeline -- so there is no world->base axis map on this path.
        delta_frame = 'ee'
    elif args.synth or not args.traj:
        print('⚠️  用合成轨迹自检（不是真实数据，只验管线自洽）\n')
        frames = synth_traj()
    else:
        frames = load_traj(args.traj)
        print('载入 %d 帧 ← %s' % (len(frames), args.traj))
    if not frames:
        print('没有可用帧'); return 1

    if not config.CALIBRATED:
        print('⚠️  config.CALIBRATED=False —— 坐标标定还没做，'
              '下面的数字只说明管线跑得通，不说明方向是对的\n')
    config.summary()
    print()
    ok = replay(frames, args.hand, arm, config, verbose=args.verbose,
                force_clutch=args.force_clutch, axis_map=axis_map,
                resampled=resampled, tool=tool, solver=solver,
                diff_ik_config=solver_cfg, delta_frame=delta_frame)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
