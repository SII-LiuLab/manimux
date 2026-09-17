#!/usr/bin/env python3
"""Servo lag probe: which follow model does position mode (ARM_STATE=1) implement?

docs/config.md#speed-budget assumes the controller's follow lag is
v**2/(2a) -- acceleration-limited braking toward each new target -- which
caps VEL_RATIO near 53% under MAX_TRACKING_ERR_DEG=5. That was derived, never
measured. If the follower matches target velocity instead (velocity
feedforward), lag is ~v*tau and the cap is 5/tau.

One joint sweeps +-amp at constant commanded velocities, once per
ACC_RATIO. Every 1 kHz controller frame is logged: inputs.joint_cmd_pos
(what the controller received), outputs.fb_joint_cmd (its own position
command -- an echo, or post-limiter), fb_joint_pos/posE/vel/cToq. Three
things separate the models:

- steady cruise lag vs v: linear (tau*v) or quadratic (v**2/2a)
- lag vs ACC_RATIO at the same v: v**2/2a scales with 1/ACC, tau*v does not
- corner shape: a pure time shift (feedforward + delay) or rounded (P loop
  without feedforward, time constant 1/Kp). Both are linear in v; only the
  transient tells them apart.

--selftest runs the analysis on synthetic data from all three models and
checks it names each one. --plan and --selftest never touch the robot.

Usage:
    python3 bench/servo_lag_probe.py --selftest
    python3 bench/servo_lag_probe.py --plan --center 28.56
    python3 bench/servo_lag_probe.py --execute --arm A --joint 7
    python3 bench/servo_lag_probe.py --analyze bench/results/servo_lag_A_J7_*.npz --plot
"""
import argparse
import configparser
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

_SDK = os.environ.get('MARVIN_SDK', '/home/jw/Downloads/TJ_FX_ROBOT_CONTRL_SDK')

_HEAD = ('host_t', 'frame_serial', 'in_frame_serial', 'cur_state', 'err_code')
_FIELDS = (('inputs', 'joint_cmd_pos'), ('outputs', 'fb_joint_cmd'),
           ('outputs', 'fb_joint_pos'), ('outputs', 'fb_joint_posE'),
           ('outputs', 'fb_joint_vel'), ('outputs', 'fb_joint_cToq'))

LIMIT_MARGIN_DEG = 5.0
WINDOW_SLACK_DEG = 3.0      # measured joint outside +-amp by this much -> abort
# Residual peak / steady lag below SHAPE_SHIFTED = pure time shift. From
# --selftest's models with 8% dropped frames: delay 0.054-0.075, P loop
# (T = 16-33 ms) and brake 0.28-0.31; 0.17 sits between. The 8 ms box keeps
# drop-interpolation glitches on the command staircase from reading as rounding.
SHAPE_SHIFTED = 0.17
SHAPE_BOX = 8               # ms box applied to cmd and pos before the shift fit
ECHO_DEG = 0.02             # fb_joint_cmd within this of joint_cmd_pos = echo


# ---------- axis parameters ----------

def read_axis(arm, joint):
    """AccMax/VelMax/limits for one axis from the SDK's robot.ini."""
    cp = configparser.ConfigParser()
    cp.optionxform = str
    path = os.path.join(_SDK, 'robot.ini')
    if not cp.read(path):
        raise SystemExit('读不到 %s（设 MARVIN_SDK 指向 SDK 目录）' % path)
    sec = cp['R.A%d.L%d.BASIC' % ({'A': 0, 'B': 1}[arm], joint - 1)]
    return dict(acc_max=float(sec['AccMax']), vel_max=float(sec['VelMax']),
                lim_lo=float(sec['LimitNeg']), lim_hi=float(sec['LimitPos']))


# ---------- command profile ----------

def build_segments(center, amp, speeds, a_cmd, dwell_s):
    """Dwell at center-amp, then per speed: ramp up, dwell, ramp back, dwell.

    Ramps are trapezoids with a short a_cmd corner, far steeper than the
    controller's own AccMax*ACC_RATIO, so the controller's response to a
    velocity step stays visible without commanding infinite acceleration.
    """
    lo, hi = center - amp, center + amp
    segs = []
    t = 0.0

    def dwell(p):
        nonlocal t
        segs.append(dict(kind='dwell', t0=t, t1=t + dwell_s, p0=p, p1=p, v=0.0))
        t += dwell_s

    def ramp(p0, p1, v):
        nonlocal t
        d = abs(p1 - p0)
        t_acc = v / a_cmd
        d_acc = 0.5 * v * t_acc
        if 2 * d_acc >= d:
            raise ValueError('%.0f°/s 在 %.1f° 行程内达不到匀速' % (v, d))
        dur = 2 * t_acc + (d - 2 * d_acc) / v
        segs.append(dict(kind='ramp', t0=t, t1=t + dur, p0=p0, p1=p1,
                         v=math.copysign(v, p1 - p0), a=a_cmd,
                         cruise0=t + t_acc, cruise1=t + dur - t_acc))
        t += dur

    dwell(lo)
    for v in speeds:
        ramp(lo, hi, v)
        dwell(hi)
        ramp(hi, lo, v)
        dwell(lo)
    return segs


def evaluate(segs, t):
    """(commanded angle, segment index) at profile time t, clamped to the ends."""
    if t <= 0:
        return segs[0]['p0'], 0
    for i, s in enumerate(segs):
        if t < s['t1']:
            break
    else:
        return segs[-1]['p1'], len(segs) - 1
    if s['kind'] == 'dwell':
        return s['p0'], i
    tau = t - s['t0']
    dur = s['t1'] - s['t0']
    v, a = abs(s['v']), s['a']
    t_acc = v / a
    if tau < t_acc:
        d = 0.5 * a * tau * tau
    elif tau > dur - t_acc:
        r = dur - tau
        d = abs(s['p1'] - s['p0']) - 0.5 * a * r * r
    else:
        d = 0.5 * v * t_acc + v * (tau - t_acc)
    return s['p0'] + math.copysign(d, s['v']), i


def check_profile(segs, hz=1000.0):
    t = np.arange(0.0, segs[-1]['t1'], 1.0 / hz)
    q = np.array([evaluate(segs, x)[0] for x in t])
    v = np.diff(q) * hz
    a = np.diff(v) * hz
    return dict(q_min=q.min(), q_max=q.max(), v_peak=np.abs(v).max(),
                a_peak=np.abs(a).max())


def print_plan(args, axis, center, speeds, acc_ratios):
    segs = build_segments(center, args.amp, speeds, args.cmd_acc, args.dwell)
    chk = check_profile(segs)
    cap = axis['vel_max'] * args.vel_ratio / 100.0
    lo, hi = center - args.amp, center + args.amp
    print('臂%s J%d  中心 %.2f°  摆幅 ±%.1f°（%.2f° .. %.2f°）  限位 %.0f .. %.0f'
          % (args.arm, args.joint, center, args.amp, lo, hi,
             axis['lim_lo'], axis['lim_hi']))
    print('VEL_RATIO %d → 控制器速度上限 %.1f°/s；AccMax %.0f°/s²；指令拐角加速度 %.0f°/s²'
          % (args.vel_ratio, cap, axis['acc_max'], args.cmd_acc))
    print('每个 ACC 档一轮，共 %d 段，%.1f 秒（不含往返起点的慢速移动）'
          % (len(segs), segs[-1]['t1']))
    print('指令流自检：位置 %.2f .. %.2f°  峰值速度 %.2f°/s  峰值加速度 %.0f°/s²'
          % (chk['q_min'], chk['q_max'], chk['v_peak'], chk['a_peak']))

    print('\n预测稳态滞后（°）：')
    print('  v°/s ' + ''.join('  v²/2a@ACC%-3d' % acc for acc in acc_ratios)
          + '  v·τ@10ms  v·τ@30ms')
    worst = 0.0
    for v in speeds:
        preds = [v * v / (2 * axis['acc_max'] * acc / 100.0) for acc in acc_ratios]
        worst = max(worst, *preds)
        print('  %4.0f ' % v + ''.join('  %12.2f' % p for p in preds)
              + '  %8.2f  %8.2f' % (v * 0.010, v * 0.030))

    problems = []
    if chk['v_peak'] > max(speeds) * 1.01:
        problems.append('指令峰值速度 %.2f 超出设定' % chk['v_peak'])
    if chk['a_peak'] > args.cmd_acc * 1.05:
        problems.append('指令峰值加速度 %.0f 超出设定' % chk['a_peak'])
    if max(speeds) > 0.9 * cap:
        problems.append('最高速度 %.0f°/s 超过控制器上限 %.1f 的 90%%，会被限速饱和'
                        % (max(speeds), cap))
    if lo < axis['lim_lo'] + LIMIT_MARGIN_DEG or hi > axis['lim_hi'] - LIMIT_MARGIN_DEG:
        problems.append('摆动范围离限位不足 %.0f°' % LIMIT_MARGIN_DEG)
    if worst + 2.0 > args.abort_err:
        problems.append('v²/2a 预测最大滞后 %.2f° 离中止阈值 %.1f° 不足 2°'
                        % (worst, args.abort_err))
    print('\n中止条件：J%d 指令-实测 > %.1f°，其余轴 > %.1f°，实测越出 ±(摆幅+%.0f°)，'
          '状态≠1 或有错误码，Ctrl+C'
          % (args.joint, args.abort_err, args.hold_tol, WINDOW_SLACK_DEG))
    for p in problems:
        print('❌ ' + p)
    if not problems:
        print('✅ 计划检查通过')
    return segs, problems


# ---------- hardware run ----------

def make_guard(j, lo, hi, abort_err, hold_tol, state_position):
    def guard(q_cmd, q_meas, st):
        if st['err_code'] or st['cur_state'] != state_position:
            return '状态 %s 错误码 %s' % (st['cur_state'], st['err_code'])
        if abs(q_cmd[j] - q_meas[j]) > abort_err:
            return 'J%d 指令-实测 %.2f°' % (j + 1, q_cmd[j] - q_meas[j])
        for k in range(7):
            if k != j and abs(q_cmd[k] - q_meas[k]) > hold_tol:
                return '保持轴 J%d 偏差 %.2f°' % (k + 1, q_cmd[k] - q_meas[k])
        if not lo - WINDOW_SLACK_DEG <= q_meas[j] <= hi + WINDOW_SLACK_DEG:
            return 'J%d 实测 %.2f° 越出摆动窗口' % (j + 1, q_meas[j])
        return None
    return guard


def run_group(conn, drv, j, q_hold, segs, hz, guard):
    """Stream the profile at hz while logging every new 1 kHz frame."""
    from drivers.arm_driver import send_joint_commands
    idx = drv.idx
    period = 1.0 / hz
    total = segs[-1]['t1']
    frames, sent = [], []
    q_cmd = list(q_hold)
    last_fs = None
    reason = None
    t_start = time.perf_counter()
    next_send = t_start
    try:
        while True:
            now = time.perf_counter()
            t = now - t_start
            if t >= total:
                break
            if now >= next_send:
                q_cmd[j], seg = evaluate(segs, t)
                send_joint_commands(conn, {drv.arm: q_cmd})
                sent.append((now, t, q_cmd[j], seg))
                next_send += period
                if next_send <= now:        # overran: rebase, don't burst
                    next_send = now + period
            s = conn.subscribe()
            o, st = s['outputs'][idx], s['states'][idx]
            fs = o['frame_serial']
            if fs != last_fs:
                last_fs = fs
                row = [time.perf_counter(), fs, s['inputs'][idx]['in_frame_serial'],
                       st['cur_state'], st['err_code']]
                for grp, key in _FIELDS:
                    row.extend(s[grp][idx][key])
                frames.append(row)
                reason = guard(q_cmd, o['fb_joint_pos'], st)
                if reason:
                    break
            time.sleep(0.0002)
    except KeyboardInterrupt:
        reason = 'Ctrl+C'
    return t_start, frames, sent, reason


def save_group(meta, segs, frames, sent):
    out_dir = os.path.join(_REPO_ROOT, 'bench', 'results')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'servo_lag_%s_J%d_vel%d_acc%d_%s.npz' % (
        meta['arm'], meta['joint'], meta['vel_ratio'], meta['acc_ratio'],
        time.strftime('%Y%m%d_%H%M%S')))
    np.savez_compressed(path, frames=np.asarray(frames, dtype=np.float64),
                        sent=np.asarray(sent, dtype=np.float64),
                        segments=json.dumps(segs), meta=json.dumps(meta))
    return path


def execute(args, axis, speeds, acc_ratios):
    import config
    from drivers.arm_driver import (ArmDriver, RobotConnection, STATE_ERROR,
                                    STATE_POSITION, move_to_joints)
    j = args.joint - 1
    conn = RobotConnection(config.ROBOT_IP)
    drv = ArmDriver(conn, args.arm, SimpleNamespace(
        VEL_RATIO=args.vel_ratio, ACC_RATIO=acc_ratios[0], ARM_STATE=STATE_POSITION))
    try:
        st = drv.state()
        if st['err'] or st['cur'] == STATE_ERROR:
            raise SystemExit('臂%s 状态 %s 错误码 %s —— 本脚本不自动清错，先用 GUI 或 '
                             'tools/wait_ready.py 排查' % (args.arm, st['cur'], st['err']))
        q0 = list(st['q'])
        gap = max(abs(a - b) for a, b in zip(st['q_cmd'], q0))
        if gap > 0.5:
            raise SystemExit('joint_cmd_pos 与实测相差 %.2f°，切位置模式可能跳变，拒绝运行' % gap)
        center = q0[j]
        print('当前 %s' % [round(v, 2) for v in q0])
        segs, problems = print_plan(args, axis, center, speeds, acc_ratios)
        if problems:
            raise SystemExit('计划检查未通过，不运行')
        if input('\n确认周围空旷、急停在手。输入 yes 开始真机运动：').strip() != 'yes':
            print('未确认，退出（未切模式、未发指令）')
            return
        q_start = list(q0)
        q_start[j] = center - args.amp
        lo, hi = center - args.amp, center + args.amp
        guard = make_guard(j, lo, hi, args.abort_err, args.hold_tol, STATE_POSITION)
        for acc in acc_ratios:
            drv.cfg = SimpleNamespace(VEL_RATIO=args.vel_ratio, ACC_RATIO=acc,
                                      ARM_STATE=STATE_POSITION)
            drv.prepare()
            got = drv.state()['acc_ratio']
            if got != acc:
                raise RuntimeError('加速度档回读 %r，下发 %d' % (got, acc))
            print('\n== ACC %d%%：移动到起点 J%d=%.2f°' % (acc, args.joint, q_start[j]))
            ok, why, _ = move_to_joints(conn, {args.arm: drv}, {args.arm: q_start},
                                        speed_deg_s=args.approach_speed, hz=args.hz,
                                        max_err_deg=config.MAX_TRACKING_ERR_DEG, quiet=True)
            if not ok:
                raise RuntimeError('移动到起点失败：%s' % why)
            print('开始扫速（%.1f 秒）…' % segs[-1]['t1'])
            t_start, frames, sent, reason = run_group(conn, drv, j, q_start, segs,
                                                      args.hz, guard)
            meta = dict(arm=args.arm, joint=args.joint, vel_ratio=args.vel_ratio,
                        acc_ratio=acc, acc_max=axis['acc_max'], vel_max=axis['vel_max'],
                        center=center, amp=args.amp, speeds=speeds, a_cmd=args.cmd_acc,
                        hz=args.hz, dwell=args.dwell, t_start_host=t_start,
                        controller_version=conn.version, abort_reason=reason)
            path = save_group(meta, segs, frames, sent)
            print('已存 %s（%d 帧，%d 条指令）' % (path, len(frames), len(sent)))
            if reason:
                drv.stop()
                raise RuntimeError('中止：%s' % reason)
            ok, why, _ = move_to_joints(conn, {args.arm: drv}, {args.arm: q0},
                                        speed_deg_s=args.approach_speed, hz=args.hz,
                                        max_err_deg=config.MAX_TRACKING_ERR_DEG, quiet=True)
            if not ok:
                raise RuntimeError('回中心失败：%s' % why)
            drv.disable()
        print('\n完成。分析：python3 bench/servo_lag_probe.py --analyze '
              'bench/results/servo_lag_%s_J%d_*.npz --plot' % (args.arm, args.joint))
    finally:
        drv.disable()
        conn.close()
        print('已下伺服并断开连接')


# ---------- analysis ----------

def _col(frames, key, j):
    names = [k for _, k in _FIELDS]
    return frames[:, len(_HEAD) + 7 * names.index(key) + j]


def _uniform(frames, j):
    """Resample onto the controller's own 1 kHz clock (frame_serial)."""
    f = frames[np.argsort(frames[:, 1], kind='stable')]
    f = f[np.concatenate(([True], np.diff(f[:, 1]) > 0))]
    fs = f[:, 1]
    grid = np.arange(fs[0], fs[-1] + 1)
    t = (grid - fs[0]) / 1000.0
    sig = {k: np.interp(grid, fs, _col(f, k, j))
           for k in ('joint_cmd_pos', 'fb_joint_cmd', 'fb_joint_pos', 'fb_joint_posE')}
    return t, np.interp(grid, fs, f[:, 0]), sig


def _best_shift(t, ref, y, t0, t1, max_shift_ms=250):
    """Delay k (ms) minimizing y(t) - ref(t - k) over [t0, t1]: (k, rms, peak)."""
    idx = np.nonzero((t >= t0) & (t <= t1))[0]

    def resid(k):
        src = idx - k
        ok = src >= 0
        return y[idx[ok]] - ref[src[ok]]

    k_best = min(range(max_shift_ms + 1), key=lambda k: float(np.mean(resid(k) ** 2)))
    r = resid(k_best)
    return k_best, float(np.sqrt(np.mean(r * r))), float(np.abs(r).max())


def analyze_group(meta, segs, frames):
    j = meta['joint'] - 1
    t, host, sig = _uniform(frames, j)
    cmd, gen = sig['joint_cmd_pos'], sig['fb_joint_cmd']
    pos, pos_e = sig['fb_joint_pos'], sig['fb_joint_posE']
    # Same box on both sides of the shift fit: smoothing commutes with a time
    # shift, so the 250 Hz command staircase cancels instead of reading as
    # corner rounding when the lag itself is only a few tenths of a degree.
    box = np.ones(SHAPE_BOX) / SHAPE_BOX
    cmd_s = np.convolve(cmd, box, mode='same')
    pos_s = np.convolve(pos, box, mode='same')

    def to_ctrl(tp):
        return float(np.interp(meta['t_start_host'] + tp, host, t))

    per = []
    for s in segs:
        if s['kind'] != 'ramp':
            continue
        sgn = math.copysign(1.0, s['v'])
        c0, c1 = to_ctrl(s['cruise0']), to_ctrl(s['cruise1'])
        w = (t >= c0 + 0.5 * (c1 - c0)) & (t <= c1 - 0.01)
        if w.sum() < 20:
            continue
        lag = sgn * (cmd - pos)
        lag_ss = float(np.median(lag[w]))
        slope = np.polyfit(t[w], lag[w], 1)[0]
        shift_ms, _, peak = _best_shift(t, cmd_s, pos_s, to_ctrl(s['t0']) - 0.05,
                                        to_ctrl(s['t1']) + 0.5)
        per.append(dict(v=abs(s['v']), lag=lag_ss,
                        lag_e=float(np.median(sgn * (cmd - pos_e)[w])),
                        gen=float(np.median(sgn * (cmd - gen)[w])),
                        servo=float(np.median(sgn * (gen - pos)[w])),
                        drift=float(slope * (t[w][-1] - t[w][0])),
                        shift_ms=float(shift_ms),
                        shape=peak / max(abs(lag_ss), 1e-3)))
    rows = []
    for v in sorted({r['v'] for r in per}):
        grp = [r for r in per if r['v'] == v]
        rows.append({k: float(np.mean([r[k] for r in grp])) for k in grp[0]})

    v = np.array([r['v'] for r in rows])
    e = np.array([r['lag'] for r in rows])
    tau = float(v @ e / (v @ v))
    k2 = float((v ** 2) @ e / ((v ** 2) @ (v ** 2)))
    both = np.linalg.lstsq(np.stack([v, v ** 2], axis=1), e, rcond=None)[0]
    fit = dict(tau=tau, lin_rms=float(np.sqrt(np.mean((e - tau * v) ** 2))),
               a_eff=(1.0 / (2 * k2) if k2 > 0 else float('inf')),
               quad_rms=float(np.sqrt(np.mean((e - k2 * v ** 2) ** 2))),
               both_tau=float(both[0]),
               both_a=(1.0 / (2 * both[1]) if both[1] > 0 else float('inf')))
    moving = np.abs(np.gradient(cmd_s, t)) > 1.0
    echo = float(np.abs(cmd - gen)[moving].max()) if moving.any() else 0.0
    return dict(meta=meta, rows=rows, fit=fit, echo=echo,
                trace=dict(t=t, cmd=cmd, gen=gen, pos=pos), segs=segs, to_ctrl=to_ctrl)


def verdict(groups):
    """(label, lines): label is 'brake', 'delay' or 'p_loop'."""
    lines = []
    fast = [r for g in groups for r in g['rows'] if r['v'] >= 20]
    shape = float(np.median([r['shape'] for r in fast]))
    accs = sorted({g['meta']['acc_ratio'] for g in groups})
    if len(accs) >= 2:
        lo = next(g for g in groups if g['meta']['acc_ratio'] == accs[0])
        hi = next(g for g in groups if g['meta']['acc_ratio'] == accs[-1])
        expected = accs[-1] / accs[0]
        ratios = []
        for rl in lo['rows']:
            rh = next((r for r in hi['rows'] if abs(r['v'] - rl['v']) < 1e-6), None)
            if rh and rl['v'] >= 20 and rh['lag'] > 0.05:
                ratios.append(rl['lag'] / rh['lag'])
        ratio = float(np.median(ratios)) if ratios else float('nan')
        acc_sensitive = ratio > 1 + 0.5 * (expected - 1)
        lines.append('ACC %d%% → %d%%：同速稳态滞后放大 %.2f×（v²/2a 预期 %.2f×，'
                     '与加速度无关的模型预期 1.00×）' % (accs[-1], accs[0], ratio, expected))
    else:
        f = groups[0]['fit']
        acc_sensitive = f['quad_rms'] < 0.5 * f['lin_rms']
        lines.append('只有 ACC %d%% 一档：二次拟合残差 %.3f°，线性拟合残差 %.3f°'
                     % (accs[0], f['quad_rms'], f['lin_rms']))
    lines.append('拐角形状：时移后残差峰值 / 稳态滞后 = %.2f（< %.2f 视为纯时移）'
                 % (shape, SHAPE_SHIFTED))
    if acc_sensitive:
        label = 'brake'
        lines.append('→ 符合"按静止终点刹车"的限加速度跟随：滞后 ≈ v²/(2a)，瓶颈在 ACC')
    elif shape < SHAPE_SHIFTED:
        label = 'delay'
        lines.append('→ 符合"速度前馈 + 纯延迟"：滞后 ≈ v·τ，按 5° 门限上限约 5/τ')
    else:
        label = 'p_loop'
        lines.append('→ 符合"无前馈 P 环"：滞后 ≈ v/Kp（线性），拐角被抹圆')
    return label, lines


def print_group(g):
    m, f = g['meta'], g['fit']
    a = m['acc_max'] * m['acc_ratio'] / 100.0
    print('\n== 臂%s J%d  VEL %d%%  ACC %d%%（a = %.0f°/s²）%s'
          % (m['arm'], m['joint'], m['vel_ratio'], m['acc_ratio'], a,
             '  ⚠️ 中止：%s' % m['abort_reason'] if m.get('abort_reason') else ''))
    print('fb_joint_cmd 与 joint_cmd_pos 运动中最大差 %.3f° → %s'
          % (g['echo'], '回显，拆不出两层' if g['echo'] < ECHO_DEG
             else '控制器内部指令，与输入不同'))
    print('  v°/s  滞后°  外编°  生成器°  伺服°   τ=e/v ms  时移ms  圆角比  窗内漂移°  v²/2a°')
    for r in g['rows']:
        print('  %4.0f  %5.2f  %5.2f  %7.2f  %5.2f  %9.1f  %6.0f  %6.2f  %9.2f  %6.2f'
              % (r['v'], r['lag'], r['lag_e'], r['gen'], r['servo'],
                 1e3 * r['lag'] / r['v'], r['shift_ms'], r['shape'], r['drift'],
                 r['v'] ** 2 / (2 * a)))
    print('  线性 e=τv：τ=%.1f ms，残差 %.3f°  |  二次 e=v²/2a：a_eff=%.0f°/s²，残差 %.3f°'
          '  |  组合：τ=%.1f ms，a=%.0f'
          % (1e3 * f['tau'], f['lin_rms'], f['a_eff'], f['quad_rms'],
             1e3 * f['both_tau'], f['both_a']))
    print('  按 5° 门限推算速度上限：线性 %.0f°/s，二次 %.0f°/s'
          % (5.0 / f['tau'] if f['tau'] > 0 else float('inf'),
             math.sqrt(2 * f['a_eff'] * 5.0)))
    if any(abs(r['drift']) > 0.15 for r in g['rows']):
        print('  ⚠️ 有速度的匀速窗内滞后仍在漂移（>0.15°），该档稳态值偏差大，加大 --amp')


def plot_groups(groups, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('没装 matplotlib，跳过 --plot')
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for g in groups:
        v = np.array([r['v'] for r in g['rows']])
        e = np.array([r['lag'] for r in g['rows']])
        vv = np.linspace(0, v.max(), 50)
        line = ax1.plot(v, e, 'o', label='ACC %d%%' % g['meta']['acc_ratio'])[0]
        ax1.plot(vv, g['fit']['tau'] * vv, '--', color=line.get_color(), alpha=0.6)
        ax1.plot(vv, vv ** 2 / (2 * g['fit']['a_eff']), ':', color=line.get_color(), alpha=0.6)
    ax1.set_xlabel('commanded speed (deg/s)')
    ax1.set_ylabel('steady lag cmd - fb_joint_pos (deg)')
    ax1.set_title('lag vs speed (-- linear fit, : quadratic fit)')
    ax1.legend()
    g = groups[0]
    s = next(s for s in g['segs'] if s['kind'] == 'ramp' and abs(s['v']) == max(g['meta']['speeds']))
    tr = g['trace']
    w = (tr['t'] >= g['to_ctrl'](s['t0']) - 0.1) & (tr['t'] <= g['to_ctrl'](s['t1']) + 0.4)
    t0 = tr['t'][w][0]
    for key in ('cmd', 'gen', 'pos'):
        ax2.plot(tr['t'][w] - t0, tr[key][w], label={'cmd': 'joint_cmd_pos', 'gen': 'fb_joint_cmd',
                                                     'pos': 'fb_joint_pos'}[key])
    ax2.set_xlabel('s')
    ax2.set_ylabel('deg')
    ax2.set_title('fastest ramp, ACC %d%%' % g['meta']['acc_ratio'])
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print('图已存 %s' % path)


def load_group(path):
    z = np.load(path, allow_pickle=False)
    return json.loads(str(z['meta'])), json.loads(str(z['segments'])), z['frames']


# ---------- selftest ----------

def _simulate(model, acc_ratio, vel_ratio=32, acc_max=900.0, seed=0, drop=0.08):
    """Synthetic run of one ACC group under a known follow model.

    delay copies the 250 Hz staircase 10 ms late; delay_smooth low-passes it
    (3 ms) first, like a real rotor. p_loop_fast (T = 16 ms) is the hardest
    P loop to tell from a pure delay. drop is the fraction of frames the
    1 kHz poll misses, ~8% on hardware.
    """
    rng = np.random.default_rng(seed + acc_ratio)
    speeds = [10.0, 20.0, 40.0, 50.0]
    segs = build_segments(0.0, 15.0, speeds, 2000.0, 1.0)
    n = int((segs[-1]['t1'] + 0.5) * 1000)
    a = acc_max * acc_ratio / 100.0
    vmax = 180.0 * vel_ratio / 100.0
    cmd, gen, pos = np.empty(n), np.empty(n), np.empty(n)
    g = p = lp = segs[0]['p0']
    vel = 0.0
    for k in range(n):
        t_send = math.floor((k - 1) / 4.0) * 0.004        # 250 Hz sends, 1 ms transport
        c = evaluate(segs, max(t_send, 0.0))[0]
        cmd[k] = c
        if model in ('delay', 'delay_smooth'):
            g = c
            if model == 'delay':
                p = cmd[max(k - 10, 0)]
            else:
                lp += (cmd[max(k - 10, 0)] - lp) * (1e-3 / 0.003)
                p = lp
        elif model in ('p_loop', 'p_loop_fast'):
            g = c
            kp = 30.0 if model == 'p_loop' else 60.0
            p += float(np.clip(kp * (c - p), -vmax, vmax)) * 1e-3
        else:                                           # brake toward a stationary target
            e = c - g
            v_des = math.copysign(min(vmax, math.sqrt(2 * a * abs(e))), e)
            vel += float(np.clip(v_des - vel, -a * 1e-3, a * 1e-3))
            g += vel * 1e-3
        gen[k] = g
        if model == 'brake':
            p = gen[max(k - 3, 0)]
        pos[k] = p
    keep = rng.random(n) > drop
    frames = np.zeros((n, len(_HEAD) + 7 * len(_FIELDS)))
    frames[:, 0] = 100.0 + np.arange(n) * 1e-3 + 8e-4 + rng.uniform(0, 4e-4, n)
    frames[:, 1] = 5000 + np.arange(n)
    frames[:, 3] = 1
    for key, x in (('joint_cmd_pos', cmd), ('fb_joint_cmd', gen),
                   ('fb_joint_pos', pos), ('fb_joint_posE', pos)):
        frames[:, len(_HEAD) + 7 * [k for _, k in _FIELDS].index(key) + 6] = x
    meta = dict(arm='A', joint=7, vel_ratio=vel_ratio, acc_ratio=acc_ratio,
                acc_max=acc_max, vel_max=180.0, center=0.0, amp=15.0, speeds=speeds,
                a_cmd=2000.0, hz=250.0, dwell=1.0, t_start_host=100.0,
                controller_version=None, abort_reason=None)
    return meta, segs, frames[keep]


def selftest():
    failures = 0
    expect = {'delay': 'delay', 'delay_smooth': 'delay', 'p_loop': 'p_loop',
              'p_loop_fast': 'p_loop', 'brake': 'brake'}
    for model, want in expect.items():
        groups = [analyze_group(*_simulate(model, acc)) for acc in (100, 30)]
        label, lines = verdict(groups)
        single, _ = verdict(groups[:1])
        ok = label == want and single == want
        failures += not ok
        print('\n######## 合成模型 %s：两档判定 %s，单档判定 %s  %s'
              % (model, label, single, '✓' if ok else '✗'))
        for g in groups:
            print_group(g)
        for line in lines:
            print(line)
    print('\n自检%s' % ('全部通过' if not failures else '失败 %d 项' % failures))
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--selftest', action='store_true')
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--analyze', nargs='+', metavar='NPZ')
    ap.add_argument('--plot', action='store_true', help='--analyze: save a PNG next to the first file')
    ap.add_argument('--arm', default='A', choices=['A', 'B'])
    ap.add_argument('--joint', type=int, default=7, choices=range(1, 8), metavar='1..7')
    ap.add_argument('--center', type=float, help='--plan only; --execute uses the measured angle')
    ap.add_argument('--amp', type=float, default=15.0, help='half sweep, deg')
    ap.add_argument('--speeds', default='10,20,40,50', help='deg/s')
    ap.add_argument('--vel-ratio', type=int, default=32)
    ap.add_argument('--acc-ratios', default='100,30')
    ap.add_argument('--cmd-acc', type=float, default=2000.0, help='command corner accel, deg/s^2')
    ap.add_argument('--dwell', type=float, default=1.0)
    ap.add_argument('--hz', type=float, default=250.0)
    ap.add_argument('--approach-speed', type=float, default=8.0)
    ap.add_argument('--abort-err', type=float, default=10.0)
    ap.add_argument('--hold-tol', type=float, default=1.5)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(1 if selftest() else 0)
    if args.analyze:
        groups = [analyze_group(*load_group(p)) for p in args.analyze]
        groups.sort(key=lambda g: -g['meta']['acc_ratio'])
        for g in groups:
            print_group(g)
        print()
        for line in verdict(groups)[1]:
            print(line)
        if args.plot:
            plot_groups(groups, os.path.splitext(args.analyze[0])[0] + '.png')
        return

    speeds = [float(x) for x in args.speeds.split(',')]
    acc_ratios = [int(x) for x in args.acc_ratios.split(',')]
    axis = read_axis(args.arm, args.joint)
    if args.plan:
        if args.center is None:
            raise SystemExit('--plan 需要 --center（预览用；--execute 用实测角度）')
        _, problems = print_plan(args, axis, args.center, speeds, acc_ratios)
        sys.exit(1 if problems else 0)
    try:
        execute(args, axis, speeds, acc_ratios)
    except KeyboardInterrupt:
        print('\n[Ctrl+C] 中断')


if __name__ == '__main__':
    main()
