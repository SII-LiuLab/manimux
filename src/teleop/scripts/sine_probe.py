#!/usr/bin/env python3
"""Small single-joint sine probe; default is an offline plan, --execute moves.

Direct teleop SDK targets, independent send/read deadlines, no IK or smoothing.
Logs host send/read timestamps, SDK target readback and measured positions.
See docs/scripts.md#sine_probepy for usage and measurement limits.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIELDS = ('joint_cmd_pos', 'fb_joint_cmd', 'fb_joint_pos')
FRAME_HEADER = ['read_begin_ns', 'read_end_ns', 'frame_serial', 'in_frame_serial',
                'cur_state', 'err_code'] + [f'{k}_j{j}' for k in FIELDS for j in range(1, 8)]
SENT_HEADER = ['send_begin_ns', 'send_end_ns'] + [f'command_j{j}' for j in range(1, 8)]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Enable and move the selected arm.')
    mode.add_argument('--analyze', type=Path, help='Analyze an existing output directory offline.')
    p.add_argument('--arm', choices=['A', 'B'], default='A')
    p.add_argument('--joint', type=int, choices=range(1, 8), default=7)
    p.add_argument('--amp-deg', type=float, default=1.0, help='Peak displacement, <=3 degrees.')
    p.add_argument('--freq-hz', type=float, default=0.5)
    p.add_argument('--hz', type=float, default=250.0, help='Host command update rate, 20..500 Hz.')
    p.add_argument('--read-hz', type=float, default=1000.0, help='Host polling rate, <=2000 Hz.')
    p.add_argument('--cycles', type=int, default=6, help='Full-amplitude cycles, 3..30.')
    p.add_argument('--ramp-s', type=float, default=2.0, help='Raised-cosine amplitude ramp at each end.')
    p.add_argument('--output', type=Path, help='New directory; default bench/results/sine_TIMESTAMP.')
    return p


def validate(a):
    for key, lo, hi in [('amp_deg', .01, 3), ('freq_hz', .1, 2), ('hz', 20, 500),
                        ('read_hz', 20, 2000), ('ramp_s', 1, 10), ('cycles', 3, 30)]:
        v = getattr(a, key)
        if not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f'{key} must be finite and in [{lo}, {hi}]')
    if a.read_hz < a.hz:
        raise ValueError('read-hz must be >= hz')


def offset(t, a):
    """Zero position/velocity at start and end; pure sine on the plateau."""
    end = 2 * a.ramp_s + a.cycles / a.freq_hz
    edge = min(t, end - t)
    if edge <= 0:
        return 0.0
    envelope = .5 * (1 - math.cos(math.pi * edge / a.ramp_s)) if edge < a.ramp_s else 1.0
    return a.amp_deg * envelope * math.sin(2 * math.pi * a.freq_hz * t)


def bounds(a):
    w, r = 2 * math.pi * a.freq_hz, math.pi / a.ramp_s
    return a.amp_deg * (w + r / 2), a.amp_deg * (w*w + w*r + r*r / 2)


def snapshot(conn, idx, clock=time.perf_counter_ns):
    begin = clock()
    s = conn.subscribe()
    o, i, st = s['outputs'][idx], s['inputs'][idx], s['states'][idx]
    values = [np.asarray(v, dtype=float).copy() for v in
              (i['joint_cmd_pos'], o['fb_joint_cmd'], o['fb_joint_pos'])]
    if any(v.shape != (7,) or not np.isfinite(v).all() for v in values):
        raise RuntimeError('Invalid SDK target/feedback vector')
    return [begin, clock(), int(o['frame_serial']), int(i['in_frame_serial']),
            int(st['cur_state']), int(st['err_code']), *np.concatenate(values).tolist()]


def guard(row, command, center, a):
    if row[4] != 1 or row[5]:
        raise RuntimeError(f'Robot state={row[4]}, error={row[5]}')
    measured = np.asarray(row[20:27])
    if np.max(np.abs(measured - command)) > 3:
        raise RuntimeError('Command/feedback error exceeds 3 degrees')
    travel = np.abs(measured - center)
    allowed = np.full(7, .5)
    allowed[a.joint - 1] = a.amp_deg + 1
    if np.any(travel > allowed):
        raise RuntimeError('Measured joint left the probe/holding envelope')


def stream(conn, idx, center, a, frames, sent, meta, send,
           clock=time.perf_counter_ns, sleep=time.sleep):
    """Single SDK owner; poll independently of sends. Never replay overdue ticks."""
    start = clock()
    meta['start_ns'] = start
    duration = 2 * a.ramp_s + a.cycles / a.freq_hz + 1.0  # final center hold
    next_send = next_read = start
    send_dt, read_dt = round(1e9 / a.hz), round(1e9 / a.read_hz)
    last_frame, fresh_ns, last_send = None, start, start
    command, row = center.copy(), None
    next_print = start
    while (clock() - start) / 1e9 < duration:
        now = clock()
        if now >= next_read:
            row = snapshot(conn, idx, clock)
            guard(row, command, center, a)
            if row[2] != last_frame:
                frames.append(row)
                last_frame, fresh_ns = row[2], row[1]
            after = clock()
            next_read += read_dt
            if next_read <= after:
                next_read = after + read_dt
        now = clock()
        if now - fresh_ns > 100_000_000:
            raise RuntimeError('Feedback frame did not advance for 100 ms')
        if now >= next_send:
            if now - last_send > max(50_000_000, 5 * send_dt):
                raise RuntimeError('Command loop stalled; refusing a catch-up jump')
            command = center.copy()
            command[a.joint - 1] += offset((now - start) / 1e9, a)
            guard(row, command, center, a)
            send(conn, {a.arm: command.tolist()})
            sent.append([now, clock(), *command.tolist()])
            last_send = now
            next_send += send_dt
            if next_send <= clock():
                next_send = clock() + send_dt
        if now >= next_print:
            j = a.joint - 1
            print(f't={(now-start)/1e9:6.2f}s sent={command[j]:8.3f} '
                  f'target={row[6+j]:8.3f} fb_cmd={row[13+j]:8.3f} '
                  f'actual={row[20+j]:8.3f} deg', flush=True)
            next_print = clock() + 500_000_000
        sleep(max(0, min(next_send, next_read) - clock()) / 1e9)


def execute(a):
    import config
    from bench.servo_lag_probe import read_axis
    from drivers.arm_driver import ArmDriver, RobotConnection, send_joint_commands

    axes = [read_axis(a.arm, j) for j in range(1, 8)]
    for j, (lo, hi) in config.JOINT_LIMIT_OVERRIDE.get(a.arm, {}).items():
        axes[j]['lim_lo'] = max(axes[j]['lim_lo'], lo)
        axes[j]['lim_hi'] = min(axes[j]['lim_hi'], hi)
    v, acc = bounds(a)
    axis = axes[a.joint - 1]
    if v > min(axis['vel_max'] * config.VEL_RATIO / 100 * .9, config.MAX_JOINT_RATE_DEG_S):
        raise ValueError('Sine velocity exceeds the configured command budget')
    if acc > axis['acc_max'] * config.ACC_RATIO / 100 * .9:
        raise ValueError('Sine acceleration exceeds the controller budget')
    out = a.output or ROOT / 'bench/results' / f'sine_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()%1000000000:09d}'
    out.mkdir(parents=True, exist_ok=False)
    meta = {**vars(a), 'output': str(out), 'analyze': None,
            'vel_ratio': config.VEL_RATIO, 'acc_ratio': config.ACC_RATIO,
            'status': 'initializing', 'timestamp_clock': 'host perf_counter_ns'}
    frames, sent, conn, drv = [], [], None, None
    owns_arm = False
    try:
        conn = RobotConnection(config.ROBOT_IP)
        drv = ArmDriver(conn, a.arm, SimpleNamespace(VEL_RATIO=config.VEL_RATIO,
                        ACC_RATIO=config.ACC_RATIO, ARM_STATE=1))
        row = snapshot(conn, drv.idx)
        center = np.asarray(row[20:27])
        if row[4] not in (0, 1) or row[5]:
            raise RuntimeError('Requires fault-free disabled or position mode; no automatic fault clearing')
        if np.max(np.abs(np.asarray(row[6:13]) - center)) > .5:
            raise RuntimeError('Existing SDK target differs from feedback by >0.5 degrees')
        for j, q in enumerate(center):
            amp = a.amp_deg if j == a.joint - 1 else 0
            if q - amp < axes[j]['lim_lo'] + config.LIMIT_MARGIN_DEG or q + amp > axes[j]['lim_hi'] - config.LIMIT_MARGIN_DEG:
                raise ValueError(f'J{j+1} range is too close to a joint limit')
        meta.update(center_deg=center.tolist(), controller_version=conn.version)
        print(f'Arm {a.arm} J{a.joint}, center={center[a.joint-1]:.3f} deg; output={out}')
        # prepare can fail after issuing a mode switch; still stop/disable on that path.
        owns_arm = drv.engaged = True
        drv.prepare()
        st = drv.state()
        if st['vel_ratio'] != config.VEL_RATIO or st['acc_ratio'] != config.ACC_RATIO:
            raise RuntimeError('Controller speed/acceleration ratio readback mismatch')
        prepared = snapshot(conn, drv.idx)
        if prepared[4] != 1 or prepared[5]:
            raise RuntimeError('Controller left position mode during preparation')
        if np.max(np.abs(np.asarray(prepared[20:27]) - center)) > .5:
            raise RuntimeError('Arm moved during preparation; refusing the old anchor')
        center = np.asarray(prepared[20:27])
        # Recheck full sine range at the actual post-prepare anchor.
        for j, q in enumerate(center):
            amp = a.amp_deg if j == a.joint - 1 else 0
            if q - amp < axes[j]['lim_lo'] + config.LIMIT_MARGIN_DEG or q + amp > axes[j]['lim_hi'] - config.LIMIT_MARGIN_DEG:
                raise ValueError(f'J{j+1} post-prepare range is too close to a joint limit')
        meta['center_deg'] = center.tolist()
        stream(conn, drv.idx, center, a, frames, sent, meta, send_joint_commands)
        meta['status'] = 'completed'
    except BaseException as exc:
        meta.update(status='aborted', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            if owns_arm:
                if meta['status'] != 'completed':
                    drv.stop()
                drv.disable()
        finally:
            try:
                if conn is not None:
                    conn.close()
            finally:
                for name, header, data in [('frames.csv', FRAME_HEADER, frames), ('sent.csv', SENT_HEADER, sent)]:
                    with (out / name).open('w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                        writer.writerows(data)
                (out / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
                print(f'Saved {len(sent)} commands / {len(frames)} unique feedback frames: {out}')
    analyze(out)


def fit_sine(t, q, frequency):
    w = 2 * np.pi * frequency
    design = np.column_stack((np.sin(w*t), np.cos(w*t), np.ones_like(t)))
    coeff = np.linalg.lstsq(design, q, rcond=None)[0]
    residual = q - design @ coeff
    return dict(amplitude_deg=float(np.hypot(*coeff[:2])),
                phase_rad=float(np.arctan2(coeff[1], coeff[0])),
                residual_rms_deg=float(np.sqrt(np.mean(residual**2))))


def phase_lag(reference, response, frequency):
    phase = (reference['phase_rad'] - response['phase_rad'] + np.pi) % (2*np.pi) - np.pi
    return float(phase / (2*np.pi*frequency) * 1000)


def plot_tracking(out, meta, signals, result):
    """Two raw traces over two steady cycles; retain the recorded time offset."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lo, fit_hi = result['fit_window_s']
    hi = min(fit_hi, lo + 2 / meta['freq_hz'])
    ts, command = signals['sent']
    tf, measured = signals['fb_joint_pos']
    center = meta.get('center_deg', [float(command[0])] * 7)[meta['joint'] - 1]
    lag = result['lag_ms']['sent_to_fb_joint_pos']
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#CBD5E1', 'axes.labelcolor': '#475569',
                         'xtick.color': '#64748B', 'ytick.color': '#64748B'}):
        fig, ax = plt.subplots(figsize=(11, 4.8), facecolor='white')
        try:
            # Include adjacent samples at the bounds for complete line segments.
            for t, q, label, color, style, width, drawstyle in (
                (ts, command, 'Command', '#2563EB', '--', 2.1, 'steps-post'),
                (tf, measured, 'Measured position', '#EA580C', '-', 2.3, 'default'),
            ):
                begin = max(0, np.searchsorted(t, lo) - 1)
                end = min(len(t), np.searchsorted(t, hi, side='right') + 1)
                ax.plot(t[begin:end], q[begin:end] - center, label=label,
                        color=color, linestyle=style, linewidth=width, drawstyle=drawstyle)
            ax.set(xlim=(lo, hi), xlabel='Time from start (s)', ylabel='Joint displacement (deg)')
            ax.margins(y=.20)
            ax.grid(axis='y', color='#E2E8F0', linewidth=.7)
            ax.set_axisbelow(True)
            ax.legend(loc='upper right', frameon=False, ncols=2, fontsize=10)
            title = f'Sine tracking  /  Arm {meta.get("arm", "?")} · J{meta["joint"]}'
            if meta.get('synthetic'):
                title = 'SYNTHETIC PREVIEW  /  ' + title
            fig.text(.09, .94, title, fontsize=16, weight='bold', color='#0F172A')
            lag_text = 'Lag unavailable' if lag is None else f'Phase-derived lag {lag:.1f} ms'
            fig.text(.09, .885, f'{meta["freq_hz"]:g} Hz sine   ·   '
                     f'{result["send_hz"]:.1f} Hz observed command rate   ·   {lag_text}',
                     fontsize=10, color='#64748B')
            fig.text(.09, .025, f'Fit window {lo:.2f}–{fit_hi:.2f} s  ·  '
                     'Recorded host timestamps  ·  Positive lag: measured position trails command',
                     fontsize=8, color='#64748B')
            fig.subplots_adjust(left=.09, right=.98, bottom=.17, top=.82)
            for ext in ('png', 'pdf'):
                fig.savefig(out / f'tracking.{ext}', dpi=200, facecolor='white')
        finally:
            plt.close(fig)
    print(f'Plot: {out / "tracking.png"} (also saved as PDF)')


def analyze(out):
    meta = json.loads((out / 'meta.json').read_text())
    sent = np.genfromtxt(out / 'sent.csv', delimiter=',', names=True, ndmin=1)
    frames = np.genfromtxt(out / 'frames.csv', delimiter=',', names=True, ndmin=1)
    if len(sent) < 3 or len(frames) < 3 or 'start_ns' not in meta:
        raise ValueError('Insufficient data for phase analysis')
    origin = meta['start_ns']
    ts = (sent['send_begin_ns'] - origin) / 1e9
    tf = ((frames['read_begin_ns'] - origin) + (frames['read_end_ns'] - origin)) / 2e9
    # Discard a full cycle after the ramp, and require >=2 complete common cycles.
    lo = meta['ramp_s'] + 1 / meta['freq_hz']
    hi = min(meta['ramp_s'] + meta['cycles'] / meta['freq_hz'], ts[-1], tf[-1])
    if hi - lo < 2 / meta['freq_hz']:
        raise ValueError('Insufficient steady sine data (need two cycles after settling)')
    j = meta['joint']
    signals = {'sent': (ts, sent[f'command_j{j}'])}
    signals.update({k: (tf, frames[f'{k}_j{j}']) for k in FIELDS})
    fits = {}
    for key, (t, q) in signals.items():
        mask = (t >= lo) & (t <= hi)
        if mask.sum() < 20:
            raise ValueError(f'Insufficient samples for {key}')
        fits[key] = fit_sine(t[mask], q[mask], meta['freq_hz'])
    result = {'fit_window_s': [lo, hi], 'fits': fits, 'lag_ms': {}}
    for ref, response in [('sent', 'joint_cmd_pos'), ('sent', 'fb_joint_pos'),
                          ('joint_cmd_pos', 'fb_joint_pos'), ('fb_joint_cmd', 'fb_joint_pos')]:
        name = f'{ref}_to_{response}'
        if min(fits[ref]['amplitude_deg'], fits[response]['amplitude_deg']) < .01:
            result['lag_ms'][name] = None
        else:
            result['lag_ms'][name] = phase_lag(fits[ref], fits[response], meta['freq_hz'])
    for key, t in [('send', ts), ('feedback_observed', tf)]:
        result[key + '_hz'] = float((len(t)-1) / (t[-1]-t[0]))
        result[key + '_dt_ms_p50_p99_max'] = np.percentile(np.diff(t)*1000, [50, 99, 100]).tolist()
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print('Positive lag means response trails reference; phase is modulo one period. '
          'Host timestamps do not identify controller receive or encoder capture times.')
    plot_tracking(out, meta, signals, result)


def main():
    p = parser()
    a = p.parse_args()
    try:
        if a.analyze:
            analyze(a.analyze)
            return
        validate(a)
        v, acc = bounds(a)
        print(f'Arm {a.arm} J{a.joint}: +/-{a.amp_deg:g} deg, {a.freq_hz:g} Hz; '
              f'send {a.hz:g} Hz, poll {a.read_hz:g} Hz; '
              f'{2*a.ramp_s+a.cycles/a.freq_hz+1:g} s including ramps and final hold. '
              f'Conservative peak bounds: {v:.2f} deg/s, {acc:.2f} deg/s^2.')
        if a.execute:
            execute(a)
        else:
            print('Offline plan only. --execute enables the selected arm at its current pose, '
                  'runs the sine, then disables it. Keep the selected joint sweep clear; '
                  'the script does not check Cartesian/self-collision clearance. '
                  'Do not run alongside another command sender.')
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, RuntimeError) as exc:
        p.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
