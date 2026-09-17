#!/usr/bin/env python3
"""Offline comparison: ik_solver.ArmIK vs. diff_ik.DiffIKSolver, same
recorded trajectory, no robot. See docs/algos.md#comparek_ikpy for
methodology and how to read the report.

Runs two fully independent closed-loop pipelines (own Retargeter, own
SafetyGate, own evolving joint state) over the same raw resampled XR
frame stream -- not the same per-frame target list, because retargeting's
Cartesian rate limiter reads back q_now every frame (see docs/algos.md),
so the two solvers' target streams legitimately diverge as their joint
trajectories diverge. That divergence is the point of the comparison,
not a bug in it.

The analytic pipeline's NullSpaceController stays disabled here --
DiffIKSolver's own redundancy handling (mu_nullspace, a cost term, not
NullSpaceController's reentrant probe-based approach) isn't equivalent
mechanically, so enabling NullSpaceController on the analytic side
wouldn't make this an apples-to-apples redundancy comparison; it would
just add a second, unrelated variable. The diff-IK pipeline DOES run
with mu_nullspace=1000 (see docs/algos.md#diff_ikpy for what that is and
why traj_c specifically doesn't show much effect from it -- J6, the
joint traj_c stresses, has ~0 null-space leverage, a known geometric
fact, not a bug in this addition).

Usage:
    python3 bench/compare_ik.py data/traj_b.jsonl
    python3 bench/compare_ik.py data/traj_c.jsonl --arm A
    python3 bench/compare_ik.py --synth
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from algos.diff_ik import DiffIKSolver
from algos.ik_solver import ArmIK
from algos.kinematics import ArmKinematics
from algos.retarget import Retargeter
from algos.safety import LowPass, SafetyGate
from replay import load_traj, resample_to_control_rate, synth_traj

Q0 = [44.04, -62.57, -8.92, -57.21, 1.45, -4.39, 2.1]   # same start as replay.py
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def run_pipeline(sim, hand, arm, cfg, kind):
    """kind: 'analytic' (ik_solver.ArmIK) or 'diff' (diff_ik.DiffIKSolver).
    Returns one record dict per resampled step -- always, even when the
    frame is disengaged or rejected, so position/J6-angle traces stay
    continuous (frozen at the last accepted q_cmd) for the report plots.
    """
    arm_type = 0 if arm == 'A' else 1
    ik = ArmIK(arm_type=arm_type, config_path=cfg.KINE_CFG,
              pos_tol_mm=cfg.IK_POS_TOL_MM, rot_tol_deg=cfg.IK_ROT_TOL_DEG,
              max_step_deg=cfg.IK_MAX_STEP_DEG,
              limit_margin_deg=cfg.LIMIT_MARGIN_DEG,
              limit_override=cfg.JOINT_LIMIT_OVERRIDE.get(arm))
    lo, hi = ik.lim_n, ik.lim_p
    gate = SafetyGate(lo, hi, cfg.MAX_JOINT_RATE_DEG_S, cfg.LIMIT_MARGIN_DEG,
                      cfg.MAX_TRACKING_ERR_DEG, cfg.MAX_CONSEC_REJECT,
                      cfg.XR_STALE_S, dt_max=cfg.MAX_STEP_DT_S,
                      max_tracking_err_s=cfg.MAX_TRACKING_ERR_S)
    # No nullspace for either pipeline -- see module docstring.
    rt = Retargeter(cfg, arm, lowpass=LowPass(cfg.LOWPASS_HZ))

    diff_solver = None
    if kind == 'diff':
        kin = ArmKinematics(ik.cfg['DH'][arm_type])
        vmax = [r[2] for r in ik.cfg['PNVA'][arm_type]]
        # mu_nullspace=1000: same value validated in diff_ik.py's own
        # self-test (visibly pulls a near-limit joint to safety without
        # disturbing task tracking there). Its real effect here is
        # configuration-dependent (see docs/algos.md#diff_ikpy) -- this
        # run is what actually measures it, not assumed in advance.
        diff_solver = DiffIKSolver(kin, ik.kine, lo, hi, vmax,
                                   limit_margin_deg=cfg.LIMIT_MARGIN_DEG,
                                   pos_tol_mm=cfg.IK_POS_TOL_MM,
                                   rot_tol_deg=cfg.IK_ROT_TOL_DEG,
                                   mu_nullspace=1000.0)

    q_meas = list(Q0)   # "measured" joints -- replay assumes perfect servo tracking
    q_cmd = list(Q0)
    period = 1.0 / cfg.CONTROL_HZ
    records = []

    for i, f in enumerate(sim):
        engaged = rt.update(f, hand, q_cmd, ik.fk_xyzabc, ik.nsp_dir,
                            dt=period, solve=None)
        rec = {'i': i, 'engaged': engaged, 'ok': False, 'reason': None,
              'solve_ms': None, 'step_deg': None, 'clamped': False,
              'min_margin_deg': None}

        if engaged:
            t0 = time.perf_counter()
            if kind == 'analytic':
                r = ik.solve_with_backoff(rt.target_xyzabc, q_cmd,
                                          zsp_dir=rt.zsp_dir,
                                          arm_angle=rt.arm_angle)
                solve_ms = (time.perf_counter() - t0) * 1e3
            else:
                r = diff_solver.solve(rt.target_xyzabc, q_cmd, period)
                solve_ms = r.solve_time_ms
            rec['solve_ms'] = solve_ms
            rec['reason'] = r.reason

            if r.ok:
                v = gate.check(r.joints, q_cmd, q_meas, xr_age_s=0.0, dt=period)
                rec['reason'] = 'ok' if v.ok else v.reason
                if v.ok:
                    rec['ok'] = True
                    rec['clamped'] = v.clamped
                    rec['step_deg'] = max(abs(a - b)
                                          for a, b in zip(v.joints, q_cmd))
                    rec['min_margin_deg'] = r.min_margin_deg
                    q_cmd = v.joints
                    q_meas = list(q_cmd)   # perfect tracking assumption

        x = ik.fk_xyzabc(q_cmd)
        rec['x'], rec['y'], rec['z'] = x[0], x[1], x[2]
        rec['j6_deg'] = q_cmd[5]
        rec['j6_margin_deg'] = min(hi[5] - q_cmd[5], q_cmd[5] - lo[5])
        records.append(rec)

    return records


def pctl(vals, p):
    return float(np.percentile(vals, p)) if vals else float('nan')


def summarize(records, label):
    engaged = [r for r in records if r['engaged']]
    ok = [r for r in engaged if r['ok']]
    steps = [r['step_deg'] for r in ok]
    solve_ms = [r['solve_ms'] for r in engaged if r['solve_ms'] is not None]
    near_limit = [r for r in ok if r['min_margin_deg'] is not None
                 and r['min_margin_deg'] < 2 * config.LIMIT_MARGIN_DEG]
    reasons = {}
    for r in engaged:
        if not r['ok']:
            reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
    return {
        'label': label,
        'n_total': len(records),
        'n_engaged': len(engaged),
        'n_ok': len(ok),
        'reject_rate': 1.0 - len(ok) / max(len(engaged), 1),
        'reject_reasons': reasons,
        'near_limit_rate': len(near_limit) / max(len(ok), 1),
        'step_p50': pctl(steps, 50), 'step_p90': pctl(steps, 90),
        'step_p99': pctl(steps, 99), 'step_max': pctl(steps, 100),
        'solve_p50': pctl(solve_ms, 50), 'solve_p99': pctl(solve_ms, 99),
        'solve_max': pctl(solve_ms, 100),
    }


def write_csv(path, *labeled_records):
    """labeled_records: (label, records) pairs, written to one combined CSV."""
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['solver', 'i', 'engaged', 'ok', 'reason', 'solve_ms',
                   'step_deg', 'clamped', 'min_margin_deg', 'x', 'y', 'z',
                   'j6_deg', 'j6_margin_deg'])
        for label, records in labeled_records:
            for r in records:
                w.writerow([label, r['i'], int(r['engaged']), int(r['ok']),
                           r['reason'], r['solve_ms'], r['step_deg'],
                           int(r['clamped']), r['min_margin_deg'],
                           r['x'], r['y'], r['z'], r['j6_deg'],
                           r['j6_margin_deg']])


def find_limit_window(records, half_width=100):
    """Index range around where J6's margin is smallest in this run."""
    idx = min(range(len(records)), key=lambda i: records[i]['j6_margin_deg'])
    lo = max(0, idx - half_width)
    hi = min(len(records), idx + half_width)
    return lo, hi, idx


def cart_speed(records, period):
    pos = np.array([[r['x'], r['y'], r['z']] for r in records])
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1) / period
    return np.concatenate([[0.0], d])   # align length with records


def plot_j6_window(rec_a, rec_d, lo, hi, period, path):
    t = np.arange(lo, hi) * period
    speed_a = cart_speed(rec_a, period)[lo:hi]
    speed_d = cart_speed(rec_d, period)[lo:hi]
    margin_a = [rec_a[i]['j6_margin_deg'] for i in range(lo, hi)]
    margin_d = [rec_d[i]['j6_margin_deg'] for i in range(lo, hi)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(t, speed_a, label='analytic (ArmIK)', color='#d62728')
    ax1.plot(t, speed_d, label='diff-IK (QP)', color='#1f77b4')
    ax1.set_ylabel('end-effector speed (mm/s)')
    ax1.legend()
    ax1.set_title('Cartesian speed through the J6-limit-approach window')

    ax2.plot(t, margin_a, label='analytic J6 margin', color='#d62728')
    ax2.plot(t, margin_d, label='diff-IK J6 margin', color='#1f77b4')
    ax2.axhline(config.LIMIT_MARGIN_DEG, color='gray', linestyle='--',
               label='limit_margin_deg')
    ax2.set_ylabel('J6 margin to limit (deg)')
    ax2.set_xlabel('time (s)')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_step_histogram(rec_a, rec_d, path):
    steps_a = [r['step_deg'] for r in rec_a if r['ok']]
    steps_d = [r['step_deg'] for r in rec_d if r['ok']]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(steps_a + steps_d + [config.MAX_JOINT_STEP_DEG]), 60)
    ax.hist(steps_a, bins=bins, alpha=0.6, label='analytic (ArmIK)',
           color='#d62728', density=True)
    ax.hist(steps_d, bins=bins, alpha=0.6, label='diff-IK (QP)',
           color='#1f77b4', density=True)
    ax.axvline(config.MAX_JOINT_STEP_DEG, color='black', linestyle='--',
              label='MAX_JOINT_STEP_DEG budget')
    ax.set_xlabel('per-frame max joint step (deg)')
    ax.set_ylabel('density')
    ax.set_title('Per-frame joint-step distribution (accepted frames only)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(path, traj_name, sum_a, sum_d, window):
    lo, hi, idx = window
    lines = []
    lines.append('# IK comparison report: %s\n' % traj_name)
    lines.append('Analytic = `ik_solver.ArmIK` (existing pipeline). '
                 'Diff-IK = `diff_ik.DiffIKSolver` (QP prototype). '
                 'Both run through the same `Retargeter`+`SafetyGate` '
                 'wiring, null-space disabled for both. See '
                 '`docs/algos.md#comparek_ikpy` for methodology.\n')

    lines.append('## 1. Reject rate\n')
    lines.append('| | analytic | diff-IK |')
    lines.append('|---|---|---|')
    lines.append('| engaged frames | %d | %d |'
                 % (sum_a['n_engaged'], sum_d['n_engaged']))
    lines.append('| reject rate | %.2f%% | %.2f%% |'
                 % (sum_a['reject_rate'] * 100, sum_d['reject_rate'] * 100))
    lines.append('| accepted-but-within-2x-margin rate | %.2f%% | %.2f%% |'
                 % (sum_a['near_limit_rate'] * 100, sum_d['near_limit_rate'] * 100))
    lines.append('| reject reasons | `%s` | `%s` |'
                 % (sum_a['reject_reasons'], sum_d['reject_reasons']))
    lines.append('')

    lines.append('## 2. Per-frame joint-step distribution (deg, accepted frames)\n')
    lines.append('| | analytic | diff-IK | budget |')
    lines.append('|---|---|---|---|')
    for k, label in (('step_p50', 'p50'), ('step_p90', 'p90'),
                     ('step_p99', 'p99'), ('step_max', 'max')):
        lines.append('| %s | %.4f | %.4f | %.2f |'
                     % (label, sum_a[k], sum_d[k], config.MAX_JOINT_STEP_DEG))
    lines.append('')
    lines.append('![step histogram](step_histogram_%s.png)\n' % traj_name)

    lines.append('## 3. J6-limit-approach window\n')
    lines.append('Window centered on frame %d (tightest J6 margin in the '
                 'analytic run), +/-%d frames.\n' % (idx, hi - idx))
    lines.append('![j6 window](j6_window_%s.png)\n' % traj_name)

    lines.append('## 4. Solve time (ms, engaged frames)\n')
    lines.append('| | analytic | diff-IK | reference |')
    lines.append('|---|---|---|---|')
    lines.append('| p50 | %.4f | %.4f | TracIK unconstrained ~0.9 |'
                 % (sum_a['solve_p50'], sum_d['solve_p50']))
    lines.append('| p99 | %.4f | %.4f | |' % (sum_a['solve_p99'], sum_d['solve_p99']))
    lines.append('| max | %.4f | %.4f | budget 4.0 (250Hz) |'
                 % (sum_a['solve_max'], sum_d['solve_max']))
    lines.append('')

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description='Offline ArmIK vs DiffIKSolver comparison')
    ap.add_argument('traj', nargs='?', help='xr_source.py --record jsonl')
    ap.add_argument('--synth', action='store_true')
    ap.add_argument('--hand', default='right', choices=['left', 'right'])
    ap.add_argument('--arm', default=None, choices=['A', 'B'])
    ap.add_argument('--scale', type=float, default=None)
    args = ap.parse_args()

    if args.scale is not None:
        config.SCALE = args.scale
    arm = args.arm or config.ARM_OF_HAND[args.hand]

    if args.synth or not args.traj:
        print('using synthetic trajectory (--synth or no path given)\n')
        frames = synth_traj()
        traj_name = 'synth'
    else:
        frames = load_traj(args.traj)
        traj_name = os.path.splitext(os.path.basename(args.traj))[0]
        print('loaded %d frames from %s' % (len(frames), args.traj))
    if not frames:
        print('no usable frames'); return 1

    config.summary()
    sim = resample_to_control_rate(frames, config.CONTROL_HZ)
    period = 1.0 / config.CONTROL_HZ
    print('\n%d recorded frames -> %d control periods @ %.0fHz\n'
         % (len(frames), len(sim), config.CONTROL_HZ))

    print('running analytic pipeline (ArmIK)...')
    rec_a = run_pipeline(sim, args.hand, arm, config, 'analytic')
    print('running diff-IK pipeline (DiffIKSolver)...')
    rec_d = run_pipeline(sim, args.hand, arm, config, 'diff')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_csv(os.path.join(RESULTS_DIR, 'compare_%s.csv' % traj_name),
             ('analytic', rec_a), ('diff', rec_d))

    sum_a = summarize(rec_a, 'analytic')
    sum_d = summarize(rec_d, 'diff')

    window = find_limit_window(rec_a)
    plot_j6_window(rec_a, rec_d, window[0], window[1], period,
                   os.path.join(RESULTS_DIR, 'j6_window_%s.png' % traj_name))
    plot_step_histogram(rec_a, rec_d,
                        os.path.join(RESULTS_DIR, 'step_histogram_%s.png' % traj_name))
    report_path = os.path.join(RESULTS_DIR, 'report_%s.md' % traj_name)
    write_report(report_path, traj_name, sum_a, sum_d, window)

    print('\n=== summary ===')
    for s in (sum_a, sum_d):
        print('%-10s engaged=%-6d reject=%.2f%% step(p50/p99/max)=%.3f/%.3f/%.3f '
             'solve(p50/p99/max)ms=%.3f/%.3f/%.3f'
             % (s['label'], s['n_engaged'], s['reject_rate'] * 100,
                s['step_p50'], s['step_p99'], s['step_max'],
                s['solve_p50'], s['solve_p99'], s['solve_max']))
    print('\nwrote %s' % report_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
