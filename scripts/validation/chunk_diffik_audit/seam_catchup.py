"""Audit inherited lag, incoming reference seams, and recovery at pi/a80.

The fixed-target branch is a diagnostic at each seam, not a counterfactual rollout.
"""
import copy
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import audit
from manimux.embodiments.robot.base import RobotModel
from manimux.runtime.executors.limits import ScalarLimits, limit_step

OUT = audit.OUT / 'seam_catchup_pi_a80_v1'
TRACES = audit.OUT / 'executor_at_pi_a80_v1' / 'traces'


def sustained_time(t, error, threshold, start, span_ns=50_000_000):
    for i in range(len(t)):
        end = np.searchsorted(t, t[i] + span_ns)
        if end < len(t) and np.all(error[i:end+1] <= threshold):
            return float((t[i] - start) / 1e9)
    return None


def fixed_target(d, kin, reference, command, index):
    """Stop the incoming reference; retain the outgoing command position/velocity."""
    dt = d['dt']
    q = {g: command[index-1, j, :7].copy() for j, g in enumerate(audit.ARMS)}
    v = {g: ((command[index-1, j, :7] - command[index-2, j, :7]) / dt
             if index > 1 and d['active'][index-1] and not d['resets'][index-1] else np.zeros(7))
         for j, g in enumerate(audit.ARMS)}
    alpha = dt / (dt + 1 / (2 * np.pi * 4))
    limits = ScalarLimits(math.pi, 80., 3.14, 'isotropic', .016)
    trajectory = []
    for _ in range(round(1. / dt)):
        target = {g: q[g] + alpha * (reference[index, j, :7] - q[g]) for j, g in enumerate(audit.ARMS)}
        q, v = limit_step(target, q, v, dt_s=dt, limits=limits)
        trajectory.append(np.stack([np.r_[q[g], reference[index, j, 7]] for j, g in enumerate(audit.ARMS)]))
    poses = audit.poses(kin, np.asarray(trajectory))
    goal = audit.poses(kin, reference[index:index+1])
    error, _ = audit.err(np.broadcast_to(goal, poses.shape), poses)
    times = np.arange(len(error), dtype=np.int64) * round(dt * 1e9)
    return {g: {f'within_{threshold}_mm_s': sustained_time(times, error[:, j], threshold, 0)
                for threshold in (5, 20)} for j, g in enumerate(audit.ARMS)}


def job(row):
    d = audit.load(row)
    kin = RobotModel.from_config(d['c']['robot']['config']).kinematics
    trace_path = TRACES / row['id'] / 'tcp_f4__smooth_isotropic.npz'
    trace = np.load(trace_path)
    ref, cmd, target = trace['reference'], trace['command'], trace['target_tcp']
    rp, cp = audit.poses(kin, ref), audit.poses(kin, cmd)
    reference_error, _ = audit.err(rp, cp)
    target_error, target_rot = audit.err(target, cp)
    # Reconstruct outgoing plans only to sample at the actual executed handoff
    # time, never to score unexecuted forecast residuals.
    adapter = audit.adapter_for(d, kin, {'velocity': math.pi, 'objective': 'tcp'})
    plans = {}
    max_reference_difference = 0.
    for p in d['plans']:
        dense = []
        for g in audit.ARMS:
            side = g.removesuffix('_arm')
            tr = p['trace'][g]
            steps = [{f'{side}_ee_pose': np.asarray(pose), f'{side}_ee_joint_state': np.asarray([grip])}
                     for pose, grip in zip(tr['ik_input']['pose_xyz_wxyz'], tr['ik_input']['aperture'])]
            _, trajectory, _, _ = adapter._decode_side(side, steps, np.asarray(tr['ik_seed_joints']))
            dense.append(trajectory)
        dense = np.stack(dense, axis=1)
        ix, x, _, step_dt, begin, clock = audit.mapping(d, p, len(dense))
        if len(ix):
            max_reference_difference = max(max_reference_difference, float(abs(audit.interp(dense, x) - ref[ix]).max()))
        plans[p['id']] = dict(dense=dense, begin=begin, clock=clock, dt=step_dt)
    assert max_reference_difference < 1e-10
    boundaries = np.flatnonzero((d['ids'][1:] != d['ids'][:-1]) & d['active'][1:] & (d['ids'][:-1] != '')) + 1
    valid_boundaries = [int(i) for i in boundaries if d['active'][i-1] and not d['resets'][i]
                        and d['t'][i] - d['t'][i-1] < 50_000_000]
    age = np.full(len(ref), np.nan)
    entries = []
    for k, i in enumerate(valid_boundaries):
        end = i + 1
        while end < len(ref) and d['ids'][end] == d['ids'][i] and d['active'][end] and not d['resets'][end]:
            end += 1
        age[i:end] = (d['t'][i:end] - d['t'][i]) / 1e9
        old = plans[d['ids'][i-1]]
        x = old['begin'] + (int(d['t'][i]) - old['clock']) / old['dt']
        covered = old['begin'] <= x <= len(old['dense']) - 1
        old_pose = audit.poses(kin, audit.interp(old['dense'], np.asarray([x])))[0] if covered else None
        frozen = fixed_target(d, kin, ref, cmd, i)
        arms = {}
        for j, g in enumerate(audit.ARMS):
            inherited = rp[i-1, j, :3, 3] - cp[i-1, j, :3, 3]
            incoming = rp[i, j, :3, 3] - cp[i-1, j, :3, 3]
            adjacent_change = rp[i, j, :3, 3] - rp[i-1, j, :3, 3]
            arm = dict(inherited_lag_mm=float(np.linalg.norm(inherited)*1000),
                incoming_distance_from_previous_command_mm=float(np.linalg.norm(incoming)*1000),
                adjacent_reference_change_mm=float(np.linalg.norm(adjacent_change)*1000),
                reference_error_after_first_command_mm=float(reference_error[i, j]),
                actual_recovery={f'within_{threshold}_mm_s': sustained_time(d['t'][i:end], reference_error[i:end, j], threshold, d['t'][i]) for threshold in (5, 20)},
                frozen_incoming_target_recovery=frozen[g])
            if covered:
                motion = old_pose[j, :3, 3] - rp[i-1, j, :3, 3]
                jump = rp[i, j, :3, 3] - old_pose[j, :3, 3]
                assert np.max(abs(inherited + motion + jump - incoming)) < 1e-12
                direction = incoming / max(np.linalg.norm(incoming), 1e-15)
                arm.update(same_time_reference_jump_mm=float(np.linalg.norm(jump)*1000),
                           outgoing_one_tick_motion_mm=float(np.linalg.norm(motion)*1000),
                           signed_projection_on_incoming_gap_mm={name: float(np.dot(vector, direction)*1000)
                               for name, vector in [('inherited_lag', inherited), ('outgoing_motion', motion), ('incoming_jump', jump)]})
            arms[g] = arm
        entries.append(dict(tick=i, seconds_from_start=float((d['t'][i]-d['t'][0])/1e9),
            outgoing_plan=d['ids'][i-1], incoming_plan=d['ids'][i],
            available_incoming_execution_s=float((d['t'][end-1]-d['t'][i])/1e9),
            same_time_outgoing_coverage=covered, arms=arms))
    bins = {}
    for name, lo, hi in [('0_100ms', 0., .1), ('100_200ms', .1, .2), ('200_400ms', .2, .4), ('400ms_plus', .4, np.inf)]:
        mask = d['masks']['all'] & (age >= lo) & (age < hi)
        bins[name] = dict(ticks=int(mask.sum()), target_command_mm=audit.stats(target_error[mask]),
            target_command_deg=audit.stats(target_rot[mask]),
            arms={g: audit.stats(target_error[mask, j]) for j, g in enumerate(audit.ARMS)})
    mask = d['masks']['all'] & np.isfinite(age)
    high = (target_error > 20.) & mask[:, None]
    qref_v = np.diff(ref[:, :, :7], axis=0) / d['dt']
    nonseam = (d['ids'][1:] == d['ids'][:-1]) & d['active'][1:] & d['active'][:-1] & ~d['resets'][1:]
    speed_saturated = np.any(abs(qref_v) >= math.pi*.99, axis=-1)
    late = nonseam & (age[1:] >= .4)
    result = dict(row=row, max_reference_reproduction_difference_rad=max_reference_difference,
        transition_count=len(boundaries), continuous_boundary_count=len(entries),
        excluded_noncontinuous_boundaries=len(boundaries)-len(entries),
        same_time_covered_boundary_count=sum(e['same_time_outgoing_coverage'] for e in entries),
        boundary_entries=entries, age_bins=bins,
        large_error_arm_ticks=int(high.sum()),
        large_error_fraction_within_200ms=float((high & (age < .2)[:, None]).sum()/high.sum()) if high.any() else None,
        available_arm_tick_fraction_within_200ms=float((mask & (age < .2)).sum()/mask.sum()) if mask.any() else None,
        late_reference_near_pi_fraction=float(speed_saturated[late].mean()) if late.any() else None,
        late_actual_error_mm=audit.stats(target_error[1:][late]),
        note='Same-time seam uses outgoing trajectory only when covered. Adjacent-tick changes include ordinary motion. '
             'Position vector projections add exactly; their norms and p95s do not. Fixed-target recovery is a stopped-reference diagnostic, '
             'not an optimized or exact closed-loop rollout. Original per-chunk seeds remain frozen.')
    audit.write(OUT / 'episodes' / f'{row["id"]}.json', result)
    return dict(episode=row['id'], seams=len(entries))


def summarize(rows):
    results = [json.loads((OUT/'episodes'/f'{r["id"]}.json').read_text()) for r in rows]
    summaries = []
    for r in results:
        arms = [a for e in r['boundary_entries'] for a in e['arms'].values()]
        observable_arms = [a for e in r['boundary_entries'] if e['available_incoming_execution_s'] >= .05 for a in e['arms'].values()]
        s = dict(row=r['row'], seams=r['continuous_boundary_count'], age_bins=r['age_bins'],
                 large_error_fraction_within_200ms=r['large_error_fraction_within_200ms'],
                 available_arm_tick_fraction_within_200ms=r['available_arm_tick_fraction_within_200ms'],
                 late_reference_near_pi_fraction=r['late_reference_near_pi_fraction'])
        for name in ('inherited_lag_mm','incoming_distance_from_previous_command_mm','same_time_reference_jump_mm'):
            s[name] = audit.stats([a[name] for a in arms if name in a])
        for threshold in (5,20):
            key=f'within_{threshold}_mm_s'
            s[f'actual_recovery_{threshold}mm'] = audit.stats([a['actual_recovery'][key] for a in arms if a['actual_recovery'][key] is not None])
            s[f'actual_not_recovered_{threshold}mm_count'] = sum(a['actual_recovery'][key] is None for a in observable_arms)
            s['actual_recovery_insufficient_window_arm_count'] = len(arms)-len(observable_arms)
            s[f'frozen_target_recovery_{threshold}mm'] = audit.stats([a['frozen_incoming_target_recovery'][key] for a in arms if a['frozen_incoming_target_recovery'][key] is not None])
            s[f'frozen_not_recovered_{threshold}mm_count'] = sum(a['frozen_incoming_target_recovery'][key] is None for a in arms)
        summaries.append(s)
        if r['row']['split']=='holdout':
            # Matched-time late-region comparison, using already completed replays.
            row=r['row'];d=audit.load(row)
            kin=RobotModel.from_config(d['c']['robot']['config']).kinematics
            base=np.load(TRACES/row['id']/'tcp_f4__smooth_isotropic.npz')
            age=np.full(len(d['t']),np.nan)
            for e in r['boundary_entries']:
                i=e['tick'];end=i+1
                while end<len(age) and d['ids'][end]==d['ids'][i] and d['active'][end] and not d['resets'][end]:end+=1
                age[i:end]=(d['t'][i:end]-d['t'][i])/1e9
            late=d['masks']['all']&(age>=.4)
            variants={
                'smooth_a80':base['command'],
                'direct_a80':np.load(TRACES/row['id']/'tcp_f4__direct_isotropic.npz')['command'],
                'smooth_no_acceleration_limit':np.load(audit.OUT/'acceleration_at_pi_v1'/'traces'/row['id']/'tcp_f4_anone.npz')['command'],
            }
            s['matched_400ms_plus_comparison']={}
            for name,command in variants.items():
                error,rotation=audit.err(base['target_tcp'],audit.poses(kin,command))
                s['matched_400ms_plus_comparison'][name]=dict(mm=audit.stats(error[late]),deg=audit.stats(rotation[late]))
    audit.write(OUT/'episode_summaries.json',summaries)
    audit.write(OUT/'validation.json',dict(episodes=len(rows),
        continuous_boundaries=sum(r['continuous_boundary_count'] for r in results),
        same_time_covered_boundaries=sum(r['same_time_covered_boundary_count'] for r in results),
        max_reference_reproduction_difference_rad=max(r['max_reference_reproduction_difference_rad'] for r in results)))
    lines=['pi rad/s, 80 rad/s^2, smooth isotropic, 4 Hz, offline TCP objective.',
           'Historical targets/seeds/scheduling are frozen. No counterfactual measured-state prediction.',
           'Error fractions describe time samples, not additive causal shares of p95.', '']
    for s in summaries:
        if s['row']['split']=='holdout':
            lines.append(f"{s['row']['task']} {s['row']['status']}: {s['row']['path']}")
            lines.append(f"  Continuous seams: {s['seams']}")
            for name in ('inherited_lag_mm','incoming_distance_from_previous_command_mm','same_time_reference_jump_mm'):
                lines.append(f"  {name}: {s[name]}")
            for name,stats in s['age_bins'].items():
                lines.append(f"  {name}: TCP target-command p95={stats['target_command_mm'].get('p95')} mm; per-arm={stats['arms']}")
            lines.append(f"  Fraction of >20 mm samples within 200 ms: {s['large_error_fraction_within_200ms']}")
            lines.append(f"  Matched late-region comparison: {s['matched_400ms_plus_comparison']}")
            lines.append(f"  Not recovered below 20 mm for 50 ms before next handoff: {s['actual_not_recovered_20mm_count']}; insufficient windows: {s['actual_recovery_insufficient_window_arm_count']}")
            lines.append('')
            print(json.dumps(s),flush=True)
    (OUT/'comparison.txt').write_text('\n'.join(lines)+'\n')
    protocol=json.loads((OUT/'protocol.json').read_text())
    protocol['report_source_sha256']=audit.digest(Path(__file__).read_bytes())
    audit.write(OUT/'protocol.json',protocol)


if __name__=='__main__':
    rows=json.loads((audit.OUT/'cohort.json').read_text())['selected']
    OUT.mkdir(parents=True,exist_ok=True)
    audit.write(OUT/'protocol.json',dict(max_velocity_rad_s=math.pi,max_acceleration_rad_s2=80.,cutoff_hz=4.,
        objective='TCP',executor='smooth',mode='isotropic',
        source_sha256=audit.digest(Path(__file__).read_bytes()),cohort_sha256=audit.digest((audit.OUT/'cohort.json').read_bytes()),
        selection='Fixed original cohort; no tuning. Historical holdout used only for follow-up diagnosis.',
        reproduce='PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/validation/chunk_diffik_audit/seam_catchup.py'))
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(job,r) for r in rows]):print(json.dumps(f.result()),flush=True)
    summarize(rows)
