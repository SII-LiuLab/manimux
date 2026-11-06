"""Compare direct/smooth and isotropic/per-joint at pi rad/s and 80 rad/s^2.

Frozen pi-rate references, original cohort and persistent command histories.
"""
import copy
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import audit
import acceleration_at_pi
import executor_ablation
import rated_speed_replay
from manimux.embodiments.robot.base import RobotModel

OUT = audit.OUT / 'executor_at_pi_a80_v1'
VARIANTS = [(executor, mode) for executor in ('smooth', 'direct') for mode in ('isotropic', 'per_joint')]


def job(row):
    d = audit.load(row)
    kin = RobotModel.from_config(d['c']['robot']['config']).kinematics
    results = []
    for context, (source_name, cutoff) in acceleration_at_pi.CONTEXTS.items():
        trace = rated_speed_replay.OUT / 'traces' / row['id'] / f'{source_name}.npz'
        data = np.load(trace)
        reference, target = data['counterfactual_reference'], data['original_target_tcp']
        source = json.loads((rated_speed_replay.OUT / 'episodes' / row['id'] / f'{source_name}.json').read_text())
        settings = copy.deepcopy(source['effective_arm_smooth'])
        settings.update(max_velocity=math.pi, max_acceleration=80.)
        commands, reports, stops = {}, {}, []
        current = dict(d)
        current['c'] = copy.deepcopy(d['c'])
        for group in audit.ARMS:
            current['c']['executor']['command_safety']['max_velocity'][group][:7] = [math.pi] * 7
        for executor, mode in VARIANTS:
            label = f'{executor}_{mode}'
            settings['mode'] = mode
            current['c']['executor']['smooth'] = copy.deepcopy(settings)
            for k in ('max_velocity', 'max_acceleration', 'max_step_dt_s', 'mode'):
                current['c']['executor']['motion_limits']['arm'][k] = settings[k]
            if executor == 'smooth':
                command = rated_speed_replay.public_replay(current, reference)
                diagnostic_settings = settings
                bits = 7
            else:
                command = executor_ablation.direct_public(current, reference, settings, limited=True)
                command[:, :, 7] = d['a']['command'][:, :, 7]
                diagnostic_settings = dict(settings, position_limit_abs=None)
                bits = 6
            kernel, flags = executor_ablation.replay(current, reference, diagnostic_settings, bits)
            difference = float(np.max(abs(kernel[:, :, :7] - command[:, :, :7])))
            assert difference < 1e-8, (label, difference)
            baseline_difference = None
            if label == 'smooth_isotropic':
                old = np.load(acceleration_at_pi.OUT / 'traces' / row['id'] / f'{context}_a80.npz')['command']
                baseline_difference = float(np.max(abs(old[:, :, :7] - command[:, :, :7])))
                assert baseline_difference < 1e-8
            checks, first = executor_ablation.guards(current, command)
            stops.append(first if first is not None else len(command))
            metrics = audit.score(current, kin, reference, command, target)
            prefix = None
            if first is not None:
                prefix_d = dict(current)
                prefix_d['masks'] = {k: mask & (np.arange(len(command)) < first) for k, mask in current['masks'].items()}
                prefix = audit.score(prefix_d, kin, reference, command, target)
            reports[label] = dict(row=row, context=context, variant=label,
                parameters=dict(executor=executor, mode=mode, max_velocity_rad_s=math.pi,
                                max_acceleration_rad_s2=80., cutoff_hz=cutoff if executor == 'smooth' else None),
                reference_sha256=audit.digest(reference.tobytes()), source_trace=str(trace),
                metrics=metrics, guards=checks, guard_valid_prefix_metrics=prefix,
                public_kernel_difference_rad=difference, baseline_difference_rad=baseline_difference,
                speed_limited_fraction=float(flags[d['active'], :, 0].mean()),
                acceleration_limited_fraction=float(flags[d['active'], :, 1].mean()))
            commands[label] = command
        stop = min(stops)
        common = dict(current)
        common['masks'] = {k: mask & (np.arange(len(reference)) < stop) for k, mask in current['masks'].items()}
        for label, report in reports.items():
            report['common_guard_valid_prefix_end_tick'] = stop
            report['common_guard_valid_prefix_metrics'] = audit.score(common, kin, reference, commands[label], target)
            audit.write(OUT / 'episodes' / row['id'] / f'{context}__{label}.json', report)
            path = OUT / 'traces' / row['id'] / f'{context}__{label}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, monotonic_ns=d['t'], plan_id=d['ids'], active=d['active'], reset=d['resets'],
                                reference=reference, command=commands[label], target_tcp=target)
            results.append(report)
    return dict(episode=row['id'], replays=len(results))


def metric_summary(metrics):
    result = {f'{layer}_{region}_{unit}': metrics[f'{layer}_{region}_{unit}'].get('p95')
              for layer in ('target_reference', 'reference_command', 'target_command')
              for region in ('all', 'pre', 'post', 'stable') for unit in ('mm', 'deg')}
    for dimension in ('joint_rad', 'tcp_m', 'tcp_rad'):
        for derivative in ('velocity', 'acceleration', 'jerk'):
            for region in ('all', 'pre', 'post', 'stable'):
                name = f'{dimension}_{derivative}_{region}'
                result[name] = metrics['motion']['command'][name].get('p95')
    return result


def summarize(rows):
    results = [json.loads(p.read_text()) for p in (OUT / 'episodes').glob('*/*.json')]
    assert len(results) == len(rows) * len(VARIANTS) * len(acceleration_at_pi.CONTEXTS)
    groups = defaultdict(list)
    for r in results:
        row = r['row']
        groups[(row['task'], row['split'], row['status'], row['stratum'], r['context'], r['variant'])].append(r)
    stratified = []
    for key, group in sorted(groups.items()):
        record = dict(zip(('task', 'split', 'status', 'stratum', 'context', 'variant'), key),
                      episodes=len(group), rejected_episodes=sum(r['guards']['first_rejected_tick'] is not None for r in group))
        for source in ('metrics', 'common_guard_valid_prefix_metrics'):
            sessions = defaultdict(list)
            for r in group:
                sessions[r['row']['session']].append(metric_summary(r[source]))
            summary = {}
            for metric in metric_summary(group[0][source]):
                means = [np.mean([v[metric] for v in values if v[metric] is not None]) for values in sessions.values()
                         if any(v[metric] is not None for v in values)]
                summary[metric] = float(np.mean(means)) if means else None
            record[f'session_balanced_episode_p95_{source}'] = summary
        stratified.append(record)
    audit.write(OUT / 'stratified_results.json', stratified)
    # Development-only review: no follow-up holdout values enter this decision.
    for task in ('pack_plate', 'pack_bottle'):
        review = []
        for executor, mode in VARIANTS:
            label = f'{executor}_{mode}'
            group = [r for r in results if r['row']['task'] == task and r['row']['split'] == 'development'
                     and r['context'] == 'tcp_f4' and r['variant'] == label]
            sessions = defaultdict(list)
            for r in group:
                sessions[r['row']['session']].append(r)
            aggregate = {}
            for name in ('target_command_all_mm', 'target_command_all_deg', 'tcp_m_jerk_all', 'tcp_rad_jerk_all'):
                aggregate[name] = float(np.mean([np.mean([metric_summary(r['metrics'])[name] for r in values]) for values in sessions.values()]))
            review.append(dict(variant=label, episodes=len(group), sessions=len(sessions),
                               rejected_episodes=sum(r['guards']['first_rejected_tick'] is not None for r in group),
                               session_balanced_episode_p95=aggregate))
        baseline = next(r for r in review if r['variant'] == 'smooth_isotropic')
        for r in review:
            r['ratios_to_smooth_isotropic'] = {k: v / baseline['session_balanced_episode_p95'][k]
                                              for k, v in r['session_balanced_episode_p95'].items()}
            ratios = r['ratios_to_smooth_isotropic']
            r['passes_prior_precision_and_smoothness_rule'] = (r['rejected_episodes'] == 0
                and ratios['target_command_all_mm'] <= 1 and ratios['target_command_all_deg'] <= 1
                and ratios['tcp_m_jerk_all'] <= 1.05 and ratios['tcp_rad_jerk_all'] <= 1.05)
        improvements = [r['variant'] for r in review if r['variant'] != 'smooth_isotropic'
                        and r['passes_prior_precision_and_smoothness_rule']]
        audit.write(OUT / f'{task}_candidate_review.json', dict(task=task, review_only=True,
            apply_to_production=False, max_velocity_rad_s=math.pi, max_acceleration_rad_s2=80.,
            cutoff_hz=4., ik_objective='offline TCP shim; not installed in production',
            recommended_variants=improvements or ['smooth_isotropic'],
            decision_scope='Development sessions only. Preserve prior precision + jerk <=105% rule; no holdout selection.',
            results=review, limitations='Arm-only open-loop replay; original targets/seeds and schedule frozen. '
                'Recorded gripper commands retained. No measured-state prediction or new hardware validation. '
                'Direct is limited by pi/80 and is not raw passthrough. Physical joint bounds unchanged.'))
    audit.write(OUT / 'validation.json', dict(episodes=len(rows), replays=len(results),
        public_kernel_max_difference_rad=max(r['public_kernel_difference_rad'] for r in results),
        prior_baseline_max_difference_rad=max(r['baseline_difference_rad'] or 0. for r in results),
        guard_rejections={f'{task}/{context}/{executor}_{mode}': sum(r['row']['task'] == task and r['context'] == context and r['variant'] == f'{executor}_{mode}' and r['guards']['first_rejected_tick'] is not None for r in results)
                          for task in ('pack_plate', 'pack_bottle') for context in acceleration_at_pi.CONTEXTS for executor, mode in VARIANTS}))
    for task in ('pack_plate', 'pack_bottle'):
        lines = [f'{task}: shared vmax=pi rad/s, executor amax=80 rad/s^2.',
                 'Fixed reference within each context; no new IK or counterfactual plant.',
                 'Full-interval metrics after a first guard rejection are hypothetical.', '']
        for row in rows:
            if row['task'] != task or row['split'] != 'holdout':
                continue
            lines.append(f"{row['status']} {row['path']}")
            for context in acceleration_at_pi.CONTEXTS:
                for executor, mode in VARIANTS:
                    r = next(r for r in results if r['row']['id'] == row['id'] and r['context'] == context and r['variant'] == f'{executor}_{mode}')
                    m = r['metrics']
                    lines.append(f"  {context} {r['variant']}: target/cmd {m['target_command_all_mm']['p95']:.3f} mm / "
                                 f"{m['target_command_all_deg']['p95']:.3f} deg; ref/cmd {m['reference_command_all_mm']['p95']:.3f} mm; "
                                 f"linear jerk {m['motion']['command']['tcp_m_jerk_all']['p95']:.3f}; "
                                 f"post-handoff {m['target_command_post_mm']['p95']:.3f} mm; "
                                 f"stable {m['target_command_stable_mm']['p95']:.3f} mm; rejected={r['guards']['first_rejected_tick'] is not None}")
        (OUT / f'{task}_comparison.txt').write_text('\n'.join(lines) + '\n')
        print('\n'.join(lines))


def main():
    rows = json.loads((audit.OUT / 'cohort.json').read_text())['selected']
    OUT.mkdir(parents=True, exist_ok=True)
    audit.write(OUT / 'protocol.json', dict(variants=VARIANTS, max_velocity_rad_s=math.pi,
        max_acceleration_rad_s2=80., contexts=acceleration_at_pi.CONTEXTS,
        cohort_sha256=audit.digest((audit.OUT / 'cohort.json').read_bytes()),
        code_sha256={str(p.relative_to(audit.ROOT)): audit.digest(p.read_bytes()) for p in
                     [Path(__file__), Path(audit.__file__), Path(rated_speed_replay.__file__), Path(executor_ablation.__file__),
                      *[audit.ROOT / f'manimux/runtime/executors/{name}.py' for name in ('smooth', 'direct', 'limits')]]},
        notes='Original cohort and strata preserved. Prior holdout is follow-up diagnostic only. '
              'Any recommendation must use development sessions, not holdout. Direct retains velocity/acceleration limits; it is not raw passthrough. '
              'Production executor replay; arm comparison retains recorded gripper commands, original timing, '
              'latency trim, handoff, holds and resets. No SDK, model service or hardware. No physical position-bound changes.',
        reproduce='PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/validation/chunk_diffik_audit/executor_at_pi_a80.py'))
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(job, row) for row in rows]):
            print(json.dumps(f.result()), flush=True)
    summarize(rows)


if __name__ == '__main__':
    main()
