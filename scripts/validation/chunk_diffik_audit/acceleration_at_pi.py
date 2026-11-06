"""Frozen-reference acceleration sweep at pi rad/s; no hardware or model calls."""
import copy
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import audit
import executor_ablation
import rated_speed_replay
from manimux.embodiments.robot.base import RobotModel

OUT = audit.OUT / 'acceleration_at_pi_v1'
ACCELERATIONS = (20., 40., 80., 160., None)
CONTEXTS = {
    'tcp_f4': ('previous_tcp_candidate_with_rated_speed', 4.),
    'production_flange_f8': ('new_yaml', 8.),
}


def job(row):
    d = audit.load(row)
    kin = RobotModel.from_config(d['c']['robot']['config']).kinematics
    results = []
    for context, (source_name, cutoff) in CONTEXTS.items():
        trace = rated_speed_replay.OUT / 'traces' / row['id'] / f'{source_name}.npz'
        data = np.load(trace)
        reference, target = data['counterfactual_reference'], data['original_target_tcp']
        source = json.loads((rated_speed_replay.OUT / 'episodes' / row['id'] / f'{source_name}.json').read_text())
        assert source['effective_arm_smooth']['max_velocity'] == math.pi
        current = dict(d)
        current['c'] = copy.deepcopy(d['c'])
        current['c']['executor']['smooth'] = copy.deepcopy(source['effective_arm_smooth'])
        current['c']['executor']['motion_limits']['arm']['max_velocity'] = math.pi
        for group in audit.ARMS:
            current['c']['executor']['command_safety']['max_velocity'][group][:7] = [math.pi] * 7
        reference_hash = audit.digest(reference.tobytes())
        for acceleration in ACCELERATIONS:
            label = 'none' if acceleration is None else str(int(acceleration))
            path = OUT / 'episodes' / row['id'] / f'{context}_a{label}.json'
            if path.exists():
                results.append(json.loads(path.read_text()))
                continue
            current['c']['executor']['smooth']['max_acceleration'] = acceleration
            current['c']['executor']['motion_limits']['arm']['max_acceleration'] = acceleration
            command = rated_speed_replay.public_replay(current, reference)
            # Production executor is authoritative; compare the diagnostic kernel.
            kernel, flags = audit.replay(current, reference, {})
            difference = float(abs(command[:, :, :7] - kernel[:, :, :7]).max())
            assert difference < 1e-8, difference
            metrics = audit.score(current, kin, reference, command, target)
            guards, first = executor_ablation.guards(current, command)
            prefix = None
            if first is not None:
                pre = dict(current)
                pre['masks'] = {k: mask & (np.arange(len(command)) < first) for k, mask in current['masks'].items()}
                prefix = audit.score(pre, kin, reference, command, target)
            baseline_match = None
            if acceleration == source['effective_arm_smooth']['max_acceleration']:
                baseline_match = float(abs(command[:, :, :7] - data['counterfactual_command'][:, :, :7]).max())
                assert baseline_match < 1e-8
            result = dict(row=row, context=context, max_velocity_rad_s=math.pi,
                          cutoff_hz=cutoff, max_acceleration_rad_s2=acceleration,
                          reference_sha256=reference_hash, source_trace=str(trace),
                          metrics=metrics, guards=guards, guard_valid_prefix_metrics=prefix,
                          public_kernel_difference_rad=difference, prior_replay_match_rad=baseline_match,
                          speed_limited_fraction=float(flags[d['active'], :, 0].mean()),
                          acceleration_limited_fraction=float(flags[d['active'], :, 1].mean()))
            audit.write(path, result)
            # Retain each variant's complete command history for later comparison.
            saved = OUT / 'traces' / row['id'] / f'{context}_a{label}.npz'
            saved.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(saved, monotonic_ns=d['t'], plan_id=d['ids'], active=d['active'],
                                reset=d['resets'], command=command, reference=reference, target_tcp=target)
            results.append(result)
    return dict(episode=row['id'], replays=len(results))


def summarize(rows):
    results = [json.loads(p.read_text()) for p in (OUT / 'episodes').glob('*/*.json')]
    assert len(results) == len(rows) * len(CONTEXTS) * len(ACCELERATIONS)
    grouped = defaultdict(list)
    for r in results:
        row = r['row']
        key = (row['task'], row['split'], row['status'], row['stratum'], r['context'], str(r['max_acceleration_rad_s2']))
        grouped[key].append(r)
    aggregates = []
    for key, group in sorted(grouped.items()):
        sessions = defaultdict(list)
        for r in group:
            sessions[r['row']['session']].append(r)
        metrics = {}
        for layer in ('target_reference', 'reference_command', 'target_command'):
            for region in ('all', 'pre', 'post', 'stable'):
                for unit in ('mm', 'deg'):
                    name = f'{layer}_{region}_{unit}'
                    means = [np.mean([r['metrics'][name]['p95'] for r in rs if 'p95' in r['metrics'][name]])
                             for rs in sessions.values() if any('p95' in r['metrics'][name] for r in rs)]
                    metrics[name] = float(np.mean(means)) if means else None
        for dimension in ('joint_rad', 'tcp_m', 'tcp_rad'):
            for derivative in ('velocity', 'acceleration', 'jerk'):
                name = f'{dimension}_{derivative}_all'
                metrics[name] = float(np.mean([np.mean([r['metrics']['motion']['command'][name]['p95'] for r in rs]) for rs in sessions.values()]))
        aggregates.append(dict(zip(('task', 'split', 'status', 'stratum', 'context', 'acceleration'), key),
                               episodes=len(group), rejected_episodes=sum(r['guards']['first_rejected_tick'] is not None for r in group),
                               session_balanced_episode_p95=metrics))
    audit.write(OUT / 'stratified_results.json', aggregates)
    audit.write(OUT / 'validation.json', dict(episodes=len(rows), replays=len(results),
        public_kernel_max_difference_rad=max(r['public_kernel_difference_rad'] for r in results),
        previous_replay_max_difference_rad=max(r['prior_replay_match_rad'] or 0. for r in results),
        no_parameter_selection=True, command_state_prediction=False))
    lines = ['Acceleration sweep at pi rad/s; frozen reference, persistent production executor.',
             'All errors are TCP p95 on historical executed intervals, not terminal or measured-state errors.',
             'A rejected full interval is hypothetical only; guard-valid-prefix metrics are stored separately.', '']
    for row in rows:
        if row['split'] != 'holdout':
            continue
        lines.append(f"{row['task']} {row['status']} {row['path']}")
        for context in CONTEXTS:
            for acceleration in ACCELERATIONS:
                r = next(r for r in results if r['row']['id'] == row['id'] and r['context'] == context and r['max_acceleration_rad_s2'] == acceleration)
                m = r['metrics']
                lines.append(f"  {context} a={acceleration}: ref/cmd={m['reference_command_all_mm']['p95']:.3f} mm; "
                             f"target/cmd={m['target_command_all_mm']['p95']:.3f} mm, {m['target_command_all_deg']['p95']:.3f} deg; "
                             f"linear jerk={m['motion']['command']['tcp_m_jerk_all']['p95']:.3f}; "
                             f"speed-limited={r['speed_limited_fraction']:.3f}; accel-limited={r['acceleration_limited_fraction']:.3f}; "
                             f"rejected={r['guards']['first_rejected_tick'] is not None}")
        lines.append('')
    (OUT / 'comparison.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    rows = json.loads((audit.OUT / 'cohort.json').read_text())['selected']
    OUT.mkdir(parents=True, exist_ok=True)
    audit.write(OUT / 'protocol.json', dict(accelerations=ACCELERATIONS, contexts=CONTEXTS,
        cohort_sha256=audit.digest((audit.OUT / 'cohort.json').read_bytes()),
        code_sha256={str(p.relative_to(audit.ROOT)): audit.digest(p.read_bytes()) for p in
                     [Path(__file__), Path(audit.__file__), Path(rated_speed_replay.__file__), Path(executor_ablation.__file__),
                      audit.ROOT / 'manimux/runtime/executors/smooth.py', audit.ROOT / 'manimux/runtime/executors/limits.py']},
        notes='No new selection. Historical holdout is follow-up diagnostic only. Original cohort, seeds, timing, '
              'latency trim and handoff retained. Fixed pi-rate references from the prior regeneration. '
              'TCP context uses the offline TCP objective; production context uses flange objective. '
              'Gripper commands retained from recording; no plant, SDK, contact or new feedback simulation.',
        reproduce='PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/validation/chunk_diffik_audit/acceleration_at_pi.py'))
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(job, row) for row in rows]):
            print(json.dumps(f.result()), flush=True)
    summarize(rows)


if __name__ == '__main__':
    main()
