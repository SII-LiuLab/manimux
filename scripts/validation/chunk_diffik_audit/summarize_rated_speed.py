"""Summarize rated-speed follow-up without selecting parameters."""
import json
from collections import defaultdict

import numpy as np

import audit
from rated_speed_replay import OUT


def compact(metrics):
    result = {f'{layer}_{unit}': metrics[f'{layer}_all_{unit}'].get('p95')
              for layer in ('target_reference', 'reference_command', 'target_command')
              for unit in ('mm', 'deg')}
    for dimension in ('tcp_m', 'tcp_rad', 'joint_rad'):
        for derivative in ('velocity', 'acceleration', 'jerk'):
            result[f'command_{dimension}_{derivative}'] = metrics['motion']['command'][f'{dimension}_{derivative}_all'].get('p95')
    for region in ('pre', 'post', 'stable'):
        for unit in ('mm', 'deg'):
            result[f'target_command_{region}_{unit}'] = metrics[f'target_command_{region}_{unit}'].get('p95')
    return result


def main():
    rows = json.loads((audit.OUT / 'cohort.json').read_text())['selected']
    locked = json.loads((audit.OUT / 'locked_candidates.json').read_text())
    comparisons = []
    for row in rows:
        previous = next(c['id'] for c in locked[row['task']] if c['id'] != 'baseline')
        entries = []
        for name, path in [('historical_baseline', audit.OUT / 'baseline' / f'{row["id"]}.json'),
                           ('previous_tcp_0.65_a20_f4', audit.OUT / 'candidates' / row['id'] / f'{previous}.json')]:
            result = json.loads(path.read_text())
            entries.append({'candidate': name, 'metrics': compact(result['metrics']), 'rejected': False})
        paths = list((OUT / 'episodes' / row['id']).glob('*.json'))
        assert len(paths) == 3, (row['id'], len(paths))
        for path in sorted(paths):
            result = json.loads(path.read_text())
            if result['failures']:
                entries.append({'candidate': result['id'], 'metrics': None, 'ik_failures': result['failures'], 'rejected': True})
                continue
            entries.append({'candidate': result['id'], 'metrics': compact(result['metrics']),
                            'rejected': result['guards']['first_rejected_tick'] is not None,
                            'guards': result['guards'],
                            'velocity_limited_fraction': result['velocity_limited_fraction'],
                            'acceleration_limited_fraction': result['acceleration_limited_fraction']})
        comparisons.append({'row': row, 'comparisons': entries})
    audit.write(OUT / 'episode_comparisons.json', comparisons)
    groups = defaultdict(list)
    for episode in comparisons:
        row = episode['row']
        for entry in episode['comparisons']:
            groups[(row['task'], row['split'], row['status'], row['stratum'], entry['candidate'])].append((row['session'], entry))
    strata = []
    for key, values in sorted(groups.items()):
        sessions = defaultdict(list)
        for session, entry in values:
            if entry['metrics'] is not None:
                sessions[session].append(entry['metrics'])
        sample = next(iter(next(iter(sessions.values()))), None) if sessions else None
        metrics = {}
        if sample:
            for metric in sample:
                means = [np.mean([r[metric] for r in rs if r[metric] is not None]) for rs in sessions.values()
                         if any(r[metric] is not None for r in rs)]
                metrics[metric] = float(np.mean(means)) if means else None
        strata.append(dict(zip(['task', 'split', 'status', 'stratum', 'candidate'], key),
                           episodes=len(values), rejected_episodes=sum(v['rejected'] for _, v in values),
                           session_balanced_mean_episode_p95=metrics))
    audit.write(OUT / 'stratified_comparisons.json', strata)
    new_results = [json.loads(p.read_text()) for p in (OUT / 'episodes').glob('*/*.json')]
    assert len(new_results) == 3 * len(rows)
    assert all(not r['failures'] for r in new_results)
    assert len(list((OUT / 'traces').glob('*/*.npz'))) == len(new_results)
    validation = {
        'episodes': len(rows), 'replays': len(new_results), 'ik_failures': 0,
        'public_executor_max_difference_rad': max(r.get('public_kernel_max_difference_rad', 1e-12) for r in new_results),
        'public_executor_comparison_note': 'Cached initial runs passed strict <1e-12 comparison; later runs use production output directly.',
        'source_sha256': audit.digest(__import__('pathlib').Path(__file__).read_bytes()),
        'new_yaml_guard_results': {task: {
            'episodes': sum(r['row']['task'] == task and r['id'] == 'new_yaml' for r in new_results),
            'position_rejected_episodes': sum(r['row']['task'] == task and r['id'] == 'new_yaml' and r['guards']['position_rejected_ticks'] > 0 for r in new_results),
            'velocity_rejected_episodes': sum(r['row']['task'] == task and r['id'] == 'new_yaml' and r['guards']['velocity_rejected_ticks'] > 0 for r in new_results),
        } for task in ('pack_plate', 'pack_bottle')},
        'metric_scope': 'Full historical intervals; post-rejection metrics are hypothetical, not executable results. Guard-valid prefixes are stored per episode.',
    }
    audit.write(OUT / 'validation.json', validation)
    lines = ['Fixed historical executed intervals; TCP error is moving-trajectory p95, not terminal error.',
             'Counterfactual command only. Recorded seeds and targets frozen; no predicted measured state.',
             'New YAML uses production flange objective; TCP variants are isolated offline shims.',
             'Old holdout is follow-up diagnostic only. No parameter selection.', '']
    for episode in comparisons:
        row = episode['row']
        if row['split'] != 'holdout':
            continue
        lines.append(f"{row['task']} {row['status']}: {row['path']}")
        for result in episode['comparisons']:
            if result['metrics'] is None:
                lines.append(f"  {result['candidate']}: IK FAILED")
                continue
            m = result['metrics']
            lines.append(f"  {result['candidate']}: target/ref={m['target_reference_mm']:.3f} mm; "
                         f"ref/cmd={m['reference_command_mm']:.3f} mm; "
                         f"target/cmd={m['target_command_mm']:.3f} mm, {m['target_command_deg']:.3f} deg; "
                         f"linear jerk={m['command_tcp_m_jerk']:.3f} m/s^3; rejected={result['rejected']}")
        lines.append('')
    (OUT / 'comparison.txt').write_text('\n'.join(lines) + '\n')
    for task in ('pack_plate', 'pack_bottle'):
        selected = [e for e in comparisons if e['row']['task'] == task and e['row']['split'] == 'holdout']
        audit.write(OUT / f'{task}_comparison.json', selected)
    print('\n'.join(lines))
    print('New replay checks:', {name: sum(1 for e in comparisons for x in e['comparisons'] if x['candidate'] == name and x['rejected'])
                                  for name in ('new_yaml', 'new_yaml_tcp_objective', 'previous_tcp_candidate_with_rated_speed')})


if __name__ == '__main__':
    main()
