"""Evaluate declared-rated-speed chunk candidates on the fixed historical cohort.

No hardware, model service, parameter selection, or counterfactual measured state.
"""
import copy
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import audit
import executor_ablation
from manimux.cli import load_config
from manimux.embodiments.robot.base import RobotModel
from manimux.runtime.executors.smooth import SmoothExecutor
from manimux.types import ActionHorizon, RobotState

OUT = audit.OUT / 'rated_speed_v1'
CONFIGS = audit.ROOT / '.local/bottle_plate/umi_dp_scratch_tf32_20261017/chunk_rated_speed_20261019'


def public_replay(d, reference):
    """Use production executor output, keeping its own command/velocity history."""
    output = d['a']['command'].copy()
    executor = SmoothExecutor(d['c']['executor']['smooth'], d['dt'])
    previous = output[0].copy()
    for i in range(len(output)):
        if d['resets'][i]:
            previous = d['a']['command'][i].copy()
        state = RobotState({g: previous[j].copy() for j, g in enumerate(audit.ARMS)}, int(d['t'][i]), i)
        if d['resets'][i]:
            executor.reset(state)
        elif not d['active'][i]:
            command = executor.hold(int(d['t'][i]), state)
            previous = np.stack([command.groups[g] for g in audit.ARMS])
        else:
            horizon = ActionHorizon(int(d['t'][i]), round(d['dt'] * 1e9), d['ids'][i],
                                    {g: reference[i, j][None] for j, g in enumerate(audit.ARMS)})
            command = executor.step(int(d['t'][i]), state, horizon)
            previous = np.stack([command.groups[g] for g in audit.ARMS])
        output[i, :, :7] = previous[:, :7]
    return output


def job(row):
    d = audit.load(row)
    kin = RobotModel.from_config(d['c']['robot']['config']).kinematics
    config_path = CONFIGS / f'{row["task"]}_latest_async_chunk_rated_speed.yaml'
    config = load_config(config_path)
    baseline_path = audit.OUT / 'baseline' / f'{row["id"]}.json'
    baseline = json.loads(baseline_path.read_text())
    assert baseline['validation']['command_max_abs_rad'] < 1e-8
    target = np.load(baseline_path.with_suffix('.npz'))['target']
    old_command, _ = audit.replay(d, d['a']['scheduled'], {}, public=True)
    assert np.max(abs(old_command[:, :, :7] - d['a']['command'][:, :, :7])) < 1e-12
    rated = config['executor']['motion_limits']['arm']['max_velocity']
    # Recreate IK at the new shared rate; never reuse old reference for this change.
    regenerated = {}
    results = []
    variants = [
        ('new_yaml', 'flange', None),
        ('new_yaml_tcp_objective', 'tcp', None),
        ('previous_tcp_candidate_with_rated_speed', 'tcp', {'cutoff_hz': 4., 'max_acceleration': 20.}),
    ]
    for name, objective, smooth_override in variants:
        path = OUT / 'episodes' / row['id'] / f'{name}.json'
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        settings = copy.deepcopy(d)
        # Preserve historical target, seed, scheduling, gripper semantics and dt.
        # Apply the new YAML's arm executor configuration and guard, with its
        # production objective unless this is explicitly named TCP diagnostic.
        for field in ('max_velocity', 'max_acceleration', 'max_step_dt_s', 'mode'):
            settings['c']['executor']['motion_limits']['arm'][field] = config['executor']['motion_limits']['arm'][field]
            settings['c']['executor']['smooth'][field] = config['executor']['smooth'][field]
        settings['c']['executor']['smooth']['cutoff_hz'] = config['executor']['smooth']['cutoff_hz']
        settings['c']['executor']['command_safety']['max_velocity'] = copy.deepcopy(config['executor']['command_safety']['max_velocity'])
        if smooth_override:
            settings['c']['executor']['smooth'].update(smooth_override)
            settings['c']['executor']['motion_limits']['arm']['max_acceleration'] = smooth_override['max_acceleration']
        choice = {'velocity': rated}
        if objective == 'tcp':
            choice['objective'] = 'tcp'
        if objective not in regenerated:
            regenerated[objective] = audit.regenerate(settings, kin, choice)
        reference, new_target, plans, diag, failures = regenerated[objective]
        result = dict(id=name, row=row, config_path=str(config_path),
                      config_sha256=audit.digest(config_path.read_bytes()),
                      effective_arm_smooth=settings['c']['executor']['smooth'],
                      objective=objective, failures=failures,
                      recorded_state_is_not_prediction=True)
        if failures:
            audit.write(path, result)
            results.append(result)
            continue
        assert np.array_equal(target, new_target, equal_nan=True)
        kernel_command, flags = audit.replay(settings, reference, {})
        command = public_replay(settings, reference)
        public_kernel_difference = float(np.max(abs(command[:, :, :7] - kernel_command[:, :, :7])))
        checks, first = executor_ablation.guards(settings, command)
        score = audit.score(settings, kin, reference, command, target)
        prefix = None
        if first is not None:
            prefix_d = dict(settings)
            prefix_d['masks'] = {k: mask & (np.arange(len(command)) < first) for k, mask in settings['masks'].items()}
            prefix = audit.score(prefix_d, kin, reference, command, target)
        result.update(metrics=score, guard_valid_prefix_metrics=prefix, guards=checks,
                      public_executor_is_authoritative=True,
                      public_kernel_max_difference_rad=public_kernel_difference,
                      velocity_limited_fraction=float(flags[d['active'], :, 0].mean()),
                      acceleration_limited_fraction=float(flags[d['active'], :, 1].mean()),
                      plans=plans,
                      ik_velocity_active_fraction=diag[:, :, :7].mean(axis=(0, 1)).tolist(),
                      ik_position_active_fraction=diag[:, :, 7:14].mean(axis=(0, 1)).tolist())
        audit.write(path, result)
        trace = OUT / 'traces' / row['id'] / f'{name}.npz'
        trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace, monotonic_ns=d['t'], plan_id=d['ids'], active=d['active'],
                            reset=d['resets'], original_target_tcp=target,
                            counterfactual_reference=reference, counterfactual_command=command,
                            recorded_command=d['a']['command'], recorded_state=d['a']['state'],
                            pre_handoff=d['masks']['pre'], post_handoff=d['masks']['post'],
                            stable=d['masks']['stable'])
        results.append(result)
    return {'episode': row['id'], 'results': [{
        'candidate': r['id'], 'failures': len(r['failures']),
        'target_command_p95_mm': r.get('metrics', {}).get('target_command_all_mm', {}).get('p95'),
        'first_rejected_tick': r.get('guards', {}).get('first_rejected_tick'),
    } for r in results]}


def main():
    rows = json.loads((audit.OUT / 'cohort.json').read_text())['selected']
    OUT.mkdir(parents=True, exist_ok=True)
    audit.write(OUT / 'provenance.json', {
        'cohort_sha256': audit.digest((audit.OUT / 'cohort.json').read_bytes()),
        'source_sha256': {str(p.relative_to(audit.ROOT)): audit.digest(p.read_bytes()) for p in
                          [Path(__file__), Path(audit.__file__), Path(executor_ablation.__file__),
                           *[audit.ROOT / s for s in audit.SOURCE]]},
        'protocol': 'Fixed cohort; all original sessions and outcomes separated. Prior holdout is follow-up diagnostic only; no selection. '
                    'Re-solve original targets from recorded seeds at the new rate. Preserve executed intervals, handoff, latency trim, command history and holds. '
                    'New YAML arm settings applied to each historical episode; historical timing/policy retained. '
                    'Public SmoothExecutor checked every tick; arm SafetyGuard checked. Gripper commands remain recorded. '
                    'No hardware response or command-to-state prediction. Full interval metrics after a rejection are hypothetical only.',
        'reproduce': 'PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/validation/chunk_diffik_audit/rated_speed_replay.py',
    })
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(job, row) for row in rows]):
            print(json.dumps(f.result()), flush=True)


if __name__ == '__main__':
    main()
