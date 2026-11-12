"""Build isolated chunk configurations at the declared rated arm speed.

Only reads/validates configuration. Does not construct hardware or start services.
This removes reduced speed budgets, not the finite velocity bounds required by IK.
"""
import copy
import hashlib
import json
from pathlib import Path

import yaml

from manimux.cli import load_config


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / '.local/bottle_plate/umi_dp_scratch_tf32_20261017'
DEST = SOURCE / 'chunk_rated_speed_20261019'


def absolute_references(value, base):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'config', 'local', 'control_profile'} and isinstance(item, str):
                value[key] = str((base / item).resolve())
            else:
                absolute_references(item, base)
    elif isinstance(value, list):
        for item in value:
            absolute_references(item, base)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    results = []
    for task in ('pack_plate', 'pack_bottle'):
        source = SOURCE / f'{task}_latest_async_chunk.yaml'
        before = load_config(source)
        raw = yaml.safe_load(source.read_text())
        absolute_references(raw, source.parent)
        assembly = yaml.safe_load(Path(raw['robot']['config']).read_text())
        rated = float(assembly['hardware']['rated_joint_velocity_rad_s'])
        raw['robot']['options'].setdefault('hardware', {})['velocity_ratio'] = 100
        executor = raw['executor']
        executor['motion_limits']['arm']['max_velocity'] = rated
        if 'max_velocity' in executor.get('smooth', {}):
            executor['smooth']['max_velocity'] = rated
        for group, rates in executor['command_safety']['max_velocity'].items():
            gripper_index = before['executor']['motion_limits']['gripper']['group_indices'][group]
            executor['command_safety']['max_velocity'][group] = [
                rate if index == gripper_index else rated
                for index, rate in enumerate(rates)
            ]
        raw['run']['output_dir'] = str(
            ROOT / 'data/experiments/chunk_rated_speed_20261019' / task
        )
        target = DEST / f'{task}_latest_async_chunk_rated_speed.yaml'
        target.write_text(
            '# Isolated speed-only experiment. Finite rated-speed bounds remain.\n'
            '# Hardware is not started by the generator. Original filter/acceleration retained.\n'
            + yaml.safe_dump(raw, sort_keys=False)
        )
        after = load_config(target)
        assert after['robot']['options']['hardware']['velocity_ratio'] == 100
        assert after['executor']['motion_limits']['arm']['max_velocity'] == rated
        assert after['executor']['smooth']['max_velocity'] == rated
        # All other resolved behavior is identical, including joint bounds and handoff.
        expected = copy.deepcopy(before)
        expected['robot']['options']['hardware']['velocity_ratio'] = 100
        expected['executor']['motion_limits']['arm']['max_velocity'] = rated
        expected['executor']['smooth']['max_velocity'] = rated
        expected['executor']['command_safety']['max_velocity'] = copy.deepcopy(
            after['executor']['command_safety']['max_velocity']
        )
        expected['run']['output_dir'] = after['run']['output_dir']
        assert expected == after, 'Unexpected resolved configuration change'
        results.append({
            'source': str(source),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'candidate': str(target),
            'candidate_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'velocity_ratio': 100,
            'shared_max_velocity_rad_s': rated,
            'guard_arm_max_velocity_rad_s': rated,
            'max_acceleration': after['executor']['motion_limits']['arm']['max_acceleration'],
            'cutoff_hz': after['executor']['smooth']['cutoff_hz'],
            'handoff_skip_steps': after['inference']['handoff_skip_steps'],
            'validation': 'Real load_config passed; only speed settings and output directory changed.',
            'limitations': 'Not unlimited; no new IK replay, hardware motion or model service. '
                           'Original production IK objective retained; offline TCP shim is not installed.',
        })
    (DEST / 'validation.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
