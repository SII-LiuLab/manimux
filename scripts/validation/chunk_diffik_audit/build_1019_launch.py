"""Generate the isolated 1019 bottle launch from pinned checkpoint metadata."""
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / '.local/bottle_plate/umi_dp_scratch_tf32_20261017'
DEST = ROOT / '.local/bottle_plate/umi_dp_tracker_fixed_tf32_20261019'


def write_yaml(name, value, comment=''):
    (DEST / name).write_text(comment + yaml.safe_dump(value, sort_keys=False))


def main():
    identity = json.loads((DEST / 'artifact_identity.json').read_text())
    base = yaml.safe_load((SOURCE / 'pack_bottle_latest_async_chunk.yaml').read_text())
    safety = copy.deepcopy(base['executor']['command_safety'])
    safety.pop('max_velocity')
    profile = dict(robot=dict(type='tianji_taccap', group_dims=base['robot']['group_dims'], options={}),
                   rate_contract=dict(command_margin=1.0, mode='isotropic', max_step_dt_s=.016,
                                      max_acceleration=80., gripper=dict(max_velocity=10., max_acceleration=12., max_closing_velocity=10.)),
                   command_safety=safety)
    write_yaml('control_1019.yaml', profile,
               '# Shared 1019 rates: rated pi rad/s * hardware 100% * margin 1.0.\n'
               '# DiffIK and executor share pi rad/s; executor acceleration is 80 rad/s^2.\n'
               '# Position bounds and gripper limits are retained from the 1017 study.\n')
    station = yaml.safe_load((SOURCE / 'station.yaml').read_text())
    station['services']['policy']['endpoint'] = 'ws://127.0.0.1:8570'
    write_yaml('station.yaml', station, '# Isolated station binding; same devices, dedicated model port 8570.\n')
    server = yaml.safe_load((SOURCE / 'pack_bottle_latest_server.yaml').read_text())
    server.update(checkpoint_path=identity['checkpoint_path'], checkpoint_file=identity['checkpoint_file'],
                  expected_artifacts=identity, port=8570)
    write_yaml('pack_bottle_latest_server.yaml', server,
               '# One pinned 1019 checkpoint; preprocessing and normalizer come from its embedded configuration.\n')
    services, experiments = {}, []
    for mode in ('chunk', 'step'):
        group = f'1019 bottle latest / {mode} DiffIK'
        services[group] = dict(cwd='../../..', command=['envs/umi_dp/.venv/bin/python', '-m', 'manimux.servers.umi_dp',
            '--experiment', '{config}', '--local', '{local}'], env={'PYTHONPATH': None},
            startup_timeout_s=300, stop_timeout_s=30)
        for schedule in ('async', 'serial'):
            name = f'pack_bottle_latest_{schedule}_{mode}'
            config = yaml.safe_load((SOURCE / f'{name}.yaml').read_text())
            config['control_profile'] = 'control_1019.yaml'
            config['local'] = 'station.yaml'
            config['run'].update(output_dir=f'./data/experiments/umi-dp-bottle-tracker-fixed-tf32-20261019/{name}',
                                 control_timing=True, timing_max_cycles=20000)
            config['recording']['enabled'] = True
            config['robot']['options']['hardware'].update(velocity_ratio=100, acceleration_ratio=100,
                                                          rated_joint_velocity_rad_s=3.141592653589793)
            config['policy']['expected_backend']['model'] = identity
            config['policy']['options']['server'] = 'ws://127.0.0.1:8570'
            config['policy']['action_dt_s'] = identity['action_dt_s']
            config['policy']['horizon_policy_steps'] = identity['action_horizon']
            config['policy_server'] = {'config': 'pack_bottle_latest_server.yaml'}
            config['executor'].pop('motion_limits', None)
            config['executor'].pop('command_safety', None)
            if mode == 'chunk':
                config['executor']['smooth']['cutoff_hz'] = 4.
            else:
                # Do not retain an unrelated older Cartesian speed budget on top
                # of the requested shared pi/80 joint limits.
                config['execution']['ik']['pose_speed_limit'] = None
            config['robogui']['policy_label'] = f'1019 bottle latest / {schedule} / {mode} DiffIK / pi-80'
            write_yaml(f'{name}.yaml', config,
                       '# Isolated 1019 live experiment. Loads without connecting; Prepare/Start use hardware.\n')
            experiments.append(dict(id=name, policy=group,
                label=f'{schedule.title()} / {mode} DiffIK / {config["executor"]["type"]} / pi-80', config=f'{name}.yaml'))
    write_yaml('experiments.yaml', dict(policy_services=services, experiments=experiments),
               '# Four methods, one checkpoint. Two adapter groups; only the selected service runs.\n')
    print(DEST)


if __name__ == '__main__':
    main()
