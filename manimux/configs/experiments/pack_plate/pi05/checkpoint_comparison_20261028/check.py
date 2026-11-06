"""Validate catalog, strategies and adapters without connecting hardware."""
from pathlib import Path

from manimux.clock import SystemClock
from manimux.embodiments.robot import build_robot
from manimux.experiment_list import load_experiment_list
from manimux.policy_adapter import build_policy_adapter
from manimux.runtime.inference import build_inference_strategy
from XPolicyLab.policy.Pi_05.pack_plate_contract import validate_artifacts

catalog = load_experiment_list(Path(__file__).with_name('catalog.yaml'))
for choice in catalog.choices.values():
    config = choice.config
    robot = build_robot(config['robot'], SystemClock())
    build_inference_strategy(config)
    adapter = build_policy_adapter(
        config['robot'], config['policy'], kinematics=robot.kinematics,
        motion_limits=config['executor']['motion_limits'],
    )
    adapter.validate(config['robot'], config['policy'])
    if config['policy_server'].get('observation_profile') == 'tianji_taccap_pi05_pack_plate':
        validate_artifacts(config['policy_server'])
print(f'Validated {len(catalog.choices)} choices and {len(catalog.policy_services)} policy services')
