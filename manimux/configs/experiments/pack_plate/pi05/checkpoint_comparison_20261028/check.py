"""Validate catalog and complete runtime construction without connecting hardware."""
import argparse
from copy import deepcopy
from pathlib import Path

from manimux.experiment_list import load_experiment_list
from manimux.runtime import build_runtime
from XPolicyLab.policy.Pi_05.pack_plate_contract import validate_artifacts

catalog = load_experiment_list(Path(__file__).with_name('catalog.yaml'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--policy', choices=sorted(catalog.policy_services),
                    help='Check this policy only; the complete catalog is still loader-validated')
args = parser.parse_args()
choices = [choice for choice in catalog.choices.values()
           if args.policy is None or choice.policy == args.policy]
for choice in choices:
    config = deepcopy(choice.config)
    config['robogui']['enabled'] = False
    # Constructor only: do not call run(), start workers, or connect hardware.
    runtime = build_runtime(config, Path(config['run']['output_dir']))
    assert not runtime._worker._started
    if runtime._decoder is not None:
        assert not runtime._decoder._started
        runtime._decoder.close()
    runtime._worker.close()
    runtime._robogui.close()
    if config['policy_server'].get('observation_profile') == 'tianji_taccap_pi05_pack_plate':
        validate_artifacts(config['policy_server'])
print(
    f'Loaded {len(catalog.choices)} catalog choices and '
    f'{len(catalog.policy_services)} policy services; '
    f'validated {len(choices)} complete runtime constructors and their Pi05 artifacts'
)
