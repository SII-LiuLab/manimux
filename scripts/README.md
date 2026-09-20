# Utility scripts

Private training recipes, launchers and notes live in the root `training/`
directory, which is ignored by Git and absent from a fresh clone. Model training
implementations remain in XPolicyLab or their upstream frameworks.

The scripts are grouped by responsibility:

- `datasets/`: convert datasets, compute statistics, and prepare model assets.
- `validation/`: offline probes, configuration checks, and diagnostic audits.
- `media/`: viewer recording and other presentation helpers.

Policy launchers live in `manimux/servers/`; model implementations and the shared
policy server belong to XPolicyLab.

Run commands from the repository root so relative config and environment paths
resolve consistently. For example:

```bash
envs/yam/.venv/bin/python manimux/servers/pi05.py --check
python scripts/datasets/convert_yam_to_lerobot.py --help
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py --help
```
