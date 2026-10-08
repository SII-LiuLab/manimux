# Utility scripts

ManiMux runtime branches do not carry training launchers or model-training data
conversion. Those workflows are maintained on the `experiment` branch. Native
model implementations remain in the separate XPolicyLab repository.

The scripts are grouped by responsibility:

- `datasets/`: prepare assets and statistics consumed by runtime inference.
- `validation/`: offline probes, configuration checks, and diagnostic audits.
- `media/`: robogui recording and other presentation helpers.

Policy launchers live in `manimux/servers/`; model implementations and the shared
policy server belong to XPolicyLab.

Run commands from the repository root so relative config and environment paths
resolve consistently. For example:

```bash
envs/yam/.venv/bin/python manimux/servers/pi05.py --check
python scripts/datasets/compute_yam_aac_ee_stats.py --help
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py --help
```
