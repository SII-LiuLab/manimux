# Utility scripts

Unified training entry: `python scripts/training/train.py --list`.
Use `--recipe <name>` or `--config <job.json>` to plan preparation + training;
add `--execute` only on the training machine to run it. See
[training entrypoints](../docs/training-entrypoints.md) for supported models,
data preparation chains and one-command examples. Existing cluster scripts
remain available for the recorded experiments.

The scripts are grouped by responsibility:

- `servers/`: launch model-side policy services.
- `datasets/`: convert datasets, compute statistics, and prepare model assets.
- `validation/`: offline probes, configuration checks, and diagnostic audits.
- `media/`: viewer recording and other presentation helpers.
- `training/`: cluster launchers for dataset stats, smoke runs, and formal training.

Run commands from the repository root so relative config and environment paths
resolve consistently. For example:

```bash
envs/yam/.venv/bin/python scripts/servers/pi05_yam_server.py --check
python scripts/datasets/convert_yam_to_lerobot.py --help
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py --help
```
