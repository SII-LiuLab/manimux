# Experiment entry points

Current experiments use `executor.type: direct`, independently of the inference
algorithm. Keep that choice explicit in new experiment YAMLs. Direct removes the
Smooth executor; timeline blending, adapter/IK behavior, explicitly configured
motion limits and robot SDK processing remain separate settings.

## Active tasks

| Task directory | Current entries |
| --- | --- |
| `put_bottles/` | Existing bottle experiments, all using Direct |
| `coin/pi05/` | 50k: Serial, RTC, Async, PAINT |
| `stack_blocks/pi05/` | 50k: Serial, RTC, Async, PAINT |
| `stack_blocks_small/pi05/` | 50k: Serial, RTC, Async, PAINT |

The previous screwdriver, red-object, pass-ball and offline experiment entries
were removed on 2026-10-09. Recorded rollouts and model/component implementations
are retained. Diagnostic scripts requiring an experiment now take an explicit
`--config`, `--infra-config` or `--experiment` instead of a retired default.

New Pi05 tasks share 30 Hz model actions, 100 Hz control, 50-step predictions,
10 denoising steps, seed 0 and Direct. Serial/Async use a 16-step prefix, RTC
uses the aligned min-16 preset, and PAINT uses its existing 12-step preset.
These are starting settings, not an equal-execution-budget claim or a new
hardware validation. Each entry declares ten layouts and three repeats;
layout references and task-specific success criteria still need to be frozen.
New tasks use the generic manual completion label until a task rubric is defined.

Policy recipes hold checkpoint and matching normalization paths. Coin uses the
existing local Pi05 checkpoint; block recipes select the verified export on the
mounted T7 drive. The drive must be mounted at the path declared in the recipe.
Xiaomi/LingBot coin and block runtime entries are not yet bound; their new
checkpoint contracts and artifact completeness need validation first. Do not
substitute a bottle checkpoint or point inference at `.partial` downloads.

Example (the same experiment config is used for model server and runtime):

```bash
envs/yam/.venv/bin/python -m manimux serve \
  --config manimux/configs/experiments/coin/pi05/yam_pi05_rtc_joint_step50000.yaml
```

The CLI manages runtime/RoboGUI; start the model and camera services using the
existing deployment guide. Configuration checking does not launch any services.
Outputs are `data/experiments/pi05-<task>-joint-step50000/<method>/session-*/`.

## Model-side algorithm support on YAM

The current experiment register contains four YAM tasks: coin, block stacking,
rotating clock and bottles. Each task has paired metric/PRM tables with a single
`Method` column containing the model and inference method. Existing small-block
configs are an alternative dataset setting, not a separate task or a pooled cohort.

Use the maintained [sampling matrix](../../../.agents/skills/manimux-inference/references/inference-matrix.md)
for current implementation support and conditions; the older capability summary
is superseded. The seven methods are Serial, Async, ACT temporal ensemble, RTC,
PAINT, AAC and AutoHorizon. DP PAINT/AutoHorizon are not applicable. A table row
is a planned comparison, not evidence of a ready checkpoint or successful robot run.

Shared values stay in policy/inference/embodiment YAMLs through `config:`;
station files hold device bindings. The actual loader is `manimux.cli.load_config`.
