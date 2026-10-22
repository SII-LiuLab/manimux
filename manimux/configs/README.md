# Configuration map

This directory contains reusable components, task definitions and experiment choices.
Use this page to locate the file to edit; use the
[user guide](../../guideline.html#experiment-lists) for setup, YAML examples and launch commands.

## Where settings belong

```text
manimux/configs/
├── lists/<study>.yaml                        # choices, trial plan, recording root
├── tasks/<task>.yaml                         # prompt, layouts, inline scoring, optional pose
├── experiments/<model-checkpoint>/
│   ├── _base_.yaml                           # shared choices, only when needed
│   └── <task>/<variant>.yaml                 # specific model/task/algorithm settings
├── embodiment/{robot,sensor,initial_poses}/  # hardware, control profiles, named poses
├── agents/                                  # shared agent recipes and motion profiles
├── examples/                                # standalone development examples
├── policy/                                  # model artifacts, serving and action contracts
├── inference/                               # algorithm base settings
├── executor/                                # execution settings
├── robogui/                                 # display and preview settings
└── local/                                   # private station bindings and local configurations
```

| Change | Owning file / field |
| --- | --- |
| Model instruction | Task `run.task_prompt` |
| Per-rollout control-step limit | Task `run.max_control_steps`; inherited by all models and methods |
| Agent planner, motion tools and runtime selection | `agents/<framework>.yaml` references one experiment and motion profile; experiments own policy and robot selection |
| Available layouts and reference requirement | Task `run.experiment_template` |
| Success criteria and partial scores | Task inline `evaluation`; no separate scoring file |
| Task starting pose | Task `initial_pose`, referencing a named embodiment preset; variants can override it |
| Selectable starting poses in RoboGUI | Robot assembly `initial_poses` references the preset library; a task's explicit `initial_pose.config` takes precedence |
| Model, algorithm, execution window or other comparison parameters | Experiment variant and its selected component recipes |
| Tasks/methods to run, layout subset and repetitions | Config list entries and `trials` |
| Recording root | Config list `recording.root` |
| CAN/IP, camera serials, service endpoints and installed artifact paths | Private `local/station.yaml` |

Examples: [coin task](tasks/coin.yaml), [medium-block task](tasks/stack_blocks_medium.yaml),
[shared Pi05 settings](experiments/pi05-latest300ep-step30000/_base_.yaml),
[RTC variant](experiments/pi05-latest300ep-step30000/coin/rtc32_beta10.yaml),
[current config list](lists/pi05-latest300ep-step30000.yaml).

## Composition and storage boundaries

- A config list selects variants; variants inherit shared settings and tasks through `_base_`.
- Later bases override earlier ones, then the current file overrides its bases. Mappings merge; lists replace. Inheritance is limited to four files per chain.
- YAML references resolve beside the file declaring them. Component `config` references select recipes; inline values override them.
- `experiment.task` is a storage identity; `run.task_prompt` is the instruction sent to the model.
- Use `run --config-list` for formal studies. Listed variants must not set `run.output_dir`; station `paths.output_dir` cannot override study storage.
- Use `run --config` to try any concrete variant directly. Only Prepare Run is available; records default to `data/trials/<task>/<variant>/session-…`. Configs without experiment identity use a filename plus path hash. `--output-dir` overrides the trial root; otherwise `run.output_dir` (default `data`) supplies its parent. Formal progress and history are not loaded.
- Camera and generic policy services accept the same `--config` or `--config-list` as runtime. Components such as a raw camera recipe are not complete runtime configs.
- Recordings go to `<recording.root>/<list.id>/<experiment.task>/<experiment.variant>/session-…/rollout-…`. Relative roots resolve beside the config list YAML.
- Repeats create recordings, not copied YAMLs. Changed scientific conditions require a new variant ID when records already exist. Historical recordings retain their saved settings and scores.

Only `experiments/pi05-latest300ep-step30000/` and its config list are published study
examples in this checkout. Other experiment groups and config lists are ignored by default.
Intentional publication can update both allowlists in the root `.gitignore` or explicitly
use `git add -f`; include required dependencies. Station secrets, weights, training jobs
and recordings remain private.

## Further reading

- [Configuration, config lists and storage](../../guideline.html#experiment-lists): complete setup and composition examples.
- [Agent integration](../../guideline.html#agent-integrations): RPent recipes, path resolution and matching runtime configuration.
- [Tasks and scoring](../../guideline.html#experiments): explicit score options for every task, plus the trial workflow.
- [Launch commands](../../guideline.html#latest300ep-launch): matching camera, model, RoboGUI and runtime services.
- [Switching and reload rules](../../guideline.html#experiment-switch-boundaries): compatible entries and service restart boundaries.
- [Policy recipes](policy/README.md) and [camera recipes](embodiment/sensor/cameras/README.md): component-specific ownership.
- [Experiment skill](../../.agents/skills/manimux-experiments/SKILL.md): agent workflow for eliciting task details, writing files and validating them.
- [Station setup skill](../../.agents/skills/manimux-station-setup/SKILL.md): agent workflow for local hardware and service bindings.
