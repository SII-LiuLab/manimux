# Configuration map

This directory contains reusable components, task definitions and experiment choices.
Use this page to locate the file to edit; use the
[user guide](../../guideline.html#experiment-lists) for setup, YAML examples and launch commands.

## Where settings belong

```text
manimux/configs/
├── lists/<study>.yaml                        # choices, trial plan, recording root
├── tasks/<task>.yaml                         # prompt, layouts, inline scoring, optional pose
├── experiments/<study>/
│   ├── _base_.yaml                           # runnable baseline: robot, cameras, policy, timing
│   └── <task>/<variant>.yaml                 # what this method changes
├── embodiment/{robot,sensor,initial_poses}/  # hardware, control profiles, named poses
├── agents/                                  # RPent recipes and motion templates
├── policy/                                  # model artifacts, serving and action contracts
├── inference/                               # algorithm base settings
├── executor/                                # execution settings
├── robogui/                                 # display and preview settings
└── local/                                   # private station bindings and local configurations
```

## Worked example: how the files fit together

The published `example-*` studies show every layer; there is no separate examples folder.
Follow one entry of `example-yam-pi05` from top to bottom:

```text
lists/example-yam-pi05.yaml                         1. study entry
  experiments:
  - id: put_bottles_async10
    config: ../experiments/example-yam-pi05/put_bottles/async10.yaml
    trials: {repeats: 3}
        │
        ▼
experiments/example-yam-pi05/put_bottles/async10.yaml   2. one method
  _base_:
  - ../_base_.yaml
  - ../../../tasks/put_bottles.yaml
  policy: {action_dt_s: 0.1}                        # only what this method changes
  inference: {algorithm: async, request_trigger: progress, chunk_policy_steps: 28}
  executor: {config: ../../../executor/yam_smooth_standard.yaml, type: smooth}
  experiment: {task: put_bottles, variant: async28_slow10_smooth}
        │
        ├─▶ experiments/example-yam-pi05/_base_.yaml      3. runnable baseline
        │     robot, cameras, policy client, Serial 30 Hz, RoboGUI, recording
        │     camera_server: {config: ../../embodiment/sensor/cameras/realsense_3_views.yaml}
        │     policy_server: {config: ../../policy/pi05/yam/put-bottles/joint-step30000.yaml}
        │           │
        │           └─▶ policy/pi05/yam/put-bottles/joint-step30000.yaml   4. model recipe
        │                 checkpoint and normalization below paths.checkpoints,
        │                 horizon, action contract, backend identity
        │
        └─▶ tasks/put_bottles.yaml                   5. task
              prompt, max steps, layouts, scoring rubric

local/station.yaml (private, not in Git)             6. this station
  CAN channels, camera serials, service endpoints, paths.checkpoints
```

The other entries differ only in layer 2: `serial.yaml` adds nothing, and `rtc.yaml` sets
`inference.algorithm: rtc`. `rpent.yaml` adds `agent: {config: ../../../agents/rpent_pi05_yam.yaml}`;
that RPent recipe adds the planner and points to the private motion profile
`local/rpent_yam_motion.yaml`.
To try one variant, run `manimux up --config <variant>`; for scored, counted trials, run
`manimux up --config-list lists/example-yam-pi05.yaml`. Build your own study the same way:
a private list and variants that inherit a shared `_base_.yaml` and a task.

## Where each setting lives

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

Examples: [put-bottles task](tasks/put_bottles.yaml),
[study base](experiments/example-yam-pi05/_base_.yaml),
[RTC variant](experiments/example-yam-pi05/put_bottles/rtc.yaml),
[example config list](lists/example-yam-pi05.yaml).

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

Only the `example-*` studies and lists are published:

| Study | Shows |
| --- | --- |
| [`example-yam-pi05`](experiments/example-yam-pi05/README.md) | Pi05 through XPolicyLab: Serial, RTC, Async 10 Hz + Smooth, RPent ([list](lists/example-yam-pi05.yaml)) |
| [`example-yam-dream1`](experiments/example-yam-dream1/README.md) | Hosted Dream-1 API: Serial, RPent ([list](lists/example-yam-dream1.yaml)) |
| [`example-franka`](experiments/example-franka/_base_.yaml) | Franka FR3: Pi05-DROID through XPolicyLab, RPent |
| [`example-rpent-synthetic`](experiments/example-rpent-synthetic/_base_.yaml) | Hardware-free dispatch and RPent GUI checks |

Your own experiment groups and config lists are ignored by default.
Intentional publication can update both allowlists in the root `.gitignore` or explicitly
use `git add -f`; include required dependencies. Station secrets, weights, training jobs
and recordings remain private.

## Further reading

- [Configuration, config lists and storage](../../guideline.html#experiment-lists): complete setup and composition examples.
- [Agent integration](../../guideline.html#agent-integrations): RPent recipes, path resolution and matching runtime configuration.
- [Tasks and scoring](../../guideline.html#experiments): explicit score options for every task, plus the trial workflow.
- [Launch commands](../../guideline.html#example-study): matching camera, model, RoboGUI and runtime services.
- [Switching and reload rules](../../guideline.html#experiment-switch-boundaries): compatible entries and service restart boundaries.
- [Policy recipes](policy/README.md) and [camera recipes](embodiment/sensor/cameras/README.md): component-specific ownership.
- [Experiment skill](../../skills/manimux-experiments/SKILL.md): agent workflow for eliciting task details, writing files and validating them.
- [Station setup skill](../../skills/manimux-station-setup/SKILL.md): agent workflow for local hardware and service bindings.
