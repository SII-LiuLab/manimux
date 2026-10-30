# Experiment variants

Only the `example-*` studies are published; they are also the runtime examples:

| Study | Base | Variants |
| --- | --- | --- |
| [`example-yam-pi05/`](example-yam-pi05/README.md) | Pi05 through XPolicyLab, YAM, Serial 30 Hz | `put_bottles/{serial,rtc,async10,rpent}` |
| [`example-yam-dream1/`](example-yam-dream1/README.md) | Hosted Dream-1 API, YAM, Serial 30 Hz | `put_bottles/{serial,rpent}` |
| [`example-franka/`](example-franka/_base_.yaml) | Franka FR3, XPolicyLab joint policy, read-only | `pick_yellow_block/{pi05_droid,rpent}` |
| [`example-rpent-synthetic/`](example-rpent-synthetic/_base_.yaml) | Synthetic robot and tool policy, no hardware | `demo/{rpent,rpent_gui,rpent_gui_debug,rpent_gui_primitives}` |

Your own experiment directories and their config list entries are local and ignored by
Git. Creating a local study does not publish it; explicitly update the allowlist in
`.gitignore` when another study is intended for sharing.

Store variants under `<study>/<task>/<variant>.yaml`. The study `_base_.yaml` is a
runnable baseline (robot, cameras, policy client and model recipe, timing, RoboGUI); each
variant inherits it and a task from `../../tasks/`, then sets only what its method
changes, as the example variants do. Component references stay in
`policy/`, `inference/`, `embodiment/` and `executor/`. Scoring lives inline
in the shared task YAML.

The task YAML also owns `run.max_control_steps`. Model bases and concrete methods
inherit that limit; do not copy it into each variant. Chunk execution windows
remain inference settings and are independent of the total rollout budget.

Variants used in formal studies declare `experiment.task` and `experiment.variant`,
without an output root or repeat budget. A file in `../lists/` selects variants,
trials and `recording.root`; launch all services with that same `--config-list`.

The same variant can be tried directly with `run --config <variant>`. Camera and
generic policy services also accept that `--config`. This route offers Prepare Run
only and saves separate trial records; it never consumes the study repeat budget.
A new standalone runtime config needs no experiment identity, layouts or scoring.

Example lists: [`../lists/example-yam-pi05.yaml`](../lists/example-yam-pi05.yaml),
[`../lists/example-yam-dream1.yaml`](../lists/example-yam-dream1.yaml).
See the [complete protocol and commands](../../../guideline.html#experiment-lists).

Raw historical sessions retain their saved parameters and paths. Their config list
`history` inputs require compatible conditions. A change only to the stop budget
retains progress without renaming variants or rewriting sessions; each session's
saved budget and condition fingerprint remain authoritative for analysis.
Sessions containing only a startup manifest and no `rollout-*` records do not
block a corrected configuration. Their manifests remain unchanged; even a partial
rollout keeps the normal compatibility check.
